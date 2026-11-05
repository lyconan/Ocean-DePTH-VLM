#!/usr/bin/env python3
"""GraphRAG 检索验证脚本 v2.0 — BioCLIP 2.5 + 新图谱 schema

v2.0 变更 (适配 BioCLIP 2.5 迁移后的新知识图谱):
- 嵌入模型: BioCLIP 2.5 (hf-hub:imageomics/bioclip-2.5-vith14, 1024维), 走 utils/bioclip_embed 共享封装
- 向量索引: entryimage_embeddings (每张 EntryImage 图一个 1024 维嵌入, cosine)
- 检索结果按 msp_id 聚合 (每 Morphospecies 保留最高分), 与 10_graphrag_pipeline._multi_view_vector_search 一致
- 图谱丰富: Morphospecies lineage (phylum/class_name/order_name/family_name) + WORMS_MATCHED→WoRMSTaxon + 同目 siblings
- 自检索校验: 查询图本身取自图谱 EntryImage, top-1 msp_id 应与来源条目一致 (验证嵌入/索引/挂接全链路)
- 移除老 insitu/exsitu 双索引与 Morphotype 节点依赖

用法 (在带 GPU + open_clip + HF 缓存的容器内):
    python scripts/10b_graphrag_retrieval_test.py --no-vlm --num-samples 10
    python scripts/10b_graphrag_retrieval_test.py --neo4j-uri bolt://172.17.0.1:7687 --no-vlm
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from neo4j import GraphDatabase

from utils.bioclip_embed import get_bioclip_model, extract_embedding, get_embedding_dim

MSP_JSON = PROJECT_ROOT / "data/processed/atlas_lookup/morphospecies_index.json"


def sample_test_images(num_samples: int) -> list:
    """从 morphospecies_index.json 采样 (msp_id, image_path) 对。"""
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    pairs = []
    for m in msp:
        for rel in m.get("images", []):
            p = PROJECT_ROOT / rel
            if p.exists():
                pairs.append((m["morphospecies_id"], str(p)))
                break
    random.shuffle(pairs)
    return pairs[:num_samples]


def vector_search(session, embedding: list, top_k: int) -> list:
    """entryimage_embeddings 向量检索 + msp 聚合 (每条目保留最高分)。"""
    rows = session.run("""
        CALL db.index.vector.queryNodes('entryimage_embeddings', $k, $embedding)
        YIELD node, score
        MATCH (ms:Morphospecies {morphospecies_id: node.msp_id})
        RETURN ms.morphospecies_id AS msp_id,
               coalesce(ms.display_name, ms.latin) AS name,
               ms.latin AS latin,
               ms.phylum AS phylum,
               ms.class_name AS class_name,
               ms.order_name AS order_name,
               ms.family_name AS family_name,
               node.image_id AS image_id,
               score
        ORDER BY score DESC
    """, k=top_k * 4, embedding=embedding)

    agg = {}
    for r in rows:
        mid = r["msp_id"]
        if mid not in agg or r["score"] > agg[mid]["score"]:
            agg[mid] = dict(r)
    return sorted(agg.values(), key=lambda x: -x["score"])[:top_k]


def enrich_top1(session, top: dict) -> dict:
    """对 top-1 条目做图谱丰富: lineage 已在节点上, 补 WoRMS 对齐 + 同目 siblings。"""
    mid = top["msp_id"]

    wa = session.run("""
        MATCH (ms:Morphospecies {morphospecies_id: $mid})-[w:WORMS_MATCHED]->(wt:WoRMSTaxon)
        RETURN wt.aphia_id AS aphia_id,
               wt.scientific_name AS worms_name,
               wt.rank AS worms_rank,
               w.mapping_type AS mapping_type
        LIMIT 1
    """, mid=mid).single()
    if wa:
        top["worms_alignment"] = dict(wa)

    sibs = session.run("""
        MATCH (ms:Morphospecies {morphospecies_id: $mid})
        MATCH (s:Morphospecies)
        WHERE s.order_name = ms.order_name AND s.morphospecies_id <> $mid
        RETURN coalesce(s.display_name, s.latin) AS sibling_name, s.phylum AS sibling_phylum
        LIMIT 5
    """, mid=mid)
    top["siblings"] = [dict(s) for s in sibs]
    return top


def main():
    ap = argparse.ArgumentParser(description="GraphRAG 检索验证 (BioCLIP 2.5 + 新图谱)")
    ap.add_argument("--num-samples", type=int, default=10, help="测试图片数量")
    ap.add_argument("--top-k", type=int, default=5, help="聚合后保留的条目数")
    ap.add_argument("--neo4j-uri", default="bolt://172.17.0.1:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="password")
    ap.add_argument("--output", default="results/graphrag_retrieval_test.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("DePTH-VLM GraphRAG 检索验证 v2.0 (BioCLIP 2.5 / 1024维 / entryimage_embeddings)")
    print("=" * 70)

    print("\n[1/4] 加载 BioCLIP 2.5 ...")
    model, processor = get_bioclip_model()
    dim = get_embedding_dim(model)
    print(f"  嵌入维度: {dim}")

    print("\n[2/4] 连接 Neo4j ...")
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    with driver.session() as s:
        for label, cql in (
            ("Morphospecies", "MATCH (ms:Morphospecies) RETURN count(ms) AS c"),
            ("EntryImage(含嵌入)", "MATCH (ei:EntryImage) WHERE ei.embedding IS NOT NULL RETURN count(ei) AS c"),
            ("WoRMSTaxon", "MATCH (t:WoRMSTaxon) RETURN count(t) AS c"),
            ("WORMS_MATCHED", "MATCH (:Morphospecies)-[:WORMS_MATCHED]->() RETURN count(*) AS c"),
        ):
            print(f"  {label}: {s.run(cql).single()['c']:,}")

    print(f"\n[3/4] 采样测试图片 ({args.num_samples} 张, 取自图谱挂接图) ...")
    test_pairs = sample_test_images(args.num_samples)
    if not test_pairs:
        print("  [错误] 未找到可用测试图片")
        driver.close()
        return
    print(f"  实际采样: {len(test_pairs)} 张")

    print("\n[4/4] 逐张执行: BioCLIP 2.5 嵌入 → 向量检索 → msp聚合 → 图谱丰富")
    all_results, top1_hits, top5_hits = [], 0, 0
    for i, (gt_mid, img_path) in enumerate(test_pairs):
        print(f"\n--- 样本 {i+1}/{len(test_pairs)} [{gt_mid}]: {Path(img_path).name} ---")
        t0 = time.time()
        emb = extract_embedding(img_path, model, processor)
        t_embed = time.time() - t0

        t1 = time.time()
        with driver.session() as s:
            hits = vector_search(s, emb.tolist(), args.top_k)
            if hits:
                hits[0] = enrich_top1(s, hits[0])
        t_search = time.time() - t1

        top1_ok = bool(hits) and hits[0]["msp_id"] == gt_mid
        top5_ok = any(h["msp_id"] == gt_mid for h in hits)
        top1_hits += top1_ok
        top5_hits += top5_ok

        print(f"  嵌入 {t_embed:.3f}s | 检索+聚合 {t_search:.3f}s | top1={top1_ok} top5={top5_ok}")
        for j, h in enumerate(hits):
            mark = " <=GT" if h["msp_id"] == gt_mid else ""
            print(f"    {j+1}. [{h['score']:.4f}] {h['msp_id']} {h['name']} ({h['phylum']}){mark}")
        top = hits[0] if hits else {}
        if top.get("worms_alignment"):
            wa = top["worms_alignment"]
            print(f"  WoRMS 对齐: {wa['worms_name']} (aphia={wa['aphia_id']}, rank={wa['worms_rank']}, type={wa['mapping_type']})")
        if top.get("siblings"):
            print(f"  同目 siblings ({len(top['siblings'])}): "
                  + ", ".join(s["sibling_name"] or "?" for s in top["siblings"][:3]))

        all_results.append({
            "gt_msp_id": gt_mid,
            "image_path": img_path,
            "embed_time": t_embed,
            "search_time": t_search,
            "top1_correct": top1_ok,
            "top5_correct": top5_ok,
            "hits": hits,
        })

    driver.close()

    n = len(all_results)
    print("\n" + "=" * 70)
    print("验证结果汇总")
    print("=" * 70)
    print(f"测试样本数:     {n}")
    print(f"自检索 Top-1:   {top1_hits}/{n} ({top1_hits/n*100:.1f}%)" if n else "")
    print(f"自检索 Top-5:   {top5_hits}/{n} ({top5_hits/n*100:.1f}%)" if n else "")
    if n:
        print(f"平均嵌入耗时:   {sum(r['embed_time'] for r in all_results)/n:.3f}s")
        print(f"平均检索耗时:   {sum(r['search_time'] for r in all_results)/n:.3f}s")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": {"n": n, "top1": top1_hits, "top5": top5_hits},
                   "results": all_results}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存: {out}")
    print("\nGraphRAG 检索验证完成!")


if __name__ == "__main__":
    main()
