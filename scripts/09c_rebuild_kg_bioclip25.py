#!/usr/bin/env python3
"""知识图谱重建 — 2,641 Morphospecies + 27,908 EntryImage (BioCLIP 2.5, 1024维) (v1.0)

用户决策 (2026-09-20):
- 完全替换: 删除老的 185 个 APSMA Morphotype 节点及全部旧关系/嵌入
- 每张图都存嵌入: 每图一个 EntryImage 节点, 向量索引建在 EntryImage.embedding 上

流程:
  1. drop 旧 512 维向量索引 (morphotype_exsitu_embeddings / morphotype_insitu_embeddings)
  2. DETACH DELETE 全部 (:Morphotype) 老节点
  3. 建 2,641 (:Morphospecies) 节点 (MERGE on morphospecies_id, 反范式化 lineage 属性)
  4. 建 27,908 (:EntryImage) 节点 + (:Morphospecies)-[:HAS_IMAGE]->(:EntryImage)
     embedding = BioCLIP 2.5 1024维 (fp16->float)
  5. (:Morphospecies)-[:WORMS_MATCHED]->(:WoRMSTaxon {aphia_id})
  6. 建 1024 维 cosine 向量索引 entryimage_embeddings
  7. QA 审计

用法:
  python scripts/09c_rebuild_kg_bioclip25.py                 # 完整重建
  python scripts/09c_rebuild_kg_bioclip25.py --dry-run       # 只统计不写入
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

BASE = Path(__file__).resolve().parent.parent
MSP_JSON = BASE / "data/processed/atlas_lookup/morphospecies_index.json"
EMB_DIR = BASE / "data/processed/bioclip25_embeddings"
NUM_SHARDS = 2
BATCH = 50

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

OLD_VECTOR_INDEXES = ["morphotype_exsitu_embeddings", "morphotype_insitu_embeddings"]
NEW_VECTOR_INDEX = "entryimage_embeddings"


def load_msp_records():
    with open(MSP_JSON, encoding="utf-8") as f:
        idx = json.load(f)
    return idx["morphospecies"]


def load_embeddings():
    """加载全部分片嵌入 -> {image_rel_path: np.float32 array}"""
    embs = {}
    for i in range(NUM_SHARDS):
        npz = EMB_DIR / f"emb_shard{i}.npz"
        if not npz.exists():
            print(f"警告: {npz} 不存在")
            continue
        data = np.load(npz, allow_pickle=False)
        for p, e in zip(data["image_paths"].tolist(), data["embeddings"]):
            embs[p] = e.astype(np.float32)
    print(f"加载嵌入: {len(embs)} 张")
    return embs


def q(session, cypher, **params):
    return session.run(cypher, **params).data()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    msp_recs = load_msp_records()
    embs = load_embeddings()
    dim = len(next(iter(embs.values()))) if embs else 1024
    print(f"条目: {len(msp_recs)}, 嵌入: {len(embs)}, dim={dim}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            # ---- 1. QA 前置统计 ----
            stats = q(session, "MATCH (m:Morphotype) RETURN count(m) AS c")[0]["c"]
            print(f"现状: Morphotype={stats}")

            if args.dry_run:
                no_img = [m for m in msp_recs if not m.get("images")]
                have_emb = sum(1 for m in msp_recs for p in m.get("images", []) if p in embs)
                print(f"[dry-run] 将删除 {stats} 个 Morphotype 节点")
                print(f"[dry-run] 将创建 {len(msp_recs)} Morphospecies ({len(no_img)} 无图)")
                print(f"[dry-run] 将创建 {have_emb} EntryImage 带嵌入")
                return

            # ---- 2. drop 旧向量索引 ----
            for idx_name in OLD_VECTOR_INDEXES:
                q(session, f"DROP INDEX {idx_name} IF EXISTS")
                print(f"已 drop 索引 {idx_name}")

            # ---- 3. 删除老 Morphotype ----
            t0 = time.time()
            q(session, "MATCH (m:Morphotype) DETACH DELETE m")
            print(f"已删除 {stats} 个老 Morphotype 节点 ({time.time()-t0:.0f}s)")

            # ---- 4. 建 Morphospecies 节点 ----
            t0 = time.time()
            rows = []
            for m in msp_recs:
                lineage = m.get("lineage") or {}
                worms = m.get("worms") or {}
                rows.append({
                    "morphospecies_id": m["morphospecies_id"],
                    "latin": m.get("latin", ""),
                    "display_name": m.get("display_name", ""),
                    "name_cn": m.get("name_cn", ""),
                    "rank": m.get("rank", ""),
                    "rank_cn": m.get("rank_cn", ""),
                    "suffix": m.get("suffix", ""),
                    "phylum": lineage.get("phylum", ""),
                    "class_name": lineage.get("class", ""),
                    "order_name": lineage.get("order", ""),
                    "family_name": lineage.get("family", ""),
                    "genus": lineage.get("genus", ""),
                    "aphia_id": worms.get("valid_aphia_id") or worms.get("aphia_id"),
                    "worms_url": worms.get("url", ""),
                    "image_count": m.get("image_count", len(m.get("images", []))),
                    "sources": json.dumps(
                        sorted({(e.get("source", "") if isinstance(e, dict) else str(e))
                                for e in m.get("entries", [])}),
                        ensure_ascii=False),
                })
            for b0 in range(0, len(rows), BATCH):
                batch = rows[b0:b0 + BATCH]
                q(session, """
                    UNWIND $rows AS row
                    MERGE (ms:Morphospecies {morphospecies_id: row.morphospecies_id})
                    SET ms += row
                """, rows=batch)
            print(f"已创建 {len(rows)} Morphospecies ({time.time()-t0:.0f}s)")

            # ---- 5. 建 EntryImage + HAS_IMAGE (带嵌入) ----
            t0 = time.time()
            img_rows = []
            for m in msp_recs:
                for rel in m.get("images", []):
                    emb = embs.get(rel)
                    if emb is None:
                        continue
                    img_rows.append({
                        "msp_id": m["morphospecies_id"],
                        "image_id": rel,
                        "embedding": emb.tolist(),
                    })
            for b0 in range(0, len(img_rows), BATCH):
                batch = img_rows[b0:b0 + BATCH]
                q(session, """
                    UNWIND $rows AS row
                    MATCH (ms:Morphospecies {morphospecies_id: row.msp_id})
                    MERGE (ei:EntryImage {image_id: row.image_id})
                    SET ei.msp_id = row.msp_id,
                        ei.embedding = row.embedding
                    MERGE (ms)-[:HAS_IMAGE]->(ei)
                """, rows=batch)
                if (b0 // BATCH) % 20 == 0:
                    print(f"  EntryImage {b0+len(batch)}/{len(img_rows)} ({time.time()-t0:.0f}s)")
            print(f"已创建 {len(img_rows)} EntryImage ({time.time()-t0:.0f}s)")

            # ---- 6. WORMS_MATCHED 关系 ----
            t0 = time.time()
            link_rows = [{"msp_id": r["morphospecies_id"], "aphia_id": r["aphia_id"]}
                         for r in rows if r["aphia_id"]]
            for b0 in range(0, len(link_rows), BATCH):
                batch = link_rows[b0:b0 + BATCH]
                q(session, """
                    UNWIND $rows AS row
                    MATCH (ms:Morphospecies {morphospecies_id: row.msp_id})
                    MATCH (wt:WoRMSTaxon {aphia_id: row.aphia_id})
                    MERGE (ms)-[:WORMS_MATCHED]->(wt)
                """, rows=batch)
            print(f"已创建 WORMS_MATCHED {len(link_rows)} 条 ({time.time()-t0:.0f}s)")

            # ---- 7. 新向量索引 (1024维 cosine) ----
            q(session, f"""
                CREATE VECTOR INDEX {NEW_VECTOR_INDEX} IF NOT EXISTS
                FOR (ei:EntryImage) ON (ei.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
            print(f"已创建向量索引 {NEW_VECTOR_INDEX} (dim={dim}, cosine)")

            # ---- 8. QA 审计 ----
            print("\n===== QA 审计 =====")
            for cy, label in [
                ("MATCH (ms:Morphospecies) RETURN count(ms) AS c", "Morphospecies"),
                ("MATCH (ei:EntryImage) RETURN count(ei) AS c", "EntryImage"),
                ("MATCH (:Morphospecies)-[:HAS_IMAGE]->(:EntryImage) RETURN count(*) AS c", "HAS_IMAGE"),
                ("MATCH (:Morphospecies)-[:WORMS_MATCHED]->(:WoRMSTaxon) RETURN count(*) AS c", "WORMS_MATCHED"),
                ("MATCH (m:Morphotype) RETURN count(m) AS c", "老Morphotype(应为0)"),
                ("MATCH (ms:Morphospecies) WHERE NOT (ms)-[:HAS_IMAGE]->() RETURN count(ms) AS c", "无图条目"),
            ]:
                print(f"  {label}: {q(session, cy)[0]['c']}")
            for r in q(session, "SHOW INDEXES YIELD name, type, state WHERE type='VECTOR' RETURN name, state"):
                print(f"  向量索引: {r['name']} ({r['state']})")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
