#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11b_eval_bioclip25_retrieval.py — BioCLIP 2.5 形态种检索评测 (三层评测·第二层)

对验证子集 (2,861 图 / 886 FathomNet 类):
  真值 bbox crop -> BioCLIP 2.5 嵌入 -> Neo4j entryimage_embeddings top-k 检索
  -> 形态种级命中统计

指标:
  strict: top-k 含真值 morphospecies_id
  loose_species: top-k 含同物种学名 (跨同种条目)
  loose_genus:   top-k 含同属条目

断点续跑: data/processed/eval_subset/.11b_cache.json (嵌入按 crop 键缓存)
用法: docker exec -e PYTHONUNBUFFERED=1 GPU_USFM_Paper python scripts/11b_eval_bioclip25_retrieval.py
"""
import json
import os
import re
import sys
import time
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE + "/scripts")

from PIL import Image
from neo4j import GraphDatabase

from utils.bioclip_embed import get_bioclip_model, extract_embeddings_batch

SUBSET_JSON = os.path.join(BASE, "data/processed/eval_subset/fathomnet_test_subset.json")
CACHE_JSON = os.path.join(BASE, "data/processed/eval_subset/.11b_cache.json")
OUT_JSON = os.path.join(BASE, "data/processed/eval_subset/11b_retrieval_report.json")
VAL_DIR = "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet"
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password")
TOPK = 10
BATCH = 32

OBJ_BLOCK_RE = re.compile(r"<object>(.*?)</object>", re.S)
NAME_RE = re.compile(r"<name>([^<]+)</name>")
BOX_RE = re.compile(
    r"<xmin>\s*(\d+)\s*</xmin>\s*<ymin>\s*(\d+)\s*</ymin>\s*"
    r"<xmax>\s*(\d+)\s*</xmax>\s*<ymax>\s*(\d+)\s*</ymax>", re.S)


def parse_objects(txt):
    """从 VOC XML 提取 (name, x0, y0, x1, y1), 兼容标签间任意空白。"""
    out = []
    for blk in OBJ_BLOCK_RE.findall(txt):
        nm = NAME_RE.search(blk)
        bx = BOX_RE.search(blk)
        if nm and bx:
            out.append((nm.group(1), bx.group(1), bx.group(2), bx.group(3), bx.group(4)))
    return out


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def norm(s):
    return " ".join(s.split()).lower()


def load_cache():
    if os.path.exists(CACHE_JSON):
        try:
            return json.load(open(CACHE_JSON, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def main():
    subset = json.load(open(SUBSET_JSON, encoding="utf-8"))
    records = subset["records"]

    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    meta = {m["morphospecies_id"]: m for m in msp}

    # ---- 收集全部 crop 任务: (crop_key, img_path, box, gt_mid) ----
    tasks = []
    for r in records:
        uuid = os.path.basename(r["image"])[:-4]
        xml = os.path.join(VAL_DIR, r["image"].replace(".png", ".xml"))
        try:
            txt = open(xml, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        gt_concepts = r["concepts"]  # concept -> msp_id
        for name, x0, y0, x1, y1 in parse_objects(txt):
            mid = gt_concepts.get(name)
            if mid is None:
                # 纯图应全部可映射; 保险跳过
                continue
            tasks.append((f"{uuid}|{name}|{x0},{y0},{x1},{y1}",
                          os.path.join(VAL_DIR, r["image"]),
                          (int(x0), int(y0), int(x1), int(y1)), mid))
    print(f"评测 crop 任务: {len(tasks)} (来自 {len(records)} 图)")

    cache = load_cache()
    todo = [t for t in tasks if t[0] not in cache]
    print(f"缓存命中 {len(tasks)-len(todo)}, 待提取 {len(todo)}")
    if todo:
        model, processor = get_bioclip_model("cuda:0")
        t0, done = time.time(), 0
        for b0 in range(0, len(todo), BATCH):
            chunk = todo[b0:b0 + BATCH]
            imgs = []
            for _, ipath, box, _ in chunk:
                try:
                    im = Image.open(ipath).convert("RGB").crop(box)
                    imgs.append(im)
                except (OSError, ValueError):
                    imgs.append(None)
            embs = extract_embeddings_batch(imgs, model, processor, batch_size=BATCH, device="cuda:0")
            for t, im, e in zip(chunk, imgs, embs):
                cache[t[0]] = None if im is None or e is None else [round(float(x), 6) for x in e]
            done += len(chunk)
            if done % 2000 < BATCH or done == len(todo):
                atomic_json(CACHE_JSON, cache)
                rate = done / (time.time() - t0)
                print(f"嵌入 {done}/{len(todo)} ({rate:.1f}/s, ETA {(len(todo)-done)/max(rate,1e-6)/60:.0f}min)", flush=True)
        atomic_json(CACHE_JSON, cache)

    # ---- 检索 + 打分 ----
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    results = []
    try:
        driver.verify_connectivity()
        with driver.session() as s:
            t0 = time.time()
            for i, (key, _, _, gt) in enumerate(tasks):
                emb = cache.get(key)
                if not emb:
                    continue
                rows = s.run("""
                    CALL db.index.vector.queryNodes('entryimage_embeddings', $k, $emb)
                    YIELD node, score
                    MATCH (ms:Morphospecies {morphospecies_id: node.msp_id})
                    RETURN ms.morphospecies_id AS id, node.msp_id AS mid, score
                """, k=TOPK, emb=emb).data()
                # 每 msp 保留最高分
                best = {}
                for r in rows:
                    best[r["mid"]] = max(best.get(r["mid"], 0), r["score"])
                ranked = sorted(best.items(), key=lambda kv: -kv[1])[:TOPK]
                results.append({"key": key, "gt": gt,
                                "ranked": [[m, sc] for m, sc in ranked]})
                if (i + 1) % 1000 == 0:
                    print(f"检索 {i+1}/{len(tasks)} ({(i+1)/(time.time()-t0):.1f}/s)", flush=True)
    finally:
        driver.close()

    # ---- 指标 ----
    def hit_at(ranked, gt, k, mode):
        for m, _ in ranked[:k]:
            if mode == "strict" and m == gt:
                return 1
            if mode == "species":
                if m == gt or norm(meta[m]["latin"]) == norm(meta[gt]["latin"]):
                    return 1
            if mode == "genus":
                g1 = norm(meta[gt]["latin"]).split()[0] if meta[gt]["latin"] else ""
                g2 = norm(meta[m]["latin"]).split()[0] if meta[m]["latin"] else ""
                if m == gt or (g1 and g1 == g2):
                    return 1
        return 0

    stats = {}
    for mode in ("strict", "species", "genus"):
        for k in (1, 3, 5):
            key = f"{mode}@top{k}"
            ok = sum(hit_at(r["ranked"], r["gt"], k, mode) for r in results)
            stats[key] = round(ok / max(len(results), 1), 4)
    stats["n_evaluated"] = len(results)
    stats["n_gt_classes"] = len({r["gt"] for r in results})
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    atomic_json(OUT_JSON, {"stats": stats,
                           "results": results})
    print(f"已写出: {OUT_JSON}")


if __name__ == "__main__":
    main()
