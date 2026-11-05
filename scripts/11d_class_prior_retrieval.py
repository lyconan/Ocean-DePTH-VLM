#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11d_class_prior_retrieval.py — 门/纲级先验过滤版检索评测 (三层评测·检索层增强)

问题: BioCLIP 检索在全库 (2625 形态种) 上做, strict@top1 仅 ~17%。
     YOLO 双路已给出纲级预测 (34 纲, 剔 Other acc 94.1%),
     用它把检索候选限制在同纲形态种子集内, 识别率能提升多少?

数据 (零重推理/零重嵌入):
  匹配框:  .11c_yolo_cache_960aug.json + .11c_yolo_cache_1920.json
           -> merge_detections(conf0.05, nms0.55) -> match_dets (与 11c2 完全同口径)
  嵌入:    .11c2_e2e_cache.json (6337 框)
  检索:    Neo4j entryimage_embeddings, K=500 大候选池 -> 每形态种最高分
           (缓存 .11d_raw_cache.json, 每框存 top-100 原始排名)

三档对比 (strict/species/genus @top1/3/5):
  A. baseline    全库 top-k (不过滤, 应与 11c2 一致)
  B. yolo_prior  候选限 YOLO 预测纲 (真实管线行为; 空集回退全库)
  C. oracle_gt   候选限 GT 纲 (先验完美时的上界)
另有 B 按 YOLO 纲对错分桶, 看先验错误时的失败模式与回退率。

用法: docker exec -e PYTHONUNBUFFERED=1 GPU_USFM_Paper python scripts/11d_class_prior_retrieval.py
"""
import json
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE + "/scripts")

from neo4j import GraphDatabase

from utils.yolo_infer import merge_detections

ES_DIR = os.path.join(BASE, "data/processed/eval_subset")
SUBSET_JSON = os.path.join(ES_DIR, "fathomnet_test_subset.json")
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")
C_AUG = os.path.join(ES_DIR, ".11c_yolo_cache_960aug.json")
C_1920 = os.path.join(ES_DIR, ".11c_yolo_cache_1920.json")
E2E_CACHE = os.path.join(ES_DIR, ".11c2_e2e_cache.json")
RAW_CACHE = os.path.join(ES_DIR, ".11d_raw_cache.json")
OUT_JSON = os.path.join(ES_DIR, "11d_class_prior_report.json")
VAL_DIR = os.environ.get(
    "DEPTH_VAL_DIR", "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password")
K_QUERY = 500      # Neo4j 向量检索候选池
RAW_TOP = 100      # 每框缓存的原始每形态种最高分条数
TOPK = 10          # 评测 top-k 上限
CONF = 0.05
IOU_THR = 0.5
SAVE_EVERY = 1000

OBJ_BLOCK_RE = re.compile(r"<object>(.*?)</object>", re.S)
NAME_RE = re.compile(r"<name>([^<]+)</name>")
BOX_RE = re.compile(
    r"<xmin>\s*(\d+)\s*</xmin>\s*<ymin>\s*(\d+)\s*</ymin>\s*"
    r"<xmax>\s*(\d+)\s*</xmax>\s*<ymax>\s*(\d+)\s*</ymax>", re.S)

from utils.yolo34_classes import YOLO_CLASSES  # 34 纲类别表 (单一事实来源, 见 utils/yolo34_classes.py)


def parse_objects(txt):
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


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def load_gts():
    subset = json.load(open(SUBSET_JSON, encoding="utf-8"))
    gts = {}
    for r in subset["records"]:
        uuid = os.path.basename(r["image"])[:-4]
        xml = os.path.join(VAL_DIR, r["image"].replace(".png", ".xml"))
        try:
            txt = open(xml, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        items = []
        for name, x0, y0, x1, y1 in parse_objects(txt):
            mid = r["concepts"].get(name)
            if mid:
                items.append((mid, (int(x0), int(y0), int(x1), int(y1))))
        if items:
            gts[uuid] = (os.path.join(VAL_DIR, r["image"]), items)
    return gts


def match_dets(gts, det_getter):
    """贪心 IoU>=0.5 匹配 (与 11c2 完全一致)。hits[(u, gi)] = (det, gt_mid)"""
    hits = {}
    for u, (_, items) in gts.items():
        dets = det_getter(u)
        used = set()
        for gi, (mid, box) in enumerate(items):
            best, bi = 0.0, -1
            for i, d in enumerate(dets):
                if i in used:
                    continue
                v = iou(box, d["box"])
                if v > best:
                    best, bi = v, i
            if best >= IOU_THR:
                used.add(bi)
                hits[(u, gi)] = (dets[bi], mid)
    return hits


def main():
    gts = load_gts()
    n_gt = sum(len(v[1]) for v in gts.values())
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    meta = {m["morphospecies_id"]: m for m in msp}
    print(f"评测图 {len(gts)}, 真值 bbox {n_gt}, 形态种 {len(meta)}")

    # 形态种 -> 34 纲 (与 11c2 GT 口径一致)
    norm_set = {norm(c): c for c in YOLO_CLASSES}
    msp_class = {}
    for mid, m in meta.items():
        lin = m.get("lineage") or {}
        g = norm(m.get("class_name") or lin.get("class") or "Other")
        msp_class[mid] = norm_set.get(g, "Other")
    cls2msps = {}
    for mid, c in msp_class.items():
        cls2msps.setdefault(c, set()).add(mid)
    print("纲内形态种数 (top):", dict(sorted(
        ((c, len(v)) for c, v in cls2msps.items()), key=lambda kv: -kv[1])[:8]))

    c_aug = json.load(open(C_AUG, encoding="utf-8"))
    c_1920 = json.load(open(C_1920, encoding="utf-8"))

    def dual_get(u):
        return merge_detections([c_aug.get(u, []), c_1920.get(u, [])],
                                conf=CONF, iou_nms=0.55)

    hits = match_dets(gts, dual_get)
    print(f"双路匹配框 {len(hits)} (recall {len(hits)/n_gt:.3f})")

    # 嵌入 (复用 11c2 缓存, key 同构: uuid|x0_y0_x1_y1)
    tasks = []  # (raw_key, (u,gi), det, gt_mid)
    for (u, gi), (d, mid) in hits.items():
        x0, y0, x1, y1 = d["box"]
        tasks.append((f"{u}|{x0}_{y0}_{x1}_{y1}", (u, gi), d, mid))
    ec = json.load(open(E2E_CACHE, encoding="utf-8")) if os.path.exists(E2E_CACHE) else {}
    have = sum(1 for t in tasks if ec.get(t[0]))
    print(f"嵌入缓存命中 {have}/{len(tasks)}")
    tasks = [t for t in tasks if ec.get(t[0])]

    # Neo4j 大候选池检索 (每形态种最高分, 存 raw top-100)
    rc = json.load(open(RAW_CACHE, encoding="utf-8")) if os.path.exists(RAW_CACHE) else {}
    todo = [t for t in tasks if t[0] not in rc]
    print(f"待检索 {len(todo)} (raw 缓存 {len(rc)})")
    if todo:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        try:
            driver.verify_connectivity()
            t0 = time.time()
            with driver.session() as s:
                for n, (key, _, _, _) in enumerate(todo):
                    emb = ec[key]
                    rows = s.run("""
                        CALL db.index.vector.queryNodes('entryimage_embeddings', $k, $emb)
                        YIELD node, score
                        RETURN node.msp_id AS mid, score
                    """, k=K_QUERY, emb=emb).data()
                    best = {}
                    for r in rows:
                        best[r["mid"]] = max(best.get(r["mid"], 0), r["score"])
                    rc[key] = [[m, round(sc, 6)] for m, sc in
                                sorted(best.items(), key=lambda kv: -kv[1])[:RAW_TOP]]
                    if (n + 1) % 500 == 0:
                        rate = (n + 1) / (time.time() - t0)
                        print(f"检索 {n+1}/{len(todo)} ({rate:.1f}/s, "
                              f"ETA {(len(todo)-n-1)/max(rate,1e-6)/60:.0f}min)", flush=True)
                        if (n + 1) % SAVE_EVERY == 0:
                            atomic_json(RAW_CACHE, rc)
        finally:
            driver.close()
        atomic_json(RAW_CACHE, rc)
        print(f"raw 缓存已写出 ({len(rc)})")

    # ---- 三档评测 ----
    def hit_at(ranked, gt, k, mode):
        for m, _ in ranked[:k]:
            if mode == "strict" and m == gt:
                return 1
            if mode == "species" and (m == gt or norm(meta[m]["latin"]) == norm(meta[gt]["latin"])):
                return 1
            if mode == "genus":
                g1 = norm(meta[gt]["latin"]).split()[0] if meta[gt]["latin"] else ""
                g2 = norm(meta[m]["latin"]).split()[0] if meta[m]["latin"] else ""
                if m == gt or (g1 and g1 == g2):
                    return 1
        return 0

    def stats_for(entries):
        out = {"n_evaluated": len(entries)}
        for mode in ("strict", "species", "genus"):
            for k in (1, 3, 5):
                out[f"{mode}@top{k}"] = round(
                    sum(hit_at(rk, gt, k, mode) for gt, rk in entries) / max(len(entries), 1), 4)
        return out

    def filtered_rank(raw, prior_cls, gt_mid):
        """按先验纲过滤 raw 列表; 空集回退全库。返回 (ranked_top10, used_fallback, gt_in_candidates)"""
        ok_set = cls2msps.get(prior_cls, set())
        flt = [[m, sc] for m, sc in raw if m in ok_set]
        if not flt:
            return raw[:TOPK], True, gt_mid in {m for m, _ in raw}
        return flt[:TOPK], False, gt_mid in {m for m, _ in flt}

    e_base, e_yolo, e_oracle = [], [], []
    e_ok, e_bad = [], []          # yolo 先验按纲对错分桶
    n_fallback = n_gt_in_cand = 0
    for key, (u, gi), d, mid in tasks:
        raw = rc.get(key)
        if not raw:
            continue
        gt_cls = msp_class[mid]
        # A. baseline
        e_base.append((mid, raw[:TOPK]))
        # B. yolo prior
        rk, fb, gtin = filtered_rank(raw, d["cls"], mid)
        n_fallback += int(fb)
        e_yolo.append((mid, rk))
        (e_ok if d["cls"] == gt_cls else e_bad).append((mid, rk))
        # C. oracle
        rk_o, fb_o, _ = filtered_rank(raw, gt_cls, mid)
        e_oracle.append((mid, rk_o))
        # 候选池里是否本来就有 GT (top-100 全库)
        n_gt_in_cand += int(mid in {m for m, _ in raw})

    result = {
        "config": {"K_query": K_QUERY, "raw_top": RAW_TOP, "topk": TOPK,
                   "conf": CONF, "iou_nms": 0.55, "paths": "dual 960+TTA/1920",
                   "fallback": "过滤后空集 -> 回退全库 top-k"},
        "n_gt_boxes": n_gt,
        "n_boxes_evaluated": len(e_base),
        "gt_in_raw_top100_rate": round(n_gt_in_cand / max(len(e_base), 1), 4),
        "A_baseline_full": stats_for(e_base),
        "B_yolo_prior": {**stats_for(e_yolo),
                         "fallback_rate": round(n_fallback / max(len(e_yolo), 1), 4)},
        "C_oracle_gt_class": stats_for(e_oracle),
        "B_split_cls_correct": {**stats_for(e_ok),
                                "share": round(len(e_ok) / max(len(e_yolo), 1), 4)},
        "B_split_cls_wrong": {**stats_for(e_bad),
                              "share": round(len(e_bad) / max(len(e_yolo), 1), 4)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    atomic_json(OUT_JSON, result)
    print("已写出:", OUT_JSON)


if __name__ == "__main__":
    main()
