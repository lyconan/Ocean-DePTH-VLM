#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11c2_dual_e2e.py — 双路并集检测 (960+TTA ∪ 1920) 端到端检索评测

问题: 双路推理把 recall 从 78.7% 提升到 ~90.8%, 那么识别准确率有提升吗?

数据: 直接合并已有推理缓存 (零重推理):
  .11c_yolo_cache_960aug.json (960+TTA, conf0.01 收集) +
  .11c_yolo_cache_1920.json   (1920 原生, conf0.01 收集)
  合并 = utils/yolo_infer.py::merge_detections(conf=0.05, iou_nms=0.55)
  — 与生产模块 DualPathDetector.predict 同一实现。

输出三层对比 (strict/species/genus @top1/5):
  A. 基线检出框  (11c: 单路960 conf0.25, 直接引用 11c_yolo_e2e_report.json)
  B. 双路并集框  (本次全量)
  C. 双路新增框  (基线漏检 & 双路检出 — 低conf小目标, 重点观察其检索质量)
另有绝对覆盖: e2e strict@top1 命中数 / 全部真值框 (检出×识别 合并口径)。

用法: docker exec -e PYTHONUNBUFFERED=1 GPU_USFM_Paper python scripts/11c2_dual_e2e.py
"""
import json
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE + "/scripts")

from PIL import Image
from neo4j import GraphDatabase

from utils.bioclip_embed import get_bioclip_model, extract_embeddings_batch
from utils.yolo_infer import merge_detections

ES_DIR = os.path.join(BASE, "data/processed/eval_subset")
SUBSET_JSON = os.path.join(ES_DIR, "fathomnet_test_subset.json")
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")
C_BASE = os.path.join(ES_DIR, ".11c_yolo_cache.json")
C_AUG = os.path.join(ES_DIR, ".11c_yolo_cache_960aug.json")
C_1920 = os.path.join(ES_DIR, ".11c_yolo_cache_1920.json")
REF_JSON = os.path.join(ES_DIR, "11c_yolo_e2e_report.json")
E2E_CACHE = os.path.join(ES_DIR, ".11c2_e2e_cache.json")
OUT_JSON = os.path.join(ES_DIR, "11c2_dual_e2e_report.json")
VAL_DIR = "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password")
DEVICE = "cuda:1"
TOPK = 10
CONF = 0.05
IOU_THR = 0.5
E2E_BATCH = 32

OBJ_BLOCK_RE = re.compile(r"<object>(.*?)</object>", re.S)
NAME_RE = re.compile(r"<name>([^<]+)</name>")
BOX_RE = re.compile(
    r"<xmin>\s*(\d+)\s*</xmin>\s*<ymin>\s*(\d+)\s*</ymin>\s*"
    r"<xmax>\s*(\d+)\s*</xmax>\s*<ymax>\s*(\d+)\s*</ymax>", re.S)

YOLO_CLASSES = ["Octocorallia", "Malacostraca", "Holothuroidea", "Hexacorallia",
                "Hexactinellida", "Actinopterygii", "Echinoidea", "Asteroidea",
                "Hydrozoa", "Polychaeta", "Demospongiae", "Cephalopoda", "Bivalvia",
                "Scyphozoa", "Crinoidea", "Elasmobranchii", "Gastropoda", "Ophiuroidea",
                "Thaliacea", "Appendicularia", "Tentaculata", "Ascidiacea", "Nuda",
                "Thecostraca", "Myxini", "Rhynchonellata", "Holocephali", "Sagittoidea",
                "Clitellata", "Pycnogonida", "Enteropneusta", "Gymnolaemata",
                "Polyplacophora", "Other"]


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


def match_dets(gts, det_getter, meta):
    """贪心 IoU>=0.5 匹配 (与 11c 完全一致)。
    返回 (matched, cls_ok, hits); hits[(u, gt_i)] = (det, gt_mid)"""
    norm_set = {norm(c): c for c in YOLO_CLASSES}
    matched = cls_ok = 0
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
                matched += 1
                lin = meta[mid].get("lineage") or {}
                g = norm(meta[mid].get("class_name") or lin.get("class") or "Other")
                gt_cls = norm_set.get(g, "Other")
                if dets[bi]["cls"] == gt_cls:
                    cls_ok += 1
                hits[(u, gi)] = (dets[bi], mid)
    return matched, cls_ok, hits


def main():
    gts = load_gts()
    n_gt = sum(len(v[1]) for v in gts.values())
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    meta = {m["morphospecies_id"]: m for m in msp}
    print(f"评测图 {len(gts)}, 真值 bbox {n_gt}")

    c_base = json.load(open(C_BASE, encoding="utf-8"))
    c_aug = json.load(open(C_AUG, encoding="utf-8"))
    c_1920 = json.load(open(C_1920, encoding="utf-8"))

    def dual_get(u):
        return merge_detections([c_aug.get(u, []), c_1920.get(u, [])],
                                conf=CONF, iou_nms=0.55)

    n_dual_det = sum(len(dual_get(u)) for u in gts)
    m_base, k_base, h_base = match_dets(gts, lambda u: c_base.get(u, []), meta)
    m_dual, k_dual, h_dual = match_dets(gts, dual_get, meta)
    det_stats = {
        "n_gt_boxes": n_gt,
        "baseline_matched": m_base,
        "baseline_recall": round(m_base / n_gt, 4),
        "baseline_cls_acc": round(k_base / max(m_base, 1), 4),
        "dual_matched": m_dual,
        "dual_recall": round(m_dual / n_gt, 4),
        "dual_cls_acc": round(k_dual / max(m_dual, 1), 4),
        "n_dual_det_boxes": n_dual_det,
    }
    print("检测层:", json.dumps(det_stats, ensure_ascii=False))

    gain_keys = [k for k in h_dual if k not in h_base]
    overlap_keys = [k for k in h_dual if k in h_base]
    print(f"双路新增检出 {len(gain_keys)}, 与基线重合 {len(overlap_keys)}")

    # 嵌入所有双路匹配框 (key = uuid|x0_y0_x1_y1 坐标哈希, 断点续跑)
    tasks = []
    for (u, gi), (d, mid) in h_dual.items():
        x0, y0, x1, y1 = d["box"]
        tasks.append((f"{u}|{x0}_{y0}_{x1}_{y1}", gts[u][0],
                      tuple(d["box"]), mid, (u, gi)))
    ec = json.load(open(E2E_CACHE, encoding="utf-8")) if os.path.exists(E2E_CACHE) else {}
    todo = [t for t in tasks if t[0] not in ec]
    print(f"待嵌入 {len(todo)} (缓存 {len(ec)})")
    if todo:
        bm, bproc = get_bioclip_model(DEVICE)
        t0, done = time.time(), 0
        for b0 in range(0, len(todo), E2E_BATCH):
            chunk = todo[b0:b0 + E2E_BATCH]
            imgs = []
            for _, ipath, box, _, _ in chunk:
                try:
                    imgs.append(Image.open(ipath).convert("RGB").crop(box))
                except (OSError, ValueError):
                    imgs.append(None)
            embs = extract_embeddings_batch(imgs, bm, bproc,
                                            batch_size=E2E_BATCH, device=DEVICE)
            for t, im, e in zip(chunk, imgs, embs):
                ec[t[0]] = None if (im is None or e is None) else [round(float(x), 6) for x in e]
            done += len(chunk)
            if done % 1600 < E2E_BATCH or done == len(todo):
                atomic_json(E2E_CACHE, ec)
                print(f"嵌入 {done}/{len(todo)} ({done/(time.time()-t0):.1f}/s)", flush=True)
        atomic_json(E2E_CACHE, ec)

    # Neo4j 检索
    keymap = {t[4]: t[0] for t in tasks}
    ranked = {}
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        driver.verify_connectivity()
        with driver.session() as s:
            for n, ((u, gi), (d, mid)) in enumerate(h_dual.items()):
                emb = ec.get(keymap[(u, gi)])
                if not emb:
                    continue
                rows = s.run("""
                    CALL db.index.vector.queryNodes('entryimage_embeddings', $k, $emb)
                    YIELD node, score
                    MATCH (ms:Morphospecies {morphospecies_id: node.msp_id})
                    RETURN node.msp_id AS mid, score
                """, k=TOPK, emb=emb).data()
                best = {}
                for r in rows:
                    best[r["mid"]] = max(best.get(r["mid"], 0), r["score"])
                ranked[(u, gi)] = (mid, sorted(best.items(),
                                               key=lambda kv: -kv[1])[:TOPK])
                if (n + 1) % 1000 == 0:
                    print(f"检索 {n+1}/{len(h_dual)}", flush=True)
    finally:
        driver.close()

    def hit_at(entry, k, mode):
        gt, rk = entry
        for m, _ in rk[:k]:
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

    def stats_for(keys):
        entries = [ranked[k] for k in keys if k in ranked]
        out = {"n_evaluated": len(entries)}
        for mode in ("strict", "species", "genus"):
            for k in (1, 5):
                out[f"{mode}@top{k}"] = round(
                    sum(hit_at(e, k, mode) for e in entries) / max(len(entries), 1), 4)
        return out

    result = {
        "config": {"conf": CONF, "iou_nms": 0.55, "paths": "960+TTA / 1920"},
        "detection": det_stats,
        "e2e_dual_all": stats_for(list(ranked.keys())),
        "e2e_overlap_with_baseline": stats_for([k for k in overlap_keys if k in ranked]),
        "e2e_dual_gain": stats_for([k for k in gain_keys if k in ranked]),
    }
    s1 = result["e2e_dual_all"]["strict@top1"]
    result["absolute_coverage"] = {
        "dual_strict_top1_hits": round(s1 * len(ranked)),
        "dual_coverage_rate": round(s1 * len(ranked) / n_gt, 4),
    }
    if os.path.exists(REF_JSON):
        ref = json.load(open(REF_JSON, encoding="utf-8"))
        r1 = (ref.get("e2e") or {}).get("strict@top1", 0)
        nb = (ref.get("detection") or {}).get("detected_iou05", 0)
        result["absolute_coverage"]["baseline_coverage_rate"] = round(r1 * nb / n_gt, 4)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    atomic_json(OUT_JSON, result)
    print("已写出:", OUT_JSON)


if __name__ == "__main__":
    main()
