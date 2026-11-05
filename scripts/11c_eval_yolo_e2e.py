#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11c_eval_yolo_e2e.py — YOLO 检测评测 + 端到端联合评测 (三层评测·第一层+贯通)

输入: 验证子集 (2,861 图, 真值 concept bbox 来自 XML)

Part1 检测层: YOLO 34纲 best.pt 推理 -> 真值 bbox IoU>=0.5 匹配
  - 检出率 recall@0.5
  - 纲分类正确率 (检出 bbox 的 34纲 vs 真值 concept 的纲, 真值纲由 concept->msp->class_name)
Part2 端到端: YOLO 检出 bbox crop -> BioCLIP2.5 嵌入 -> 图谱检索 top-k
  - strict@1/5, species@1/5, genus@1/5 (与 11b 真值crop指标对比, 量化检测误差影响)

断点续跑: data/processed/eval_subset/.11c_yolo_cache.json + .11c_e2e_cache.json
用法: docker exec -e PYTHONUNBUFFERED=1 GPU_USFM_Paper python scripts/11c_eval_yolo_e2e.py
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
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")
WEIGHTS = os.path.join(BASE, "runs/detect/yolo11m_class34-2/weights/best.pt")
YOLO_CACHE = os.path.join(BASE, "data/processed/eval_subset/.11c_yolo_cache.json")
E2E_CACHE = os.path.join(BASE, "data/processed/eval_subset/.11c_e2e_cache.json")
OUT_JSON = os.path.join(BASE, "data/processed/eval_subset/11c_yolo_e2e_report.json")
VAL_DIR = "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password")
DEVICE = "cuda:1"
TOPK = 10
CONF = 0.25
IOU_THR = 0.5
E2E_BATCH = 32

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

# YOLO 34纲 -> 生物真值纲名同形; 真值纲不在 33 生物纲内视为 Other
YOLO_CLASSES = ["Octocorallia", "Malacostraca", "Holothuroidea", "Hexacorallia",
                "Hexactinellida", "Actinopterygii", "Echinoidea", "Asteroidea",
                "Hydrozoa", "Polychaeta", "Demospongiae", "Cephalopoda", "Bivalvia",
                "Scyphozoa", "Crinoidea", "Elasmobranchii", "Gastropoda", "Ophiuroidea",
                "Thaliacea", "Appendicularia", "Tentaculata", "Ascidiacea", "Nuda",
                "Thecostraca", "Myxini", "Rhynchonellata", "Holocephali", "Sagittoidea",
                "Clitellata", "Pycnogonida", "Enteropneusta", "Gymnolaemata",
                "Polyplacophora", "Other"]


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


def main():
    subset = json.load(open(SUBSET_JSON, encoding="utf-8"))
    records = subset["records"]
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    meta = {m["morphospecies_id"]: m for m in msp}

    # 真值: uuid -> [(concept, gt_mid, box)]
    gts = {}
    for r in records:
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
                items.append((name, mid, (int(x0), int(y0), int(x1), int(y1))))
        if items:
            gts[uuid] = (os.path.join(VAL_DIR, r["image"]), items)
    n_gt = sum(len(v[1]) for v in gts.values())
    print(f"评测图 {len(gts)}, 真值 bbox {n_gt}")

    # ---- Part1: YOLO 推理 (断点续跑) ----
    ycache = json.load(open(YOLO_CACHE, encoding="utf-8")) if os.path.exists(YOLO_CACHE) else {}
    todo = [u for u in gts if u not in ycache]
    print(f"YOLO 待推理 {len(todo)} (缓存 {len(ycache)})")
    if todo:
        from ultralytics import YOLO
        model = YOLO(WEIGHTS)
        t0 = time.time()
        import torch
        PB = 64  # 分批推理, 防显存累积
        for b0 in range(0, len(todo), PB):
            chunk_u = todo[b0:b0 + PB]
            paths = [gts[u][0] for u in chunk_u]
            results = model.predict(paths, conf=CONF, device=DEVICE,
                                    imgsz=960, verbose=False)
            for u, res in zip(chunk_u, results):
                dets = []
                for b in res.boxes:
                    x0, y0, x1, y1 = [round(float(v)) for v in b.xyxy[0].tolist()]
                    dets.append({"cls": YOLO_CLASSES[int(b.cls[0])], "conf": round(float(b.conf[0]), 3),
                                 "box": [x0, y0, x1, y1]})
                ycache[u] = dets
            atomic_json(YOLO_CACHE, ycache)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"YOLO {len(ycache)}/{len(gts)} ({len(ycache)/(time.time()-t0):.1f}img/s)", flush=True)
        atomic_json(YOLO_CACHE, ycache)
        print("YOLO 推理完成")

    # ---- 检测层指标 ----
    matched_total = matched_cls = 0
    e2e_tasks = []   # (key, img_path, box, gt_mid, det_cls)
    for u, (ipath, items) in gts.items():
        dets = ycache.get(u, [])
        used = set()
        for name, mid, box in items:
            best, bi = 0, -1
            for i, d in enumerate(dets):
                if i in used:
                    continue
                v = iou(box, d["box"])
                if v > best:
                    best, bi = v, i
            if best >= IOU_THR:
                used.add(bi)
                matched_total += 1
                lin = meta[mid].get("lineage") or {}
                gt_cls = meta[mid].get("class_name") or lin.get("class") or "Other"
                if norm(gt_cls) not in {norm(c) for c in YOLO_CLASSES}:
                    gt_cls = "Other"
                else:
                    gt_cls = next(c for c in YOLO_CLASSES if norm(c) == norm(gt_cls))
                if dets[bi]["cls"] == gt_cls:
                    matched_cls += 1
                e2e_tasks.append((f"{u}|{bi}", ipath, tuple(dets[bi]["box"]), mid))
    n_gt_classes = len({(u, i) for u, (p, items) in gts.items() for i in range(len(items))})
    stats = {
        "n_gt_boxes": n_gt,
        "detected_iou05": matched_total,
        "recall_iou05": round(matched_total / max(n_gt, 1), 4),
        "cls_correct": matched_cls,
        "cls_acc_on_matched": round(matched_cls / max(matched_total, 1), 4),
    }
    print("Part1 检测层:", json.dumps(stats, ensure_ascii=False))

    # ---- Part2: 端到端 (检出框 crop -> 嵌入 -> 检索) ----
    ec = json.load(open(E2E_CACHE, encoding="utf-8")) if os.path.exists(E2E_CACHE) else {}
    todo = [t for t in e2e_tasks if t[0] not in ec]
    print(f"端到端待嵌入 {len(todo)}")
    if todo:
        bm, bproc = get_bioclip_model(DEVICE)
        t0, done = time.time(), 0
        for b0 in range(0, len(todo), E2E_BATCH):
            chunk = todo[b0:b0 + E2E_BATCH]
            imgs = []
            for _, ipath, box, _ in chunk:
                try:
                    imgs.append(Image.open(ipath).convert("RGB").crop(box))
                except (OSError, ValueError):
                    imgs.append(None)
            embs = extract_embeddings_batch(imgs, bm, bproc, batch_size=E2E_BATCH, device=DEVICE)
            for t, im, e in zip(chunk, imgs, embs):
                ec[t[0]] = None if im is None or e is None else [round(float(x), 6) for x in e]
            done += len(chunk)
            if done % 2000 < E2E_BATCH or done == len(todo):
                atomic_json(E2E_CACHE, ec)
                rate = done / (time.time() - t0)
                print(f"端到端嵌入 {done}/{len(todo)} ({rate:.1f}/s)", flush=True)
        atomic_json(E2E_CACHE, ec)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    e2e = []
    try:
        driver.verify_connectivity()
        with driver.session() as s:
            for key, _, _, gt in e2e_tasks:
                emb = ec.get(key)
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
                e2e.append({"key": key, "gt": gt,
                            "ranked": sorted(best.items(), key=lambda kv: -kv[1])[:TOPK]})
    finally:
        driver.close()

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

    e2e_stats = {"n_evaluated": len(e2e)}
    for mode in ("strict", "species", "genus"):
        for k in (1, 5):
            e2e_stats[f"{mode}@top{k}"] = round(
                sum(hit_at(r["ranked"], r["gt"], k, mode) for r in e2e) / max(len(e2e), 1), 4)
    print("Part2 端到端:", json.dumps(e2e_stats, ensure_ascii=False))
    atomic_json(OUT_JSON, {"detection": stats, "e2e": e2e_stats})
    print(f"已写出: {OUT_JSON}")


if __name__ == "__main__":
    main()
