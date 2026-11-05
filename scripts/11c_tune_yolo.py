#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11c_tune_yolo.py — YOLO 推理参数调优: imgsz / conf 对检测 recall 的影响

背景: 11c 基线 imgsz=960, conf=0.25 -> recall@IoU0.5 = 78.7%
归因: 90% 漏检为完全无重叠 + 漏检框中位面积仅命中框的 47% -> 小目标下采样丢失
方案: 原生分辨率推理 (imgsz=1920, 图像中位 1920x1080) + 极低 conf 收集,
      后处理扫描 conf 阈值, 一次跑出 recall-conf 曲线。

用法: docker exec -e PYTHONUNBUFFERED=1 GPU_USFM_Paper python scripts/11c_tune_yolo.py [imgsz] [conf] [suffix]
"""
import json
import os
import re
import sys
import time
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET_JSON = os.path.join(BASE, "data/processed/eval_subset/fathomnet_test_subset.json")
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")
WEIGHTS = os.path.join(BASE, "runs/detect/yolo11m_class34-2/weights/best.pt")
VAL_DIR = "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet"
DEVICE = "cuda:1"
IOU_THR = 0.5

IMGSZ = int(sys.argv[1]) if len(sys.argv) > 1 else 1920
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
SUF = sys.argv[3] if len(sys.argv) > 3 else f"_{IMGSZ}"
AUG = (len(sys.argv) > 4 and sys.argv[4] == "aug")
CACHE = os.path.join(BASE, f"data/processed/eval_subset/.11c_yolo_cache{SUF}.json")

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


def main():
    subset = json.load(open(SUBSET_JSON, encoding="utf-8"))
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    meta = {m["morphospecies_id"]: m for m in msp}

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
    n_gt = sum(len(v[1]) for v in gts.values())
    print(f"评测图 {len(gts)}, 真值 bbox {n_gt}")

    ycache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    todo = [u for u in gts if u not in ycache]
    print(f"imgsz={IMGSZ} conf={CONF} aug={AUG} 待推理 {len(todo)} (缓存 {len(ycache)})")
    if todo:
        from ultralytics import YOLO
        import torch
        model = YOLO(WEIGHTS)
        t0 = time.time()
        PB = 16
        for b0 in range(0, len(todo), PB):
            chunk_u = todo[b0:b0 + PB]
            paths = [gts[u][0] for u in chunk_u]
            results = model.predict(paths, conf=CONF, device=DEVICE,
                                    imgsz=IMGSZ, verbose=False, augment=AUG)
            for u, res in zip(chunk_u, results):
                dets = []
                for b in res.boxes:
                    x0, y0, x1, y1 = [round(float(v)) for v in b.xyxy[0].tolist()]
                    dets.append({"cls": YOLO_CLASSES[int(b.cls[0])],
                                 "conf": round(float(b.conf[0]), 4),
                                 "box": [x0, y0, x1, y1]})
                ycache[u] = dets
            atomic_json(CACHE, ycache)
            torch.cuda.empty_cache()
            if (b0 // PB) % 10 == 0:
                print(f"  {len(ycache)}/{len(gts)} ({len(ycache)/(time.time()-t0):.1f}img/s)", flush=True)
        print("推理完成")

    norm_set = {norm(c): c for c in YOLO_CLASSES}
    print(f"\n=== conf 阈值扫描 (imgsz={IMGSZ}) ===")
    print(f"{'conf':>6} {'检出':>6} {'recall':>8} {'纲正确':>6} {'纲acc':>7} {'总检出框':>8}")
    for thr in (0.01, 0.05, 0.10, 0.25, 0.50):
        matched = cls_ok = n_det_total = 0
        for u, (_, items) in gts.items():
            dets = [d for d in ycache.get(u, []) if d["conf"] >= thr]
            n_det_total += len(dets)
            used = set()
            for mid, box in items:
                best, bi = 0, -1
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
        print(f"{thr:>6} {matched:>6} {matched/n_gt:>7.1%} {cls_ok:>6} "
              f"{cls_ok/max(matched,1):>7.1%} {n_det_total:>8}")


if __name__ == "__main__":
    main()
