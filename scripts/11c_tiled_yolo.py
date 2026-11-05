#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11c_tiled_yolo.py — 切片推理 (SAHI 思路): 补齐小目标检测

已验证: 全图960+TTA (尺度0.5/0.415/0.335) U 全图1920 (尺度1.0) -> recall 90.8%
本脚本: 原生 1280x1280 切片 (stride 640) 按 imgsz 推理, 补中间尺度 + 小目标像素密度,
       只对"多尺度并集后仍有漏检"的图跑, 结果写入独立缓存供并集评测。

用法: docker exec -e PYTHONUNBUFFERED=1 GPU_USFM_Paper python scripts/11c_tiled_yolo.py [tile] [stride] [imgsz] [conf]
"""
import json
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET_JSON = os.path.join(BASE, "data/processed/eval_subset/fathomnet_test_subset.json")
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")
WEIGHTS = os.path.join(BASE, "runs/detect/yolo11m_class34-2/weights/best.pt")
VAL_DIR = "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet"
ES_DIR = os.path.join(BASE, "data/processed/eval_subset")
DEVICE = "cuda:1"
IOU_THR = 0.5

TILE = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 640
IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 960
CONF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
CACHE = os.path.join(ES_DIR, f".11c_yolo_cache_tiled{TILE}_{IMGSZ}.json")

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


def find_miss_images(gts):
    """多尺度并集 (960TTA + 1920, conf 0.01) 后仍有漏检的图。"""
    c_aug = json.load(open(os.path.join(ES_DIR, ".11c_yolo_cache_960aug.json"), encoding="utf-8"))
    c_1920 = json.load(open(os.path.join(ES_DIR, ".11c_yolo_cache_1920.json"), encoding="utf-8"))
    miss = set()
    for u, (_, items) in gts.items():
        dets = [d for d in c_aug.get(u, []) if d["conf"] >= 0.01] + \
               [d for d in c_1920.get(u, []) if d["conf"] >= 0.01]
        for mid, box in items:
            if all(iou(box, d["box"]) < IOU_THR for d in dets):
                miss.add(u)
                break
    return miss


def main():
    gts = load_gts()
    n_gt = sum(len(v[1]) for v in gts.values())
    miss_imgs = find_miss_images(gts)
    print(f"评测图 {len(gts)}, 真值 bbox {n_gt}, 并集后仍有漏检的图 {len(miss_imgs)}")

    tcache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    todo = [u for u in miss_imgs if u not in tcache]
    print(f"tile={TILE} stride={STRIDE} imgsz={IMGSZ} conf={CONF} 待切片 {len(todo)}")
    if todo:
        from ultralytics import YOLO
        from PIL import Image
        import torch
        model = YOLO(WEIGHTS)
        t0 = time.time()
        for idx_i, u in enumerate(todo):
            ipath = gts[u][0]
            with Image.open(ipath) as im:
                W, H = im.size
                crops, offs = [], []
                xs = list(range(0, max(W - TILE, 0) + 1, STRIDE)) or [0]
                if xs[-1] + TILE < W:
                    xs.append(W - TILE)
                ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)) or [0]
                if ys[-1] + TILE < H:
                    ys.append(H - TILE)
                for x0 in xs:
                    for y0 in ys:
                        crops.append(im.crop((x0, y0, min(x0 + TILE, W), min(y0 + TILE, H))))
                        offs.append((x0, y0))
            results = model.predict(crops, conf=CONF, device=DEVICE,
                                    imgsz=IMGSZ, verbose=False)
            dets = []
            for res, (ox, oy) in zip(results, offs):
                for b in res.boxes:
                    x0, y0, x1, y1 = [round(float(v)) for v in b.xyxy[0].tolist()]
                    dets.append({"cls": YOLO_CLASSES[int(b.cls[0])],
                                 "conf": round(float(b.conf[0]), 4),
                                 "box": [x0 + ox, y0 + oy, x1 + ox, y1 + oy]})
            tcache[u] = dets
            if idx_i % 10 == 0:
                atomic_json(CACHE, tcache)
                print(f"  {idx_i+1}/{len(todo)} ({(idx_i+1)/(time.time()-t0):.1f}img/s)", flush=True)
        atomic_json(CACHE, tcache)
        print("切片推理完成")

    # 评测: 切片单独 + 三路并集
    c_aug = json.load(open(os.path.join(ES_DIR, ".11c_yolo_cache_960aug.json"), encoding="utf-8"))
    c_1920 = json.load(open(os.path.join(ES_DIR, ".11c_yolo_cache_1920.json"), encoding="utf-8"))
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    meta = {m["morphospecies_id"]: m for m in msp}
    norm = lambda s: " ".join(s.split()).lower()
    norm_set = {norm(c): c for c in YOLO_CLASSES}

    for label, get_dets in (
        ("切片单独@0.01", lambda u: [d for d in tcache.get(u, []) if d["conf"] >= 0.01]),
        ("TTA+1920+切片 并集@0.01", lambda u: [d for d in c_aug.get(u, []) if d["conf"] >= 0.01]
            + [d for d in c_1920.get(u, []) if d["conf"] >= 0.01]
            + [d for d in tcache.get(u, []) if d["conf"] >= 0.01]),
    ):
        matched = cls_ok = 0
        for u, (_, items) in gts.items():
            dets = get_dets(u)
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
        print(f"{label}: recall {matched}/{n_gt} = {matched/n_gt:.1%}, 纲acc {cls_ok/max(matched,1):.1%}")


if __name__ == "__main__":
    main()
