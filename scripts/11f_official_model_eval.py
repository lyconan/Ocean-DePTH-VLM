#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11f_official_model_eval.py — FathomNet 官方预训练模型 (MBARI-315k-yolov8, 499类) 对比评测

问题: MBARI/CVisionAI 在 315k 海洋图像上训练的官方 yolov8x, 在我们 2,848 图
      FathomNet 测试子集上, 检出率和纲级 acc 相比自训 yolo11m (34纲) 有提升吗?

模型: HuggingFace FathomNet/MBARI-315k-yolov8
      - yolov8x (68.6M), 原生 imgsz 640 训练 (2023-08), 499 FathomNet concept (部分种级)
      - 注意: 其训练集含 FathomNet 全量快照, 可能覆盖我们的测试图 → 对比时留意。

评测:
  A. concept→纲 映射 (缓存 mbari_315k_concept_class.json):
     图谱 latin 精确匹配 / 概念本身是34纲名 / WoRMS record (别名 teleostei→actinopterygii) /
     门级概念与 trash/tube 等非分类概念 → Other
  B. 推理: conf 0.01 收集, imgsz 640 (原生) 与 1280 (放大) 两档, 断点续跑缓存
  C. 贪心 IoU>=0.5 匹配 (与 11c/11c2 完全一致), conf 0.25/0.05/0.01 三档:
     recall + 纲级 cls_acc 双口径 (含Other / 剔Other) + 非Other 漏检/混淆 top
参照: 自训双路 960TTA+1920 @conf0.05: recall 87.4% / 含O acc 78.8% / 剔O acc 94.1%

用法: docker exec -e PYTHONUNBUFFERED=1 GPU_USFM_Paper \
        python scripts/11f_official_model_eval.py [--imgsz 640 1280] [--map-only] [--eval-only]
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ES_DIR = os.path.join(BASE, "data/processed/eval_subset")
EXT_DIR = os.path.join(BASE, "data/external")
SUBSET_JSON = os.path.join(ES_DIR, "fathomnet_test_subset.json")
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")
WEIGHTS = os.path.join(EXT_DIR, "mbari_315k_yolov8.pt")
CFG_YAML = os.path.join(EXT_DIR, "mbari_315k_config.yaml")
MAP_JSON = os.path.join(EXT_DIR, "mbari_315k_concept_class.json")
OUT_JSON = os.path.join(ES_DIR, "11f_official_model_report.json")
VAL_DIR = "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet"

DEVICE = os.environ.get("DEVICE", "cuda:1")
CONF_COLLECT = 0.01
CONF_EVAL = [0.25, 0.05, 0.01]
IOU_THR = 0.5
CHUNK = 16
SAVE_EVERY = 200

YOLO_CLASSES = ["Octocorallia", "Malacostraca", "Holothuroidea", "Hexacorallia",
                "Hexactinellida", "Actinopterygii", "Echinoidea", "Asteroidea",
                "Hydrozoa", "Polychaeta", "Demospongiae", "Cephalopoda", "Bivalvia",
                "Scyphozoa", "Crinoidea", "Elasmobranchii", "Gastropoda", "Ophiuroidea",
                "Thaliacea", "Appendicularia", "Tentaculata", "Ascidiacea", "Nuda",
                "Thecostraca", "Myxini", "Rhynchonellata", "Holocephali", "Sagittoidea",
                "Clitellata", "Pycnogonida", "Enteropneusta", "Gymnolaemata",
                "Polyplacophora", "Other"]

# WoRMS 新分类与我们 34 纲体系的 rank 差异别名 (norm 后小写)
ALIAS = {"teleostei": "actinopterygii", "actinopteri": "actinopterygii",
         "neopterygii": "actinopterygii", "elasmobranchiomorphi": "elasmobranchii"}

OBJ_BLOCK_RE = re.compile(r"<object>(.*?)</object>", re.S)
NAME_RE = re.compile(r"<name>([^<]+)</name>")
BOX_RE = re.compile(
    r"<xmin>\s*(\d+)\s*</xmin>\s*<ymin>\s*(\d+)\s*</ymin>\s*"
    r"<xmax>\s*(\d+)\s*</xmax>\s*<ymax>\s*(\d+)\s*</ymax>", re.S)


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


# ---------------- Part A: concept → 纲 映射 ----------------

def _http_json(url, tries=4, timeout=20):
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (i + 1))
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def worms_class(concept):
    """WoRMS AphiaRecordsByName → (纲名norm, rank)。
    依次尝试: 原名 / 去 ' eggs' / 去 ' sp. X' / 去 ' complex' / 去 ' cf. X'。查不到返回 (None, None)。"""
    cands = [concept]
    for suf in (r"\s+eggs$", r"\s+sp\.\s*\w+$", r"\s+complex$", r"\s+cf\.\s*\w+$"):
        s = re.sub(suf, "", concept)
        if s != concept and s not in cands:
            cands.append(s)
    for cand in cands:
        txt = _http_json("https://www.marinespecies.org/rest/AphiaRecordsByName/"
                         + urllib.request.quote(cand) + "?marine_only=false")
        if not txt:
            continue
        try:
            recs = json.loads(txt)
        except ValueError:
            continue
        rec = next((r for r in recs if r.get("status") == "accepted"), recs[0] if recs else None)
        if not rec:
            continue
        for field in ("class", "phylum"):  # class 为空时回退 phylum (门级概念)
            v = norm(rec.get(field) or "")
            if v:
                return v, rec.get("rank")
        return "", rec.get("rank")
    return None, None


def build_concept_map():
    if os.path.exists(MAP_JSON):
        return json.load(open(MAP_JSON, encoding="utf-8"))

    import yaml
    cfg = yaml.safe_load(open(CFG_YAML, encoding="utf-8"))
    concepts = list(cfg["names"].values())

    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    lat2cls = {}
    for m in msp:
        lin = m.get("lineage") or {}
        c = m.get("class_name") or lin.get("class")
        if c:
            lat2cls.setdefault(norm(m.get("latin", "")), c)

    n34 = {norm(c): c for c in YOLO_CLASSES}
    cmap, src = {}, {}
    for c in concepts:
        if norm(c) in lat2cls:                      # 1. 图谱精确匹配
            cmap[c] = n34.get(norm(lat2cls[norm(c)]), "Other")
            src[c] = "atlas"
        elif norm(c) in n34:                        # 2. 概念本身是纲名
            cmap[c] = n34[norm(c)]
            src[c] = "self"
        else:                                       # 3. WoRMS
            w, rank = worms_class(c)
            if w is None:
                cmap[c], src[c] = "Other", "not_found"
            else:
                w = ALIAS.get(w, w)
                cmap[c] = n34.get(w, "Other")
                src[c] = "worms:%s" % (rank or "?")
            time.sleep(1.0)
    atomic_json(MAP_JSON, {"concept_class": cmap, "source": src})
    return {"concept_class": cmap, "source": src}


# ---------------- Part B: 推理 ----------------

def infer(imgsz):
    cache_path = os.path.join(ES_DIR, ".11f_cache_official_%d.json" % imgsz)
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    gts = load_gts()
    todo = [u for u in gts if u not in cache]
    print(f"[infer imgsz={imgsz}] 待推理 {len(todo)} / {len(gts)} (缓存 {len(cache)})", flush=True)
    if not todo:
        return cache_path

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    t0 = time.time()
    for b0 in range(0, len(todo), CHUNK):
        chunk = todo[b0:b0 + CHUNK]
        results = model.predict([gts[u][0] for u in chunk], conf=CONF_COLLECT,
                                imgsz=imgsz, device=DEVICE, verbose=False, stream=True)
        for u, r in zip(chunk, results):
            dets = []
            if r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().tolist()
                conf = r.boxes.conf.cpu().tolist()
                cls = r.boxes.cls.cpu().tolist()
                for box, cf, ci in zip(xyxy, conf, cls):
                    dets.append({"box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                                 "conf": round(cf, 5), "cid": int(ci)})
            cache[u] = dets
        done = min(b0 + CHUNK, len(todo))
        if done % SAVE_EVERY < CHUNK or done == len(todo):
            atomic_json(cache_path, cache)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(todo)} ({rate:.1f} img/s, ETA {(len(todo)-done)/max(rate,1e-6)/60:.0f} min)",
                  flush=True)
    atomic_json(cache_path, cache)
    return cache_path


# ---------------- Part C: 评测 ----------------

def evaluate(gts, meta, names, cmap, cache, conf):
    norm_set = {norm(c): c for c in YOLO_CLASSES}
    dets_all = sum(len(v) for v in cache.values())

    def det_getter(u):
        out = []
        for d in cache.get(u, []):
            if d["conf"] >= conf:
                out.append({"box": d["box"],
                            "cls": cmap.get(names.get(d["cid"], ""), "Other")})
        return out

    matched = ok = 0
    n_gt = gt_other_total = 0
    n_nonother = m_nonother = ok_nonother = 0
    miss_by_cls, conf_pairs = {}, {}
    for u, (_, items) in gts.items():
        dets = det_getter(u)
        used = set()
        for gi, (mid, box) in enumerate(items):
            n_gt += 1
            lin = meta[mid].get("lineage") or {}
            g = norm(meta[mid].get("class_name") or lin.get("class") or "Other")
            gt_cls = norm_set.get(g, "Other")
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
                pred_cls = dets[bi]["cls"]
                if gt_cls == "Other":
                    gt_other_total += 1
                else:
                    n_nonother += 1
                    m_nonother += 1
                    if pred_cls == gt_cls:
                        ok_nonother += 1
                        ok += 1
                    else:
                        conf_pairs[f"{gt_cls}->{pred_cls}"] = \
                            conf_pairs.get(f"{gt_cls}->{pred_cls}", 0) + 1
            else:
                if gt_cls == "Other":
                    gt_other_total += 1
                else:
                    n_nonother += 1
                    miss_by_cls[gt_cls] = miss_by_cls.get(gt_cls, 0) + 1

    top_miss = dict(sorted(miss_by_cls.items(), key=lambda kv: -kv[1])[:8])
    top_conf = dict(sorted(conf_pairs.items(), key=lambda kv: -kv[1])[:10])
    return {
        "conf": conf,
        "n_det_boxes": dets_all,
        "recall": round(matched / max(n_gt, 1), 4),
        "acc_with_other": round(ok / max(matched, 1), 4),
        "gt_other_boxes": gt_other_total,
        "recall_no_other": round(m_nonother / max(n_nonother, 1), 4),
        "acc_no_other": round(ok_nonother / max(m_nonother, 1), 4),
        "miss_no_other_top": top_miss,
        "confusion_no_other_top": top_conf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280])
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    m = build_concept_map()
    cmap = m["concept_class"]
    from collections import Counter
    print("概念→纲 映射:", dict(Counter(cmap.values()).most_common()))
    print("来源:", dict(Counter(m["source"].values()).most_common()))
    if args.map_only:
        return

    import yaml
    names = {int(k): v for k, v in yaml.safe_load(open(CFG_YAML, encoding="utf-8"))["names"].items()}
    gts = load_gts()
    n_gt = sum(len(v[1]) for v in gts.values())
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    meta = {m2["morphospecies_id"]: m2 for m2 in msp}
    print(f"评测图 {len(gts)}, 真值 bbox {n_gt}")

    if not args.eval_only:
        for imgsz in args.imgsz:
            infer(imgsz)

    report = {
        "config": {"weights": "FathomNet/MBARI-315k-yolov8 (yolov8x, 499 concepts)",
                   "imgsz": args.imgsz, "iou_thr": IOU_THR,
                   "note": "官方模型训练集含 FathomNet 快照, 可能覆盖本测试图 (对比时留意)"},
        "reference_ours": {
            "dual_960tta_1920@conf0.05": {"recall": 0.8737, "acc_with_other": 0.7881,
                                          "acc_no_other": 0.9412},
            "tri_add_tiled@conf0.01": {"recall": 0.8966, "acc_with_other": 0.7799,
                                       "acc_no_other": 0.9318},
        },
        "n_gt_boxes": n_gt,
        "official": {},
    }
    for imgsz in args.imgsz:
        cache_path = os.path.join(ES_DIR, ".11f_cache_official_%d.json" % imgsz)
        cache = json.load(open(cache_path, encoding="utf-8"))
        report["official"][str(imgsz)] = {("conf%.2f" % c): evaluate(gts, meta, names,
                                                                    cmap, cache, c)
                                          for c in CONF_EVAL}
        print(f"[imgsz={imgsz}] " + json.dumps(
            {k: {kk: v[kk] for kk in ("recall", "recall_no_other", "acc_no_other")}
             for k, v in report["official"][str(imgsz)].items()}, ensure_ascii=False))

    atomic_json(OUT_JSON, report)
    print("已写出:", OUT_JSON)


if __name__ == "__main__":
    main()
