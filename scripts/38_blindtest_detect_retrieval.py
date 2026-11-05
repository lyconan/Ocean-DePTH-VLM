#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
38_blindtest_detect_retrieval.py — 盲测集 检测+粗类别+精细检索 基准 (无VLM)
数据: data/processed/blindtest/manifest.json (1,744 盲测场景, 库中已无同源图)
场景图: /mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet/Val/<uuid>.png
GT:     同目录 <uuid>.xml (concept 级框标注, concept->mid 经 morphospecies_index)
        同目录 <uuid>.txt (34纲 YOLO 标注, 作粗类别GT, 与xml框按IoU配准)
管线:   YOLO双路(960+TTA ∪ 1920, conf0.05/nms0.55) -> 检出框与GT贪心IoU0.5匹配
        -> 匹配框 tight裁剪 -> BioCLIP v2 嵌入+翻转TTA -> 向量索引 top-10
        -> mid聚合 -> 物种 strict top-1/5/10 + 属级 top-1/5/10
输出:   /exp/blindtest/e2e_report.json + per_object.jsonl
运行:   容器 GPU0 (同36号脚本启动方式, 需挂载场景图目录)
"""
import os
import sys
import json
import time
import pickle
import xml.etree.ElementTree as ET

sys.path.insert(0, "/work/scripts")
import numpy as np
from PIL import Image
from neo4j import GraphDatabase

VAL_DIR = "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet/Val"
MSP_JSON = "/work/data/processed/atlas_lookup/morphospecies_index.json"
MANIFEST = "/exp/blindtest/manifest.json"
OUT_DIR = "/exp/blindtest"
IOU_MATCH = 0.5
TOPK = 10
YOLO_WEIGHTS = "runs/detect/yolo11m_class34-2/weights/best.pt"
CF_W = float(os.environ.get("CF_W", "0.5"))  # cfusion 图像分权重 (质心分权重=1-CF_W)
QE = os.environ.get("QE", "0") == "1"  # P2: Query Expansion 开关

import torch
from utils.bioclip_embed import get_bioclip_model, extract_embeddings_batch
from utils.yolo_infer import DualPathDetector


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def parse_xml(path, concept2mid):
    """返回 [(box_xyxy, concept, mid_or_None)]"""
    out = []
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return out
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
            x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        out.append(([x1, y1, x2, y2], name, concept2mid.get(name.lower())))
    return out


def parse_txt(path, w, h):
    """34纲 YOLO 标注 -> [(class_id, box_xyxy)]"""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((int(p[0]),
                    [(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def main():
    man = json.load(open(MANIFEST))
    scenes = man["scenes"]
    print(f"盲测场景: {len(scenes)}", flush=True)

    # concept -> mid 映射 (latin + display_name, 含逐级截断变体)
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    concept2mid = {}
    for rec in idx["morphospecies"]:
        mid = rec.get("morphospecies_id")
        if not mid:
            continue
        names = set()
        for k in ("latin", "display_name"):
            v = (rec.get(k) or "").strip()
            if not v:
                continue
            names.add(v.lower())
            toks = v.split()
            while len(toks) > 1:
                toks = toks[:-1]
                names.add(" ".join(toks).lower())
        for nm in names:
            concept2mid.setdefault(nm, mid)
    print(f"concept映射: {len(concept2mid)}", flush=True)

    # WoRMS纲别 -> 模型34类 别名映射 (data/processed/taxonomy/yolo34_class_alias_map.json)
    _alias_raw = json.load(open("/work/data/processed/taxonomy/yolo34_class_alias_map.json"))
    CLASS_ALIAS = {k: v for k, v in _alias_raw.items() if not k.startswith("_")}

    # mid -> 纲别 GT (lineage.class, 不依赖 txt 类id)
    # 高阶元 fallback: lineage 缺 class 时按 latin 查别名表 (Anthozoa/Actinopterygii等)
    mid2class = {}
    for rec in idx["morphospecies"]:
        mid = rec.get("morphospecies_id")
        if not mid:
            continue
        lc = (rec.get("lineage") or {}).get("class")
        if not lc:
            latin = (rec.get("latin") or "").strip()
            lc = _alias_raw.get(latin) and latin or None
        if lc:
            mid2class[mid] = lc
    print(f"纲别映射: {len(mid2class)}", flush=True)

    # mid -> genus (Neo4j)
    drv = GraphDatabase.driver("bolt://depth_vlm_neo4j:7687", auth=("neo4j", "password"))
    with drv.session() as s:
        mid2genus = {r["mid"]: (r["genus"] or "") for r in s.run(
            "MATCH (ms:Morphospecies) RETURN ms.morphospecies_id AS mid, ms.genus AS genus")}
    print(f"genus映射: {len(mid2genus)}", flush=True)

    # mid 质心旁路 (2,638个, 已按盲测剔除后的库刷新: centroids_v2.npz)
    _npz = np.load("/work/data/experiments/ft_20261102/centroids_v2.npz")
    CEN_C = _npz["C"].astype("float32")
    CEN_MIDS = [str(m) for m in _npz["mids"]]
    CEN_IDX = {m: i for i, m in enumerate(CEN_MIDS)}
    print(f"mid质心: {len(CEN_MIDS)}", flush=True)

    # 库嵌入矩阵常驻内存 (34,026 x 1024, 暴力余弦替代逐条 HNSW 查询)
    with drv.session() as s:
        _lib = s.run(
            "MATCH (n:EntryImage) RETURN n.msp_id AS mid, n.embedding AS emb"
        ).data()
    LIB_MIDS = np.array([str(r["mid"]) for r in _lib])
    LIB_C = np.asarray([r["emb"] for r in _lib], dtype="float32")
    del _lib
    LIB_C /= np.maximum(np.linalg.norm(LIB_C, axis=1, keepdims=True), 1e-12)
    print(f"库嵌入矩阵: {LIB_C.shape}", flush=True)
    drv.close()

    # 模型
    yolo = DualPathDetector(YOLO_WEIGHTS, device="cuda:0",
                            paths=((960, True), (1920, False)))
    model, proc = get_bioclip_model()

    f_rep = open(f"{OUT_DIR}/e2e_report.json", "w")
    f_obj = open(f"{OUT_DIR}/per_object.jsonl", "w")

    # 累计统计
    S = dict(scenes=0, gt_total=0, gt_lib=0, gt_out=0, det_total=0,
             matched=0, det_unmatched=0, coarse_ok=0, coarse_n=0,
             s1=0, s5=0, s10=0, g1=0, g5=0, g10=0, retr_n=0,
             c1=0, c5=0, c10=0, f1=0, f5=0, f10=0,
             d1=0, d5=0, d10=0, r1=0, r5=0, r10=0)
    t0 = time.time()
    T = {"yolo": 0.0, "embed": 0.0, "retr": 0.0}  # 分阶段计时

    # 检测框缓存: sig 不匹配 (换权重/conf/路径) 自动重建
    DET_SIG = {"weights": YOLO_WEIGHTS, "conf": 0.05, "iou_nms": 0.55,
               "paths": "960tta+1920"}
    DET_CACHE_PATH = "/exp/blindtest/det_cache.pkl"
    DET_CACHE = {}
    if os.path.exists(DET_CACHE_PATH):
        try:
            _dc = pickle.load(open(DET_CACHE_PATH, "rb"))
            if _dc.get("sig") == DET_SIG:
                DET_CACHE = _dc["dets"]
                print(f"检测框缓存: 命中 {len(DET_CACHE)} 张", flush=True)
            else:
                print("检测框缓存: 配置变更, 自动重建", flush=True)
        except Exception:
            print("检测框缓存: 读取失败, 自动重建", flush=True)

    # E系列提速: 场景窗口化 — YOLO 跨场景批处理 + 跨场景合并嵌入 + 查询嵌入缓存
    scene_items = [(u, sc) for u, sc in sorted(scenes.items())
                   if os.path.exists(os.path.join(VAL_DIR, u + ".png"))]
    WIN = 24
    CACHE_PATH = "/exp/blindtest/emb_cache.npz"  # E3: 键=uuid|box, 值=TTA归一化嵌入
    CACHE = {}
    if os.path.exists(CACHE_PATH):
        _c = np.load(CACHE_PATH)
        CACHE = {k: v for k, v in zip(_c["k"], _c["v"])}
        print(f"嵌入缓存: 命中 {len(CACHE)} 条", flush=True)
    si = 0
    last_rep = -1
    for w0 in range(0, len(scene_items), WIN):
        window = scene_items[w0:w0 + WIN]
        # E1': YOLO 跨场景批处理 (仅推理缓存缺失的图)
        miss_uuids = [u for u, _ in window if u not in DET_CACHE]
        if miss_uuids:
            _ts = time.time()
            dets_miss = yolo.predict(
                [os.path.join(VAL_DIR, u + ".png") for u in miss_uuids],
                conf=0.05, iou_nms=0.55)  # 定版 conf=0.05 (用户决策)
            T["yolo"] += time.time() - _ts
            for _u, _dl in zip(miss_uuids, dets_miss):
                DET_CACHE[_u] = _dl if _dl else []
        dets_win = [DET_CACHE[u] for u, _ in window]
        win_tasks = []  # (row, det, mid, crop) — 嵌入延后批处理
        win_rows = []
        for (uuid, sc), _dets in zip(window, dets_win):
            S["scenes"] += 1
            si += 1
            pil = Image.open(os.path.join(VAL_DIR, uuid + ".png")).convert("RGB")
            W, H = pil.size
            gts = parse_xml(os.path.join(VAL_DIR, uuid + ".xml"), concept2mid)
            for _, _, m in gts:
                S["gt_total"] += 1
                S["gt_lib" if m else "gt_out"] += 1

            dets = _dets if _dets else []
            S["det_total"] += len(dets)
            det_boxes = [d["box"] for d in dets]

            used = set()
            scene_rows = []
            for gt_box, concept, mid in gts:
                # 贪心找最佳未用检测
                best, bi = 0.0, -1
                for di, db in enumerate(det_boxes):
                    if di in used:
                        continue
                    v = iou(gt_box, db)
                    if v > best:
                        best, bi = v, di
                row = {"uuid": uuid, "concept": concept, "mid": mid,
                       "iou": round(best, 3), "detected": best >= IOU_MATCH}
                scene_rows.append(row)
                if best >= IOU_MATCH:
                    used.add(bi)
                    S["matched"] += 1
                    d = dets[bi]
                    # 粗类别: GT纲别 经别名映射 -> 模型类名集合比对
                    gc = mid2class.get(mid)
                    gcs = [g for g in CLASS_ALIAS.get(gc, [gc] if gc else []) if g]
                    if gcs:
                        S["coarse_n"] += 1
                        if str(d.get("name", "")).strip().lower() in {
                                g.lower() for g in gcs}:
                            S["coarse_ok"] += 1
                    # 检出框 tight裁剪, 收集待批量嵌入
                    if mid:
                        crop = pil.crop((int(max(0, d["box"][0])), int(max(0, d["box"][1])),
                                         int(min(W, d["box"][2])), int(min(H, d["box"][3]))))
                        if crop.width >= 8 and crop.height >= 8:
                            win_tasks.append((row, d, mid, crop))
            win_rows.extend(scene_rows)

        # E2'/E3: 整窗口合并嵌入 (num_workers=10 重叠 CPU 预处理与 GPU) + 查询嵌入缓存
        if win_tasks:
            crops = [t[3] for t in win_tasks]
            keys = [f"{t[0]['uuid']}|{','.join(map(str, t[1]['box']))}"
                    for t in win_tasks]
            miss = [i for i, k in enumerate(keys) if k not in CACHE]
            _ts = time.time()
            if miss:
                E = extract_embeddings_batch([crops[i] for i in miss], model, proc,
                                             batch_size=64, num_workers=10)
                Ef = extract_embeddings_batch(
                    [crops[i].transpose(Image.FLIP_LEFT_RIGHT) for i in miss],
                    model, proc, batch_size=64, num_workers=10)
                E = E + Ef
                E = E / E.norm(dim=1, keepdim=True).clamp_min(1e-12)
                _Em = E.numpy().astype("float32")
                for j, i in enumerate(miss):
                    CACHE[keys[i]] = _Em[j]
            En = np.stack([CACHE[k] for k in keys])
            T["embed"] += time.time() - _ts

            _ts = time.time()
            for i, (row, d, mid, crop) in enumerate(win_tasks):
                ev = En[i]
                # P2 Query Expansion: top-5 库图均值与查询向量等权融合再检
                sims = LIB_C @ ev
                if QE:
                    _t5 = np.argpartition(sims, -5)[-5:]
                    qv = ev + LIB_C[_t5].mean(axis=0)
                    qv /= max(float(np.linalg.norm(qv)), 1e-12)
                    sims = LIB_C @ qv
                # 内存矩阵暴力余弦 -> mid聚合(取最高分, 候选60条图)
                top = np.argpartition(sims, -60)[-60:]
                top = top[np.argsort(-sims[top])]
                best = {}
                for j in top:
                    m0 = str(LIB_MIDS[j])
                    if m0 not in best or sims[j] > best[m0]:
                        best[m0] = float(sims[j])
                det_name = str(d.get("name", "")).strip().lower()

                def same_class(mc):
                    cc = mid2class.get(mc)
                    if not cc:
                        return False
                    return det_name in {
                        g.lower() for g in CLASS_ALIAS.get(cc, [cc]) if g}

                mids = sorted(best, key=lambda m: -best[m])[:TOPK]
                soft = sorted(best, key=lambda m: -best[m] *
                              (1.06 if same_class(m) else 1.0))[:TOPK]
                hard = [m for m in mids if same_class(m)][:TOPK]

                # 质心旁路: 查询向量 x 2,638 质心全量余弦
                cen_sim = CEN_C @ ev
                order_c = np.argsort(-cen_sim)
                cen_top = [CEN_MIDS[i] for i in order_c[:20]]
                cen_rank = {m: i for i, m in enumerate(cen_top)}
                # 变体4 cfusion: CF_W*图像分 + (1-CF_W)*质心分, 叠加 soft 先验
                def cen_score(m):
                    return float(cen_sim[CEN_IDX[m]]) if m in CEN_IDX else 0.0
                cfusion = sorted(best, key=lambda m: -
                                 (CF_W * best[m] + (1 - CF_W) * cen_score(m)) *
                                 (1.06 if same_class(m) else 1.0))[:TOPK]
                # 变体5 RRF: 图像排名 + 质心排名 (k=60), 候选并集
                img_rank = {m: i for i, m in enumerate(mids)}
                cand = list(dict.fromkeys(list(best.keys()) + cen_top))
                rrf = sorted(cand, key=lambda m: -(
                    1.0 / (60 + img_rank.get(m, 60)) +
                    1.0 / (60 + cen_rank.get(m, 60))))[:TOPK]

                S["retr_n"] += 1
                for tag, seq, pfx in (("base", mids, "s"),
                                      ("soft", soft, "c"),
                                      ("hard", hard, "f"),
                                      ("cfusion", cfusion, "d"),
                                      ("rrf", rrf, "r")):
                    if seq[:1] == [mid]:
                        S[pfx + "1"] += 1
                    if mid in seq[:5]:
                        S[pfx + "5"] += 1
                    if mid in seq[:10]:
                        S[pfx + "10"] += 1
                g = mid2genus.get(mid)
                if g:
                    gm = [mid2genus.get(m, "") for m in mids]
                    if gm[:1] == [g]:
                        S["g1"] += 1
                    if g in gm[:5]:
                        S["g5"] += 1
                    if g in gm[:10]:
                        S["g10"] += 1
                row.update({"retrieval_top10": mids,
                            "retrieval_top10_soft": soft,
                            "retrieval_top10_hard": hard,
                            "retrieval_top10_cfusion": cfusion,
                            "retrieval_top10_rrf": rrf,
                            "det_box": d["box"],
                            "det_name": d.get("name", ""),
                            "det_conf": round(float(d.get("conf", 0)), 4)})
            T["retr"] += time.time() - _ts
        for row in win_rows:
            f_obj.write(json.dumps(row, ensure_ascii=False) + "\n")

        if si // 100 != last_rep:
            last_rep = si // 100
            n = max(S["retr_n"], 1)
            print(f"  {si}/{len(scenes)} 场景 {time.time()-t0:.0f}s | "
                  f"recall {S['matched']}/{S['gt_lib']} "
                  f"| 物种 t1 {S['s1']/n*100:.1f} t5 {S['s5']/n*100:.1f} "
                  f"t10 {S['s10']/n*100:.1f}", flush=True)

    S["det_unmatched"] = S["det_total"] - S["matched"]
    # E3: 查询嵌入缓存落盘 (重跑基准时命中则跳过全部推理)
    if CACHE:
        _ks = list(CACHE.keys())
        np.savez(CACHE_PATH, k=np.array(_ks),
                 v=np.stack([CACHE[k] for k in _ks]).astype("float32"))
    # 检测框缓存落盘
    if DET_CACHE:
        pickle.dump({"sig": DET_SIG, "dets": DET_CACHE},
                    open(DET_CACHE_PATH, "wb"))

    n = max(S["retr_n"], 1)
    gl = max(S["gt_lib"], 1)
    cn = max(S["coarse_n"], 1)
    report = {
        "stats": S,
        "detection_recall@0.5": S["matched"] / gl,
        "coarse34_acc": S["coarse_ok"] / cn,
        "species_top1": S["s1"] / n,
        "species_top5": S["s5"] / n,
        "species_top10": S["s10"] / n,
        "species_top1_soft": S["c1"] / n,
        "species_top5_soft": S["c5"] / n,
        "species_top10_soft": S["c10"] / n,
        "species_top1_hard": S["f1"] / n,
        "species_top5_hard": S["f5"] / n,
        "species_top10_hard": S["f10"] / n,
        "species_top1_cfusion": S["d1"] / n,
        "species_top5_cfusion": S["d5"] / n,
        "species_top10_cfusion": S["d10"] / n,
        "species_top1_rrf": S["r1"] / n,
        "species_top5_rrf": S["r5"] / n,
        "species_top10_rrf": S["r10"] / n,
        "genus_top1": S["g1"] / n,
        "genus_top5": S["g5"] / n,
        "genus_top10": S["g10"] / n,
        "elapsed_s": time.time() - t0,
        "qe": QE,
        "t_yolo": round(T["yolo"], 1),
        "t_embed": round(T["embed"], 1),
        "t_retr": round(T["retr"], 1),
    }
    json.dump(report, f_rep, ensure_ascii=False, indent=1)
    f_rep.close()
    f_obj.close()
    drv.close()
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
