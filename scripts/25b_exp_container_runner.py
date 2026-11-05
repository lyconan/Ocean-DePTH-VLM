#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25b_exp_container_runner.py — 检索提升实验 (容器内, GPU0)
输入: /exp/emb.f32 + /exp/meta.jsonl (全库嵌入), /exp/queries.json (查询集)
对比:
  查询侧变体: base / flip_tta / square / sq_tta / zoom / all_tta
  库侧检索:   S1 原图NN(排除自身) / S2 条目中心(留一校正)
指标: 分层 top-1/top-5 同条目命中率
"""
import os
import json
import numpy as np
import torch
from PIL import Image
import open_clip

EXP = "/exp"
WS = "/work"
DIM = 1024
DEVICE = "cuda:0"


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def load_matrix():
    mat = np.fromfile(f"{EXP}/emb.f32", dtype=np.float32).reshape(-1, DIM)
    meta = [json.loads(l) for l in open(f"{EXP}/meta.jsonl")]
    assert len(meta) == len(mat), f"{len(meta)} vs {len(mat)}"
    return mat, meta


def build_centroids(mat, mids):
    uniq = sorted(set(mids))
    idx_of = {m: i for i, m in enumerate(uniq)}
    C = np.zeros((len(uniq), DIM), dtype=np.float32)
    cnt = np.zeros(len(uniq), dtype=np.float32)
    mid_arr = np.array(mids)
    for i, u in enumerate(uniq):
        rows = mat[mid_arr == u]
        C[i] = rows.mean(axis=0)
        cnt[i] = len(rows)
    return l2(C), idx_of, cnt, uniq


def variants(pil):
    w, h = pil.size
    s = min(w, h)
    sq = pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    m = 0.075
    dw, dh = int(w * m), int(h * m)
    zoom = pil.crop((dw, dh, w - dw, h - dh))
    return pil, sq, zoom


def main():
    print("加载库矩阵...", flush=True)
    mat, meta = load_matrix()
    mat_n = l2(mat)
    mids = [m["mid"] for m in meta]
    ids = [m["id"] for m in meta]
    id2idx = {x: i for i, x in enumerate(ids)}
    C, idx_of, cnt, mids_uniq = build_centroids(mat, mids)
    C_t = torch.from_numpy(C).to(DEVICE)
    mat_t = torch.from_numpy(mat_n).to(DEVICE)
    print(f"库: {mat.shape[0]} 图 / {C.shape[0]} 条目中心", flush=True)

    queries = json.load(open(f"{EXP}/queries.json"))
    print(f"查询 {len(queries)} 条, 加载 BioCLIP 2.5...", flush=True)
    model, _, processor = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2.5-vith14")
    model = model.float().eval().to(DEVICE)

    @torch.no_grad()
    def embed(pil):
        x = processor(pil).unsqueeze(0).to(DEVICE)
        f = model.encode_image(x)
        return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).float().cpu().numpy()

    small_sides = []
    results = {}
    for stratum in sorted({q["stratum"] for q in queries}):
        qs = [q for q in queries if q["stratum"] == stratum]
        # 逐查询提特征 (变体 x 6)
        VAR = ["base", "flip_tta", "square", "sq_tta", "zoom", "all_tta"]
        E = {v: [] for v in VAR}
        qmeta = []
        for q in qs:
            path = os.path.join(WS, q["qid"])
            pil = Image.open(path).convert("RGB")
            small_sides.append(min(pil.size))
            base, sq, zoom = variants(pil)
            e_base = embed(base)
            e_flip = embed(base.transpose(Image.FLIP_LEFT_RIGHT))
            e_sq = embed(sq)
            e_sqf = embed(sq.transpose(Image.FLIP_LEFT_RIGHT))
            e_zoom = embed(zoom)
            E["base"].append(e_base)
            E["flip_tta"].append(l2(e_base + e_flip))
            E["square"].append(e_sq)
            E["sq_tta"].append(l2(e_sq + e_sqf))
            E["zoom"].append(e_zoom)
            E["all_tta"].append(l2(e_base + e_flip + e_sq + e_sqf))
            qmeta.append((q["qid"], q["mid"], id2idx[q["qid"]]))

        for var in VAR:
            Q = torch.from_numpy(np.stack(E[var])).to(DEVICE)
            for space in ["S1", "S2"]:
                if space == "S1":
                    sims = Q @ mat_t.T  # (nq, N)
                    for row, (qid, mid, qi) in enumerate(qmeta):
                        sims[row, qi] = -1.0
                    top5 = sims.topk(5, dim=1).indices.cpu().numpy()
                    hit_mids_all = mids  # 图像级: 索引 -> 该图 mid
                else:
                    sims = Q @ C_t.T  # (nq, n_mid)
                    for row, (qid, mid, qi) in enumerate(qmeta):
                        u = idx_of[mid]
                        n = int(cnt[u])
                        e_self = mat_t[qi]
                        c_loo = (C_t[u] * n - e_self) / max(n - 1, 1)
                        c_loo = c_loo / c_loo.norm().clip(1e-12)
                        sims[row, u] = Q[row] @ c_loo
                    top5 = sims.topk(5, dim=1).indices.cpu().numpy()
                    hit_mids_all = mids_uniq  # 条目级
                t1 = t5 = 0
                for row, (qid, mid, qi) in enumerate(qmeta):
                    hit_mids = [hit_mids_all[top5[row][k]] for k in range(5)]
                    if hit_mids[0] == mid:
                        t1 += 1
                    if mid in hit_mids:
                        t5 += 1
                nq = len(qmeta)
                results[f"{stratum}|{var}|{space}"] = {
                    "n": nq, "top1": t1 / nq, "top5": t5 / nq}
                print(f"{stratum} | {var:8s} | {space}: top-1 {t1/nq*100:5.1f}% | top-5 {t5/nq*100:5.1f}%", flush=True)

    small_sides.sort()
    n = len(small_sides)
    print(f"\n查询图短边: 中位 {small_sides[n//2]}px / p25 {small_sides[n//4]}px / p90 {small_sides[int(n*0.9)]}px")
    json.dump(results, open("/exp/results.json", "w"), ensure_ascii=False, indent=1)
    print("已存 /exp/results.json", flush=True)


if __name__ == "__main__":
    main()
