#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""30_offline_recheck.py — 离线精确检索复检 (不经向量索引, 排除索引脏数据干扰)
模仿 23 号脚本口径: 每层 rand 抽 500 (仅 mid 总图数>=2), 排除自身, top-5 同 mid 命中。
对比 base (emb.f32) vs ft (emb_ft.f32)。
"""
import json
from collections import Counter

import numpy as np
import torch

DIM = 1024

def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)

STR = {"FN框图": "data/processed/fathomnet_expansion/boxes/",
       "YOLO框图": "data/processed/yolo_boxes/",
       "图集插图": "data/processed/atlas_lookup/"}

def main():
    meta = [json.loads(l) for l in open("/exp/meta.jsonl")]
    ids = np.array([m["id"] for m in meta])
    mids = np.array([m["mid"] for m in meta])
    old = l2(np.fromfile("/exp/emb.f32", dtype=np.float32).reshape(-1, DIM))
    ft = l2(np.fromfile("/exp/emb_ft.f32", dtype=np.float32).reshape(-1, DIM))
    rng = np.random.default_rng(2026)
    M_old = torch.from_numpy(old).cuda()
    M_ft = torch.from_numpy(ft).cuda()
    for name, pre in STR.items():
        mask = np.array([i.startswith(pre) for i in ids])
        idxs = np.where(mask)[0]
        cnt = Counter(mids[mask])
        ok = [i for i in idxs if cnt[mids[i]] >= 2]
        sel = rng.choice(ok, min(500, len(ok)), replace=False)
        for tag, mat, M in [("base", old, M_old), ("ft", ft, M_ft)]:
            Q = torch.from_numpy(mat[sel]).cuda()
            sims = Q @ M.T
            for r, i in enumerate(sel):
                sims[r, i] = -1
            t5 = sims.topk(5, 1).indices.cpu().numpy()
            t1 = sum(1 for r, i in enumerate(sel) if mids[t5[r][0]] == mids[i])
            t5h = sum(1 for r, i in enumerate(sel) if mids[i] in [mids[t5[r][k]] for k in range(5)])
            print(f"{name} {tag}: n={len(sel)} top-1 {t1/len(sel)*100:.1f}% top-5 {t5h/len(sel)*100:.1f}%", flush=True)

if __name__ == "__main__":
    main()
