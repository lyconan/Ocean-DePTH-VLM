#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28_ft_verify.py — LoRA 微调后验证 (容器内 GPU0)
步骤:
  1) 用 LoRA 适配器重建视觉塔, 全库 36,147 张重提 -> /exp/emb_ft.f32
  2) 同 /exp/queries.json 查询集 + 同协议 (base 变体, S1 原图NN排除自身)
     与基线 /exp/results.json 的 "<stratum>|base|S1" 分层对比
产出: /exp/ft_verify.json + 控制台对比表
"""
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import open_clip
import peft

EXP = "/exp"
WS = "/work"
DIM = 1024
DEVICE = "cuda:0"
BS = 64


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


class ImgDS(Dataset):
    def __init__(self, paths, proc):
        self.paths = paths
        self.proc = proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.proc(Image.open(self.paths[i]).convert("RGB"))


def main():
    t0 = time.time()
    meta = [json.loads(l) for l in open(f"{EXP}/meta.jsonl")]
    ids = [m["id"] for m in meta]
    mids = [m["mid"] for m in meta]
    id2idx = {x: i for i, x in enumerate(ids)}
    print(f"库: {len(ids)} 张", flush=True)

    model, _, proc = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2.5-vith14")
    model = model.float()
    visual = peft.PeftModel.from_pretrained(model.visual, f"{EXP}/lora_adapter")
    visual = visual.eval().to(DEVICE)
    print("LoRA 适配器加载成功", flush=True)

    # 1) 全库重提
    out_path = f"{EXP}/emb_ft.f32"
    if not os.path.exists(out_path):
        paths = [os.path.join(WS, i) for i in ids]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"警告: {len(missing)} 张文件缺失, 示例 {missing[:2]}", flush=True)
        feats = []
        dl = DataLoader(ImgDS(paths, proc), batch_size=BS, shuffle=False,
                        num_workers=8, pin_memory=True)
        with torch.no_grad():
            for k, x in enumerate(dl):
                x = x.to(DEVICE, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = visual(x).float()
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy())
                if (k + 1) % 100 == 0:
                    print(f"  {len(feats)*BS}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
        mat = np.concatenate(feats, 0).astype(np.float32)
        mat.tofile(out_path)
        print(f"已写 {out_path} {mat.shape}", flush=True)
    mat = np.fromfile(out_path, dtype=np.float32).reshape(-1, DIM)
    assert len(mat) == len(ids), f"{len(mat)} vs {len(ids)}"
    mat = l2(mat)
    mat_t = torch.from_numpy(mat).to(DEVICE)

    # 2) 查询集同协议对比
    queries = json.load(open(f"{EXP}/queries.json"))
    results_base = json.load(open(f"{EXP}/results.json"))
    print(f"查询 {len(queries)} 条, 开始对比...", flush=True)

    res = {}
    for stratum in sorted({q["stratum"] for q in queries}):
        qs = [q for q in queries if q["stratum"] == stratum]
        Q = []
        for q in qs:
            pil = Image.open(os.path.join(WS, q["qid"])).convert("RGB")
            x = proc(pil).unsqueeze(0)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = visual(x.to(DEVICE)).float()
                f = f / f.norm(dim=-1, keepdim=True)
            Q.append(f.squeeze(0).cpu().numpy())
        Q = torch.from_numpy(np.stack(Q)).to(DEVICE)
        sims = Q @ mat_t.T
        for r, q in enumerate(qs):
            sims[r, id2idx[q["qid"]]] = -1.0
        top5 = sims.topk(5, dim=1).indices.cpu().numpy()
        t1 = t5 = 0
        for r, q in enumerate(qs):
            hit = [mids[top5[r][k]] for k in range(5)]
            if hit[0] == q["mid"]:
                t1 += 1
            if q["mid"] in hit:
                t5 += 1
        nq = len(qs)
        res[stratum] = {"n": nq, "top1": t1 / nq, "top5": t5 / nq}
        base = results_base.get(f"{stratum}|base|S1", {})
        b1 = base.get("top1", 0) * 100
        b5 = base.get("top5", 0) * 100
        print(f"{stratum}: n={nq} | base top-1 {b1:.1f}% -> ft {t1/nq*100:.1f}% "
              f"({(t1/nq-b1/100)*100:+.1f}pp) | top-5 {b5:.1f}% -> {t5/nq*100:.1f}% "
              f"({(t5/nq-b5/100)*100:+.1f}pp)", flush=True)

    # 总体 (加权)
    n = sum(r["n"] for r in res.values())
    t1 = sum(r["top1"] * r["n"] for r in res.values()) / n
    t5 = sum(r["top5"] * r["n"] for r in res.values()) / n
    print(f"总体(加权): top-1 {t1*100:.1f}% | top-5 {t5*100:.1f}%", flush=True)
    json.dump({"ft": res, "overall": {"top1": t1, "top5": t5}},
              open(f"{EXP}/ft_verify.json", "w"), ensure_ascii=False, indent=1)
    print(f"已写 /exp/ft_verify.json  总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
