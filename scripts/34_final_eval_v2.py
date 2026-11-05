#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
34_final_eval_v2.py — v2 适配器全库重提 + 多口径终评 (容器内 GPU0)
1) lora_adapter_v2 重提 36,147 张 -> /exp/emb_ft2.f32
2) 三层新随机样本 (n=500/层, 排除自身) 图像级 top-1/top-5
3) FN 层多口径: 图像级 / +翻转TTA / +融合重排(top-50, a=0.5) / TTA+重排
产出: /exp/emb_ft2.f32, /exp/final_eval.json
"""
import json
import os
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import open_clip
import peft

EXP = "/exp"
WS = "/work"
DIM = 1024
DEVICE = "cuda:0"
BS = 64
A_FUSION = 0.5


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


class ImgDS(Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.proc(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def embed_query(visual, proc, path):
    pil = Image.open(path).convert("RGB")
    x = proc(pil).unsqueeze(0).to(DEVICE)
    xf = proc(pil.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = visual(x).float()
        ff = visual(xf).float()
    f = f / f.norm(dim=-1, keepdim=True)
    ff = ff / ff.norm(dim=-1, keepdim=True)
    return f.squeeze(0).cpu().numpy(), ff.squeeze(0).cpu().numpy()


def main():
    t0 = time.time()
    meta = [json.loads(l) for l in open(f"{EXP}/meta.jsonl")]
    ids = np.array([m["id"] for m in meta])
    mids = np.array([m["mid"] for m in meta])
    uniq = np.array(sorted(set(mids)))
    mid2u = {m: i for i, m in enumerate(uniq)}
    u_of = np.array([mid2u[m] for m in mids])

    model, _, proc = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2.5-vith14")
    model = model.float()
    visual = peft.PeftModel.from_pretrained(model.visual, f"{EXP}/lora_adapter_v2").eval().to(DEVICE)
    print("v2 适配器加载成功", flush=True)

    out_path = f"{EXP}/emb_ft2.f32"
    if not os.path.exists(out_path):
        paths = [os.path.join(WS, i) for i in ids]
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
                    print(f"  重提 {len(feats)*BS}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
        np.concatenate(feats, 0).astype(np.float32).tofile(out_path)
        print(f"已写 {out_path}", flush=True)
    ft = l2(np.fromfile(out_path, dtype=np.float32).reshape(-1, DIM))

    # 质心 (融合重排用)
    C = np.zeros((len(uniq), DIM), dtype=np.float32)
    for u in range(len(uniq)):
        C[u] = ft[u_of == u].mean(0)
    C = l2(C)

    res = {}
    STR = {"FN框图": "data/processed/fathomnet_expansion/boxes/",
           "YOLO框图": "data/processed/yolo_boxes/",
           "图集插图": "data/processed/atlas_lookup/"}
    rng = np.random.default_rng(2026)
    for name, pre in STR.items():
        mask = np.array([i.startswith(pre) for i in ids])
        cnt = Counter(mids[mask])
        ok = [i for i in np.where(mask)[0] if cnt[mids[i]] >= 2]
        sel = rng.choice(ok, min(500, len(ok)), replace=False)

        M = torch.from_numpy(ft).cuda()
        Q = torch.from_numpy(ft[sel]).cuda()
        sims = Q @ M.T
        for r, i in enumerate(sel):
            sims[r, i] = -1
        top5 = sims.topk(5, 1).indices.cpu().numpy()
        t1 = sum(1 for r, i in enumerate(sel) if mids[top5[r][0]] == mids[i])
        t5 = sum(1 for r, i in enumerate(sel) if mids[i] in [mids[top5[r][k]] for k in range(5)])
        res[name] = {"n": len(sel), "top1": t1 / len(sel), "top5": t5 / len(sel)}
        print(f"{name} 图像级: n={len(sel)} top-1 {t1/len(sel)*100:.1f}% top-5 {t5/len(sel)*100:.1f}%", flush=True)

        if name == "FN框图":
            # --- FN 多口径 ---
            K = 50
            topK = sims.topk(K, 1).indices.cpu().numpy()
            cand_sim = np.take_along_axis(sims.cpu().numpy(), topK, axis=1)
            n1 = n5 = nt1 = nt5 = 0
            for r, qi in enumerate(sel):
                q, qf = embed_query(visual, proc, os.path.join(WS, ids[qi]))
                q_tta = l2((q + qf)[None])[0]
                # TTA 图像级 top5 (整库)
                s_tta = ft @ q_tta
                s_tta[qi] = -1
                t5t = np.argsort(-s_tta)[:5]
                if mids[t5t[0]] == mids[qi]:
                    nt1 += 1
                if mids[qi] in [mids[t5t[k]] for k in range(5)]:
                    nt5 += 1
                # 融合重排 (图像级 top-50 候选, 用 TTA 查询向量)
                cand = topK[r]
                cu = u_of[cand]
                cs = cand_sim[r]
                score_u = {}
                for u in np.unique(cu):
                    score_u[u] = A_FUSION * cs[cu == u].max() + (1 - A_FUSION) * (q_tta @ C[u])
                top5u = sorted(score_u, key=score_u.get, reverse=True)[:5]
                q_u = mid2u[mids[qi]]
                if top5u[0] == q_u:
                    n1 += 1
                if q_u in top5u:
                    n5 += 1
            n = len(sel)
            res["FN+TTA"] = {"n": n, "top1": nt1 / n, "top5": nt5 / n}
            res["FN+TTA+融合重排"] = {"n": n, "top1": n1 / n, "top5": n5 / n}
            print(f"FN +翻转TTA: top-1 {nt1/n*100:.1f}% top-5 {nt5/n*100:.1f}%", flush=True)
            print(f"FN +TTA+融合重排(a={A_FUSION}): top-1 {n1/n*100:.1f}% top-5 {n5/n*100:.1f}%", flush=True)

    json.dump(res, open(f"{EXP}/final_eval.json", "w"), ensure_ascii=False, indent=1)
    print(f"完成, 总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
