#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27_finetune_lora.py — BioCLIP 2.5 视觉塔 LoRA + ArcFace 微调 (容器内 GPU0)
运行: docker run --rm --gpus '"device=0"' --shm-size 8g \
        -v /tmp/exp:/exp -v $WS:/work:ro \
        -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
        -e PYTHONPATH=/exp/pylibs \
        yolo34_train:v1 /root/miniconda3/envs/vllm/bin/python /work/scripts/27_finetune_lora.py

方案:
  - peft LoRA 包视觉塔 32 层 MultiheadAttention (q/k/v/out_proj), rank=16, alpha=32, dropout=0.1
  - ArcFace 主损失 (s=30, m=0.35), 类空间 2357 (FN 1509 + 图集 848)
  - 蒸馏正则: 0.3 * mean(1 - cos(LoRA嵌入, 原始嵌入)) 防遗忘 (原始塔=注入前 deepcopy, 冻结)
  - 加权采样: FN 1/sqrt(类内样本数), 图集额外 x0.15
  - bf16 autocast, AdamW lr 1e-4, cosine + warmup 100 步, 3 epochs, batch 64
  - 每 epoch 在 FN val (4045张, 文件名 val_ 前缀, 防泄漏) 上评估 top-1/top-5
产出: /exp/lora_adapter/ (peft adapter) + /exp/ft_head.pt (ArcFace权重) + /exp/ft_log.jsonl
"""
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import open_clip
import peft

EXP = "/exp"
WS = "/work"
DEVICE = "cuda:0"
SEED = 42

R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
EPOCHS = 6
BATCH = 64
LR = 1e-4
WEIGHT_DECAY = 0.0
WARMUP = 100
S_ARC = 30.0
M_ARC = 0.30
DISTILL_COEF = 0.1
NUM_WORKERS = 8

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class ArcFace(nn.Module):
    """标准 ArcFace: 归一化特征与类权重, 对目标类加角距 m, easy-margin 裁剪."""
    def __init__(self, dim, n_cls, s=S_ARC, m=M_ARC):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, feats, y=None):
        Wn = F.normalize(self.W, dim=-1)
        fn = F.normalize(feats, dim=-1)
        cos = (fn @ Wn.t()).float()  # autocast 下 matmul 可能出 bf16, 统一回 fp32
        if y is None:
            return self.s * cos
        th = cos.gather(1, y.view(-1, 1))
        if M_ARC > 0:
            # cos(theta + m) = cos*m - sin*sin_m, easy margin 裁剪
            th_m = math.cos(M_ARC) * th - math.sin(M_ARC) * torch.sqrt((1.0 - th ** 2).clamp(1e-9))
            th_m = torch.where(th > math.cos(math.pi - M_ARC), th_m, th)
            cos = cos.scatter(1, y.view(-1, 1), th_m)
        return self.s * cos

class ManifestDataset(Dataset):
    def __init__(self, items, proc):
        self.items = items
        self.proc = proc

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(os.path.join(WS, it["p"])).convert("RGB")
        return self.proc(img), it["y"], it.get("w", 1.0)

@torch.no_grad()
def evaluate(val_items, visual, head, proc):
    visual.eval()
    ds = ManifestDataset(val_items, proc)
    dl = DataLoader(ds, batch_size=96, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    t1 = t5 = n = 0
    for x, y, _ in dl:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = visual(x).float()
        logits = head(f)  # s*cos, 排序与 cos 一致
        t5k = torch.topk(logits, 5, dim=1).indices
        t1 += (t5k[:, 0] == y.to(DEVICE)).sum().item()
        t5 += (t5k == y.to(DEVICE).view(-1, 1)).any(1).sum().item()
        n += len(y)
    visual.train()
    return t1 / n, t5 / n, n

def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    data = json.load(open(f"{EXP}/ft_data.json"))
    train_items, val_items = data["train"], [v for v in data["val"]]
    labels = data["labels"]
    n_cls = len(labels)
    print(f"数据: train {len(train_items)} / val {len(val_items)} / 类 {n_cls}", flush=True)

    model, _, proc = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2.5-vith14")
    model = model.float()

    # 蒸馏基准: 注入 LoRA 前的原始视觉塔 (冻结, eval)
    base_visual = copy.deepcopy(model.visual).to(DEVICE)
    base_visual.eval()
    for p in base_visual.parameters():
        p.requires_grad_(False)

    visual = peft.get_peft_model(model.visual, peft.LoraConfig(
        r=R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["attn", "c_fc", "c_proj"])).to(DEVICE)
    n_train = sum(p.numel() for p in visual.parameters() if p.requires_grad)
    print(f"LoRA 可训练参数: {n_train/1e6:.2f}M", flush=True)

    head = ArcFace(1024, n_cls).to(DEVICE)

    params = [p for p in visual.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(train_items) / BATCH)
    total_steps = EPOCHS * steps_per_epoch
    def lr_at(step):
        if step < WARMUP:
            return LR * step / max(WARMUP, 1)
        t = (step - WARMUP) / max(total_steps - WARMUP, 1)
        return LR * 0.5 * (1 + math.cos(math.pi * t))

    # 加权采样 (manifest 的 w 已含类平衡与图集权重)
    weights = torch.tensor([it.get("w", 1.0) for it in train_items], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)

    log_f = open(f"{EXP}/ft_log.jsonl", "a", encoding="utf-8")
    def log(d):
        d["t"] = time.strftime("%H:%M:%S")
        log_f.write(json.dumps(d, ensure_ascii=False) + "\n"); log_f.flush()
        print(json.dumps(d, ensure_ascii=False), flush=True)

    cfg = {"r": R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "epochs": EPOCHS,
           "batch": BATCH, "lr": LR, "warmup": WARMUP, "s": S_ARC, "m": M_ARC,
           "distill": DISTILL_COEF, "n_cls": n_cls, "n_train": len(train_items),
           "n_val": len(val_items), "seed": SEED}
    json.dump(cfg, open(f"{EXP}/ft_config.json", "w"), ensure_ascii=False, indent=1)

    # epoch 0 基线评估 (未训练)
    top1, top5, n = evaluate(val_items, visual, head, proc)
    log({"ev": "pre", "top1": round(top1, 4), "top5": round(top5, 4), "n": n})
    best_top1 = top1

    gstep = 0
    for ep in range(EPOCHS):
        dl = DataLoader(ManifestDataset(train_items, proc), batch_size=BATCH,
                        sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True,
                        drop_last=True, persistent_workers=True)
        t0 = time.time()
        run_ce = run_d = run_l = 0.0
        nb = 0
        for x, y, _ in dl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = visual(x).float()
                logits = head(f, y)
                loss_ce = F.cross_entropy(logits, y)
                with torch.no_grad():
                    f_base = base_visual(x).float()
                f_n = F.normalize(f, dim=-1)
                fb_n = F.normalize(f_base, dim=-1)
                loss_d = (1 - (f_n * fb_n).sum(-1)).mean()
                loss = loss_ce + DISTILL_COEF * loss_d

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            run_ce += loss_ce.item(); run_d += loss_d.item(); run_l += loss.item(); nb += 1
            gstep += 1
            if gstep % 50 == 0:
                log({"ev": "train", "ep": ep, "step": gstep, "lr": round(lr_at(gstep), 7),
                     "ce": round(run_ce/nb, 4), "distill": round(run_d/nb, 4),
                     "loss": round(run_l/nb, 4), "sec": round(time.time()-t0, 1)})
                run_ce = run_d = run_l = 0.0; nb = 0

        top1, top5, n = evaluate(val_items, visual, head, proc)
        log({"ev": "post", "ep": ep, "top1": round(top1, 4), "top5": round(top5, 4),
             "n": n, "sec": round(time.time()-t0, 1)})
        # 保存最优与最终
        if top1 >= best_top1 or ep == EPOCHS - 1:
            visual.save_pretrained(f"{EXP}/lora_adapter_v2")
            torch.save({"W": head.W.detach().cpu(), "labels": labels},
                       f"{EXP}/ft_head.pt")
            best_top1 = max(best_top1, top1)
            log({"ev": "saved", "ep": ep, "top1": round(top1, 4)})

    log({"ev": "done", "best_top1": round(best_top1, 4)})

if __name__ == "__main__":
    main()
