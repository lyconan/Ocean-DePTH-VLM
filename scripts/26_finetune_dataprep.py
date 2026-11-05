#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26_finetune_dataprep.py — BioCLIP 2.5 LoRA 微调数据准备 (宿主机运行, 纯 stdlib)

输入:
  - data/processed/fathomnet_expansion/boxes/<MID>/{train_*, val_*}.jpg  (27,342张/1,543类)
    文件名 train_/val_ 前缀即防泄漏划分 (按源图划分, 同一源图的框不会跨集合)
  - /tmp/atlas_pairs_raw.txt  (cypher-shell 导出的 atlas image_id -> msp_id, 1,600条)

输出: /tmp/exp/ft_data.json
  train: [{p:相对路径, y:label_idx, w:采样权重}]
    - 大类每类随机截断至 MAX_PER_CLASS=32
    - 采样权重 = 1/sqrt(类内样本数) (长尾平衡), 图集样本额外乘 ATLAS_WEIGHT=0.15
    - 标签空间 = FN mids ∪ atlas mids (图集纳入标签空间: ArcFace 把同 mid 多张插画拉近,
      直接提升图集层检索; 与 FN 仅 54 个 mid 重叠, 干扰有限)
  val:   [{p, y}]  (FN val 前缀, 且 mid 在训练标签空间内)
"""
import json
import os
import random
import re
from collections import Counter, defaultdict

random.seed(42)

WS = "/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM"
FN_DIR = os.path.join(WS, "data/processed/fathomnet_expansion/boxes")
OUT = "/tmp/exp/ft_data.json"
ATLAS_PAIRS = "/tmp/atlas_pairs_raw.txt"
MAX_PER_CLASS = 64
ATLAS_WEIGHT = 0.15
EXTS = (".jpg", ".jpeg", ".png")

def main():
    # 1) 扫描 FN 框图
    train_by_mid = defaultdict(list)  # mid -> [relpath]
    val_by_mid = defaultdict(list)
    for mid in sorted(os.listdir(FN_DIR)):
        d = os.path.join(FN_DIR, mid)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.lower().endswith(EXTS):
                continue
            rel = f"data/processed/fathomnet_expansion/boxes/{mid}/{f}"
            if f.startswith("train_"):
                train_by_mid[mid].append(rel)
            elif f.startswith("val_"):
                val_by_mid[mid].append(rel)

    n_tr_raw = sum(len(v) for v in train_by_mid.values())
    n_va_raw = sum(len(v) for v in val_by_mid.values())
    print(f"FN: train {n_tr_raw} / val {n_va_raw} / mids {len(train_by_mid)}")

    # 2) 标签空间 = 有 train 图的 mid; val 中不在标签空间的丢弃
    labels = sorted(train_by_mid.keys())
    lab2idx = {m: i for i, m in enumerate(labels)}
    val_mids_dropped = [m for m in val_by_mid if m not in lab2idx]
    print(f"标签空间 {len(labels)} 类; val 中无 train 图的 mid: {len(val_mids_dropped)} 个 "
          f"({sum(len(val_by_mid[m]) for m in val_mids_dropped)} 张图丢弃)")

    # 3) 大类截断 + 构建训练清单 (FN)
    train = []
    cls_count = {}
    for mid in labels:
        imgs = sorted(train_by_mid[mid])
        if len(imgs) > MAX_PER_CLASS:
            imgs = random.sample(imgs, MAX_PER_CLASS)
        cls_count[mid] = len(imgs)
        y = lab2idx[mid]
        for p in imgs:
            train.append({"p": p, "y": y, "w": 1.0, "src": "fn"})
    n_fn = len(train)
    n_fn_classes = len(labels)

    # 4) 图集样本: 全部 atlas mids 纳入标签空间 (与 FN 重叠的 mid 直接用)
    atlas_items = []  # (iid, mid)
    with open(ATLAS_PAIRS, encoding="utf-8") as f:
        lines = [l for l in f if l.strip() and not l.startswith("iid")]
    for l in lines:
        m = re.match(r'^"([^"]+)",\s*"([^"]*)"', l)
        if m:
            atlas_items.append((m.group(1), m.group(2)))

    atlas_mids = sorted({mid for _, mid in atlas_items if mid} -
                        set(labels))
    labels = labels + atlas_mids
    lab2idx.update({m: i for i, m in enumerate(atlas_mids, start=len(lab2idx))})
    cls_count.update({m: 0 for m in atlas_mids})
    print(f"标签空间扩展: +{len(atlas_mids)} 个 atlas 独有 mid -> 共 {len(labels)} 类")

    for iid, mid in atlas_items:
        if mid not in lab2idx:
            continue
        full = os.path.join(WS, iid)
        if not os.path.isfile(full):
            continue
        train.append({"p": iid, "y": lab2idx[mid], "w": 1.0, "src": "atlas"})
        cls_count[mid] += 1
    n_atlas = len(train) - n_fn
    atlas_mid_cov = sum(1 for m in atlas_mids if cls_count[m] > 0)

    # 5) val 清单
    val = []
    for mid in labels:
        for p in sorted(val_by_mid[mid]):
            val.append({"p": p, "y": lab2idx[mid]})
    n_va = len(val)

    # 6) 采样权重: FN = 1/sqrt(类内样本数); atlas = ATLAS_WEIGHT * 1/sqrt(类内样本数)
    for t in train:
        t["w"] = 1.0 / (cls_count[labels[t["y"]]] ** 0.5)
        if t["src"] == "atlas":
            t["w"] *= ATLAS_WEIGHT

    dist = Counter(cls_count.values())
    print(f"FN 训练 {n_fn} 张 ({n_fn_classes} 类, 类分布: min {min(cls_count[m] for m in labels[:n_fn_classes])}, "
          f"中位 {sorted(cls_count[m] for m in labels[:n_fn_classes])[n_fn_classes//2]})")
    print(f"图集样本 {n_atlas} 张, 覆盖 {atlas_mid_cov} 个 atlas mid (ATLAS_WEIGHT={ATLAS_WEIGHT})")
    print(f"val {n_va} 张")
    print(f"训练清单合计 {len(train)} 条")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"train": train, "val": val, "labels": labels,
                   "cls_count": cls_count}, f, ensure_ascii=False)
    print(f"已写 {OUT}")

if __name__ == "__main__":
    main()
