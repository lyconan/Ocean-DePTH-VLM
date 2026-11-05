#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
37_closedbook_eval.py — 488 个"仅 train 框"类别的闭卷口径评测 (附录用)
背景: 这些类别的全部图像都在 LoRA 训练集+画廊中, 无未见样本, 无法盲测;
      用类内留一自检索 (排除自身, top-5 同 mid 命中) 给出参考值, 报告中必须标注"闭卷"
协议: 每 mid 最多 10 张库图作查询, 翻转TTA, 向量索引 top-6 排自身后取 top-5
运行: 容器 GPU0, 同 36 号脚本启动方式
"""
import os
import numpy as np
from PIL import Image
from neo4j import GraphDatabase
import sys
sys.path.insert(0, "/work/scripts")
from utils.bioclip_embed import get_bioclip_model, extract_embedding

BOX_DIR = "/work/data/processed/fathomnet_expansion/boxes"
URI = "bolt://depth_vlm_neo4j:7687"
AUTH = ("neo4j", "password")
CAP_PER_MID = 10


def main():
    # 1. 找出"只有 train 框、无 val 框"的 mids
    train_only = []
    for mid in sorted(os.listdir(BOX_DIR)):
        d = os.path.join(BOX_DIR, mid)
        if not os.path.isdir(d):
            continue
        fns = [f for f in os.listdir(d) if f.endswith(".jpg")]
        has_val = any(f.startswith("val_") for f in fns)
        if not has_val:
            train_only.append(mid)
    print(f"仅 train 框的 mids: {len(train_only)}", flush=True)

    model, proc = get_bioclip_model()
    drv = GraphDatabase.driver(URI, auth=AUTH)
    t1 = t5 = n = 0
    sims = []
    with drv.session() as s:
        for mi, mid in enumerate(train_only):
            d = os.path.join(BOX_DIR, mid)
            fns = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))[:CAP_PER_MID]
            for fn in fns:
                p = os.path.join(d, fn)
                pil = Image.open(p).convert("RGB")
                e = extract_embedding(pil, model, proc)
                ef = extract_embedding(pil.transpose(Image.FLIP_LEFT_RIGHT), model, proc)
                e = e + ef
                e = e / e.norm().clamp_min(1e-12)
                rows = s.run("""
                    CALL db.index.vector.queryNodes('entryimage_embeddings', 6, $qv)
                    YIELD node, score
                    RETURN node.msp_id AS mid, node.image_id AS id, score
                """, qv=e.tolist()).data()
                hits = [r for r in rows if r["id"] != p]
                if not hits:
                    continue
                n += 1
                mids5 = [h["mid"] for h in hits[:5]]
                if mids5[0] == mid:
                    t1 += 1
                if mid in mids5:
                    t5 += 1
                sims.append(hits[0]["score"])
            if (mi + 1) % 100 == 0:
                print(f"  {mi+1}/{len(train_only)} mids | 当前 top-1 {t1/max(n,1)*100:.1f}%", flush=True)
    drv.close()
    print(f"\n=== 闭卷口径 (488 仅train类, 排自身) ===")
    print(f"查询 {n} | top-1 {t1}/{n} ({t1/max(n,1)*100:.1f}%) | "
          f"top-5 {t5}/{n} ({t5/max(n,1)*100:.1f}%) | 平均top-1相似度 {np.mean(sims):.4f}")


if __name__ == "__main__":
    main()
