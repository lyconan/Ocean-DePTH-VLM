#!/usr/bin/env python3
"""13c_embed_yolo_boxes.py — JEDI/深海框切图 BioCLIP 2.5 嵌入 (双卡奇偶分片)

用法 (容器内): export PYTHONPATH=项目根
  CUDA_VISIBLE_DEVICES=0 python scripts/13c_embed_yolo_boxes.py --shard 0
  CUDA_VISIBLE_DEVICES=1 python scripts/13c_embed_yolo_boxes.py --shard 1
输出: data/processed/bioclip25_embeddings/emb_yolo_shard{0,1}.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.utils.bioclip_embed import get_bioclip_model, extract_embeddings_batch

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
MAN = ROOT / 'data/processed/yolo_boxes_manifest.json'
EMB_DIR = ROOT / 'data/processed/bioclip25_embeddings'
B = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    args = ap.parse_args()
    dev = f'cuda:{args.shard}'

    man = json.load(open(MAN))
    rels = [r['box_rel'] for r in man]
    mine = rels[args.shard::2]
    missing = [r for r in mine if not (ROOT / r).exists()]
    assert not missing, f'缺图 {len(missing)}: {missing[:3]}'
    print(f'[shard{args.shard}] {len(mine)}/{len(rels)} 图, device={dev}')

    model, processor = get_bioclip_model(device=dev)
    embs = extract_embeddings_batch([str(ROOT / r) for r in mine], model, processor,
                                    batch_size=B, device=dev)
    embs = embs.numpy() if hasattr(embs, 'numpy') else np.asarray(embs)
    embs = embs.astype(np.float32)
    norms = np.linalg.norm(embs, axis=1)
    print(f'[shard{args.shard}] 嵌入 {embs.shape}, L2 {norms.min():.4f}~{norms.max():.4f}')

    out = EMB_DIR / f'emb_yolo_shard{args.shard}.npz'
    np.savez(out, image_paths=np.array(mine), embeddings=embs)
    chk = np.load(out, allow_pickle=True)
    assert chk['embeddings'].shape == (len(mine), 1024), '落盘校验失败'
    print(f'[shard{args.shard}] [完成] {out}, 断言通过')


if __name__ == '__main__':
    main()
