#!/usr/bin/env python3
"""12c_embed_boxes.py — FathomNet 框切图 BioCLIP 2.5 嵌入提取 (双卡分片)

用法:
  python scripts/12c_embed_boxes.py --shard 0 &
  python scripts/12c_embed_boxes.py --shard 1 &
输出: data/processed/bioclip25_embeddings/emb_boxes_shard{0,1}.npz
      (image_paths: N 个 box_rel, embeddings: N×1024 L2 归一化, 与 09c 字段对齐)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
from utils.bioclip_embed import get_bioclip_model, extract_embeddings_batch  # noqa: E402

MANIFEST = ROOT / 'data/processed/fathomnet_expansion/boxes_manifest.json'
OUT_DIR = ROOT / 'data/processed/bioclip25_embeddings'
CKPT_EVERY = 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True, choices=[0, 1])
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = args.device or f'cuda:{args.shard}'
    man = json.load(open(MANIFEST))
    paths = sorted(r['box_rel'] for r in man)
    shard = paths[args.shard::2]                     # 奇偶分片
    print(f'[shard {args.shard}] {len(shard)}/{len(paths)} 框, device={device}')

    model, processor = get_bioclip_model(device=device)
    model.eval()

    out_path = OUT_DIR / f'emb_boxes_shard{args.shard}.npz'
    # 断点续跑: 已存 npz 的 paths/embs 直接续写
    all_paths, all_embs = [], []
    if out_path.exists():
        z = np.load(out_path)
        all_paths = z['image_paths'].tolist()
        all_embs = [z['embeddings']]
        print(f'  断点: 已有 {len(all_paths)} 嵌入')
    done = set(all_paths)
    todo = [p for p in shard if p not in done]
    print(f'  待提 {len(todo)}')

    t0 = time.time()
    B = 64
    for s in range(0, len(todo), B):
        chunk = todo[s:s + B]
        embs = extract_embeddings_batch([str(ROOT / p) for p in chunk],
                                        model, processor, batch_size=B, device=device)
        all_paths.extend(chunk)
        all_embs.append(embs.numpy() if isinstance(embs, torch.Tensor) else np.asarray(embs))
        if (s // B) % 40 == 0 and s > 0:
            rate = s / (time.time() - t0)
            eta = (len(todo) - s) / max(rate, 0.1) / 60
            print(f'  {s}/{len(todo)}, {rate:.1f} 图/s, ETA {eta:.0f}min')
        if (s + B) % CKPT_EVERY < B or s + B >= len(todo):
            tmp = out_path.with_suffix('.tmp.npz')
            np.savez(tmp, image_paths=np.array(all_paths),
                     embeddings=np.concatenate(all_embs, axis=0))
            os.replace(tmp, out_path)
    embs = np.concatenate(all_embs, axis=0)
    tmp = out_path.with_suffix('.tmp.npz')
    np.savez(tmp, image_paths=np.array(all_paths), embeddings=embs)
    os.replace(tmp, out_path)
    print(f'[shard {args.shard}] 完成: {embs.shape}, L2范数样本 {np.linalg.norm(embs[0]):.4f} → {out_path}')


if __name__ == '__main__':
    main()
