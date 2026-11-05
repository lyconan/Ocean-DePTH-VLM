#!/usr/bin/env python3
"""12f_embed_supplement_boxes.py — 补框切图 BioCLIP 2.5 嵌入 (双卡分片)

读 boxes_manifest_sup.json, 奇偶分片, 每卡一 shard。
输出: data/processed/bioclip25_embeddings/emb_boxes_sup_shard{0,1}.npz
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
SUP_MANIFEST = ROOT / 'data/processed/fathomnet_expansion/boxes_manifest_sup.json'
OUT_DIR = ROOT / 'data/processed/bioclip25_embeddings'
CKPT_EVERY = 3000


def load_model(device):
    from scripts.utils.bioclip_embed import get_bioclip_model
    return get_bioclip_model(device=device)


def run_shard(shard_id: int, device: str):
    man = json.load(open(SUP_MANIFEST))
    items = man[shard_id::2]                       # 奇偶分片
    out_npz = OUT_DIR / f'emb_boxes_sup_shard{shard_id}.npz'
    ckpt = OUT_DIR / f'emb_boxes_sup_shard{shard_id}.ckpt.npz'
    done = {}
    if ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        done = {p: e for p, e in zip(z['image_paths'], z['embeddings'])}
        print(f'[shard{shard_id}] 断点续跑: {len(done)}/{len(items)}')
    todo = [(r['box_rel'], ROOT / r['box_rel']) for r in items
            if r['box_rel'] not in done and (ROOT / r['box_rel']).exists()]
    print(f'[shard{shard_id}] 待嵌入 {len(todo)} (缺文件 {len(items) - len(done) - len(todo)})')
    if not todo and len(done) == len(items):
        paths = [r['box_rel'] for r in items]
        embs = [done[p] for p in paths]
        np.savez(out_npz, image_paths=np.array(paths), embeddings=np.array(embs, dtype=np.float32))
        print(f'[shard{shard_id}] 完成: {out_npz}')
        return
    from scripts.utils.bioclip_embed import extract_embeddings_batch
    model, processor = load_model(device)
    t0 = time.time()
    paths_done, embs_done = list(done.keys()), list(done.values())
    B = 64
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        embs = extract_embeddings_batch([str(p) for _, p in batch],
                                        model, processor, batch_size=B, device=device)
        embs = embs.numpy() if hasattr(embs, 'numpy') else np.asarray(embs)
        for (rel, _), e in zip(batch, embs):
            done[rel] = e
            paths_done.append(rel); embs_done.append(e)
        if ((i // B) + 1) % 40 == 0:
            np.savez(ckpt, image_paths=np.array(paths_done),
                     embeddings=np.array(embs_done, dtype=np.float32))
            rate = len(done) / (time.time() - t0)
            print(f'[shard{shard_id}] {len(done)}/{len(items)}, {rate:.1f} 图/s')
    paths = [r['box_rel'] for r in items if r['box_rel'] in done]
    embs = [done[p] for p in paths]
    np.savez(out_npz, image_paths=np.array(paths), embeddings=np.array(embs, dtype=np.float32))
    norms = np.linalg.norm(np.array(embs, dtype=np.float32), axis=1)
    print(f'[shard{shard_id}] 完成: {out_npz} shape=({len(paths)}, 1024) L2={norms.min():.4f}~{norms.max():.4f}')
    if ckpt.exists():
        ckpt.unlink()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    dev = args.device or (f'cuda:{args.shard}' if torch.cuda.is_available() else 'cpu')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_shard(args.shard, dev)
