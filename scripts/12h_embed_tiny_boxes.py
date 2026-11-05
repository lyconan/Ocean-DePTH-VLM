#!/usr/bin/env python3
"""12h_embed_tiny_boxes.py — 12g tiny 扩域图 (16-31px 框扩 margin 至 64px 区域) BioCLIP 2.5 嵌入

输入: data/processed/fathomnet_expansion/boxes_manifest_tiny.json
输出: data/processed/bioclip25_embeddings/emb_boxes_tiny.npz

用法: 需 PYTHONPATH=项目根; python3 scripts/12h_embed_tiny_boxes.py
"""
import json
from pathlib import Path

import numpy as np

from scripts.utils.bioclip_embed import get_bioclip_model, extract_embeddings_batch

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
MAN = ROOT / 'data/processed/fathomnet_expansion/boxes_manifest_tiny.json'
OUT = ROOT / 'data/processed/bioclip25_embeddings/emb_boxes_tiny.npz'
DEVICE = 'cuda:0'


def main():
    man = json.load(open(MAN))
    paths = [str(ROOT / r['box_rel']) for r in man]
    missing = [p for p in paths if not Path(p).exists()]
    assert not missing, f'缺图 {len(missing)}: {missing[:3]}'
    print(f'tiny manifest: {len(man)} 图')

    model, processor = get_bioclip_model(device=DEVICE)
    embs = extract_embeddings_batch(paths, model, processor, batch_size=32, device=DEVICE)
    embs = embs.numpy() if hasattr(embs, 'numpy') else np.asarray(embs)
    embs = embs.astype(np.float32)
    norms = np.linalg.norm(embs, axis=1)
    print(f'嵌入: {embs.shape}, L2 {norms.min():.4f}~{norms.max():.4f}')

    np.savez(OUT, image_paths=np.array([r['box_rel'] for r in man]), embeddings=embs)
    chk = np.load(OUT, allow_pickle=True)
    assert chk['embeddings'].shape == (len(man), 1024), '落盘校验失败'
    print(f'[完成] {OUT} 共 {len(man)} 嵌入, 断言通过')


if __name__ == '__main__':
    main()
