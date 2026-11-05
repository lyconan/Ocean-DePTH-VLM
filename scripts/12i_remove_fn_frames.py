#!/usr/bin/env python3
"""12i_remove_fn_frames.py — 删除 index 中残留的 FN expansion 整帧引用 (tiny扩域后收尾)

背景: 55 张整帧/7 条目为 FN 无可用框(≥16px)时的整帧兜底; 用户决策整帧特征粒度
不对称属噪声, 全部退出 index (磁盘 jpg 保留不删, 可恢复)。
后果: FNM_00324 (11px) / FNM_00590 (14px) 两条目变 0 图 (FN 全集无可用框)。

用法: python3 scripts/12i_remove_fn_frames.py
"""
import json
import os
from pathlib import Path

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
IDX_PATH = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
PREFIX = 'data/processed/fathomnet_expansion/species/'


def main():
    bak = IDX_PATH.with_suffix('.json.bak_rmframe')
    if not bak.exists():
        os.system(f'cp "{IDX_PATH}" "{bak}"')
        print(f'[备份] {bak}')

    idx = json.load(open(IDX_PATH))
    n_removed = 0
    for m in idx['morphospecies']:
        imgs = m.get('images', [])
        kept = [r for r in imgs if not r.startswith(PREFIX)]
        if len(kept) != len(imgs):
            n_removed += len(imgs) - len(kept)
            m['images'] = kept
            m['image_count'] = len(kept)
    print(f'移除整帧引用 {n_removed}')

    tmp = IDX_PATH.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(idx, f, ensure_ascii=False)
    os.replace(tmp, IDX_PATH)

    check = json.load(open(IDX_PATH))
    left = sum(1 for m in check['morphospecies'] for r in m.get('images', []) if r.startswith(PREFIX))
    assert left == 0, f'落盘校验失败: 残留 {left}'
    zero = sum(1 for m in check['morphospecies'] if m['image_count'] == 0)
    total = sum(m['image_count'] for m in check['morphospecies'])
    print(f'[完成] 全库 {total} 图, 0图条目 {zero}: '
          + ', '.join(m['morphospecies_id'] for m in check['morphospecies'] if m['image_count'] == 0))


if __name__ == '__main__':
    main()
