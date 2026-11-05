#!/usr/bin/env python3
"""12b_update_index_for_boxes.py — morphospecies_index 挂接更新 (方案A)

变更:
  1. 每条目 images 移除 fathomnet_expansion/species/* 整帧 (由框切图替代)
  2. 加入 boxes/<msp_id>/*.jpg 框切图 (来自 boxes_manifest.json)
  3. image_count 重算, 校验 ≤50 上限
  4. 先备份 .bak (原子写)

用法: python scripts/12b_update_index_for_boxes.py
"""
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
IDX_PATH = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
MANIFEST = ROOT / 'data/processed/fathomnet_expansion/boxes_manifest.json'
LIMIT = 50


def main():
    # 备份
    bak = IDX_PATH.with_suffix('.json.bak_boxes')
    if not bak.exists():
        os.system(f'cp "{IDX_PATH}" "{bak}"')
        print(f'[备份] {bak}')

    idx = json.load(open(IDX_PATH))
    man = json.load(open(MANIFEST))
    boxes_by_msp = {}
    for r in man:
        boxes_by_msp.setdefault(r['msp_id'], []).append(r['box_rel'])
    print(f'manifest: {len(man)} 框 / {len(boxes_by_msp)} 条目')

    n_removed = n_added = n_over = n_keep_frame = 0
    for m in idx['morphospecies']:
        mid = m['morphospecies_id']
        imgs = m.get('images', [])
        added = boxes_by_msp.get(mid, [])
        if mid in boxes_by_msp:
            # 有框替代: 移除整帧, 加入框图
            kept = [rel for rel in imgs if 'fathomnet_expansion' not in rel]
            n_removed += len(imgs) - len(kept)
        else:
            # 无框替代 (tiny 全滤/无额度/交集外): 整帧保留, 避免变 0 图
            kept = list(imgs)
            if any(rel.startswith('data/processed/fathomnet_expansion/species/') for rel in kept):
                n_keep_frame += 1
        kept += added
        n_added += len(added)
        if len(kept) > LIMIT:
            n_over += 1
            print(f'  [超限] {mid}: {len(kept)} > {LIMIT}')
        m['images'] = kept
        m['image_count'] = len(kept)
    print(f'无框替代、整帧保留的条目: {n_keep_frame}')

    # 原子写回
    tmp = IDX_PATH.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(idx, f, ensure_ascii=False)
    os.replace(tmp, IDX_PATH)

    total = sum(m['image_count'] for m in idx['morphospecies'])
    zero = sum(1 for m in idx['morphospecies'] if m['image_count'] == 0)
    print(f'[完成] 移除整帧 {n_removed}, 加入框图 {n_added}, 超限条目 {n_over}')
    print(f'       全库 {len(idx["morphospecies"])} 条目 / {total} 图, 0图条目 {zero}')
    # 落盘断言
    check = json.load(open(IDX_PATH))
    assert sum(len(m['images']) for m in check['morphospecies']) == total, '落盘校验失败'
    print('[断言] 落盘校验通过')


if __name__ == '__main__':
    main()
