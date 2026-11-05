#!/usr/bin/env python3
"""13b_update_index_for_yolo_boxes.py — index 挂接更新 (JEDI/深海框切图, 二期)

变更:
  1. 有 YOLO 框替代的条目: 移除 jedi/deepsea 整帧, 加入 yolo_boxes 框图
  2. 无框条目 (推理无检出): 整帧保留, 避免变 0 图
  3. image_count 重算, 校验 ≤50; 备份 .bak_yolo
"""
import json
import os
from pathlib import Path

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
IDX_PATH = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
MANIFEST = ROOT / 'data/processed/yolo_boxes_manifest.json'
LIMIT = 50


def is_frame(rel):
    return 'jedi_jodd_supplement' in rel or 'deepsea_expansion' in rel


def main():
    bak = IDX_PATH.with_suffix('.json.bak_yolo')
    if not bak.exists():
        os.system(f'cp "{IDX_PATH}" "{bak}"')
        print(f'[备份] {bak}')
    idx = json.load(open(IDX_PATH))
    man = json.load(open(MANIFEST))
    boxes_by_msp = {}
    for r in man:
        boxes_by_msp.setdefault(r['msp_id'], []).append(r['box_rel'])
    print(f'manifest: {len(man)} 框 / {len(boxes_by_msp)} 条目')

    n_removed = n_added = n_over = n_keep = 0
    for m in idx['morphospecies']:
        mid = m['morphospecies_id']
        imgs = m.get('images', [])
        added = boxes_by_msp.get(mid, [])
        if added:
            kept = [rel for rel in imgs if not is_frame(rel)]
            n_removed += len(imgs) - len(kept)
        else:
            kept = list(imgs)
            if any(is_frame(rel) for rel in kept):
                n_keep += 1
        kept += added
        n_added += len(added)
        if len(kept) > LIMIT:
            n_over += 1
            print(f'  [超限] {mid}: {len(kept)}')
        m['images'] = kept
        m['image_count'] = len(kept)
    print(f'整帧退出 {n_removed}, 加入框图 {n_added}, 无检出保留整帧条目 {n_keep}')

    tmp = IDX_PATH.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(idx, f, ensure_ascii=False)
    os.replace(tmp, IDX_PATH)
    check = json.load(open(IDX_PATH))
    assert sum(len(m['images']) for m in check['morphospecies']) == sum(m['image_count'] for m in check['morphospecies'])
    total = sum(m['image_count'] for m in check['morphospecies'])
    zero = sum(1 for m in check['morphospecies'] if m['image_count'] == 0)
    print(f'[完成] 全库 {total} 图, 0图条目 {zero}, 断言通过')


if __name__ == '__main__':
    main()
