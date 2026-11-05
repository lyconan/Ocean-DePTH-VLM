#!/usr/bin/env python3
"""13e_frame_split_stats.py — 帧级单框/多框统计 (用户方案: 单框帧直接采信, 多框帧待定)

口径: detect_cache 每帧检出框 (双路合并 conf>=0.05);
      有效框 = 过滤 <16px 后剩余。
单框帧 = 有效框恰 1 个; 多框帧 = 有效框 >= 2。
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
WORK = Path('/tmp/jedi_ds_boxes')
IDX = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
TINY_MIN = 16

from utils.module_loader import load_script_module  # noqa: E402
m13d = load_script_module('13d_verify_yolo_boxes')


def main():
    dets = {}
    with open(WORK / 'detect_cache.jsonl') as f:
        for line in f:
            try:
                r = json.loads(line)
                dets[r['frame_rel']] = r['dets']
            except Exception:
                pass
    idx = json.load(open(IDX))
    m2m = {m['morphospecies_id']: m for m in idx['morphospecies']}
    rel2msp = {}
    for m in idx['morphospecies']:
        for rel in m.get('images', []):
            if 'jedi_jodd_supplement' in rel or 'deepsea_expansion' in rel:
                rel2msp[rel] = m['morphospecies_id']

    raw_n = Counter()
    eff_n = Counter()
    single, multi, zero_eff = [], [], []
    l1_rej_single = 0
    multi_boxes = 0
    for fr, dl in dets.items():
        raw_n[min(len(dl), 9)] += 1
        eff = [d for d in dl if min(d['box'][2] - d['box'][0], d['box'][3] - d['box'][1]) >= TINY_MIN]
        eff_n[min(len(eff), 9)] += 1
        mid = rel2msp.get(fr)
        if not eff:
            zero_eff.append(fr)
        elif len(eff) == 1:
            m = m2m.get(mid, {})
            if not m13d.l1_pass(eff[0].get('name', 'Other'), m13d.entry_names(m)):
                l1_rej_single += 1
            else:
                single.append((fr, eff[0].get('name', ''), round(eff[0]['conf'], 3)))
        else:
            multi.append(fr)
            multi_boxes += len(eff)

    print(f'原始检出分布(帧): 0框{raw_n[0]} 1框{raw_n[1]} 2框{raw_n[2]} 3-5框{sum(v for k, v in raw_n.items() if 3 <= k <= 5)} 6+框{sum(v for k, v in raw_n.items() if k >= 6)}')
    print(f'有效框(≥16px)分布(帧): 0框{eff_n[0]} 1框{eff_n[1]} 2框{eff_n[2]} 3-5框{sum(v for k, v in eff_n.items() if 3 <= k <= 5)} 6+框{sum(v for k, v in eff_n.items() if k >= 6)}')
    print(f'\n单框帧 (采信直接入库): {len(single)} 帧')
    print(f'  其中 L1 会拒的: {l1_rej_single} (拒后该帧无框, 整帧保留兜底)')
    print(f'  采信入库: {len(single) - l1_rej_single} 框')
    print(f'多框帧 (待定): {len(multi)} 帧, 共 {multi_boxes} 框')
    print(f'无有效框帧: {len(zero_eff)} 帧 (整帧保留)')

    json.dump({'single': single, 'multi_frames': multi,
               'multi_boxes': multi_boxes, 'zero_eff': zero_eff},
              open(WORK / 'frame_split.json', 'w'), ensure_ascii=False)
    print(f'\nframe_split.json → {WORK}')


if __name__ == '__main__':
    main()
