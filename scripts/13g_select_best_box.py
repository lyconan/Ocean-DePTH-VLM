#!/usr/bin/env python3
"""13g_select_best_box.py — 多框帧逐帧选优 (用户定稿: 每帧取与条目谱系最像的 1 框)

背景: 13d 的 L2 是逐框排噪 (s_pos > s_neg), 同帧多个框可能全部保留。
用户语义: 整帧只有一个物种标签, 多框帧应从 top-3 中选与条目谱系名
          最接近 (s_pos 最大) 的那 1 框, 其余丢弃 (同帧冗余/异类排除)。
trusted 单框帧不受影响, 全部保留。

输入: WORK/boxes_manifest.json (13d keep 子集, 行含 s_pos/conf/trusted)
输出: 新 manifest (备份旧为 .bak_l2all) + 选择统计

用法 (容器内): PYTHONPATH=项目根 python3 scripts/13g_select_best_box.py [--revert]
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
WORK = Path('/tmp/jedi_ds_boxes')
MAN = WORK / 'boxes_manifest.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--revert', action='store_true', help='回滚到 .bak_l2all')
    args = ap.parse_args()

    bak = MAN.with_suffix('.json.bak_l2all')
    if args.revert:
        if not bak.exists():
            print('无备份可回滚')
            return
        man = json.load(open(bak))
        tmp = MAN.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(man, f, ensure_ascii=False)
        os.replace(tmp, MAN)
        print(f'[回滚] {bak.name} → manifest ({len(man)} 框)')
        return

    man = json.load(open(MAN))
    if not bak.exists():
        import shutil
        shutil.copy2(MAN, bak)
        print(f'[备份] {bak.name} ({len(man)} 框)')

    trusted = [r for r in man if r.get('trusted')]
    multi = [r for r in man if not r.get('trusted')]
    by_frame = defaultdict(list)
    for r in multi:
        by_frame[r['frame_rel']].append(r)
    # 每帧 s_pos 最大者 (平局取 conf 高); s_pos 缺失的行 (不应有) 用 -1
    kept, dropped = [], []
    for fr, rows in by_frame.items():
        best = max(rows, key=lambda r: (r.get('s_pos', -1), r.get('conf', 0)))
        kept.append(best)
        dropped.extend(r for r in rows if r is not best)
    new_man = trusted + kept
    print(f'[选择] 多框帧 {len(by_frame)} 帧: 保留 {len(kept)}, 丢弃 {len(dropped)}')
    print(f'[选择] manifest {len(man)} → {len(new_man)} '
          f'(trusted {len(trusted)} + 多框帧选优 {len(kept)})')

    # 条目覆盖影响: 丢弃后变 0 框的条目 (13b 重跑时将保留整帧兜底)
    old_mids = {r['msp_id'] for r in man}
    new_mids = {r['msp_id'] for r in new_man}
    lost = sorted(old_mids - new_mids)
    print(f'[选择] 条目覆盖 {len(old_mids)} → {len(new_mids)}, 丢失框的条目 {len(lost)}'
          + (f' (示例 {lost[:5]})' if lost else ''))
    by_src = defaultdict(int)
    for r in new_man:
        by_src[r['source']] += 1
    per_msp = defaultdict(int)
    for r in new_man:
        per_msp[r['msp_id']] += 1
    print(f'[选择] 来源 {dict(by_src)}; 条目框数 top5: '
          f'{sorted(per_msp.items(), key=lambda kv: -kv[1])[:5]}')

    tmp = MAN.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(new_man, f, ensure_ascii=False)
    os.replace(tmp, MAN)
    chk = json.load(open(MAN))
    assert len(chk) == len(new_man), '落盘断言失败'
    print(f'[完成] manifest {len(chk)} 框, 断言通过')


if __name__ == '__main__':
    main()
