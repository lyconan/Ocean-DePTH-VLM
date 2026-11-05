#!/usr/bin/env python3
"""12a_crop_fathomnet_boxes.py — FathomNet 整帧 → 框切图（方案A）

数据链路:
  COCO 标注 (opc-03/Sea_Data/FathomNet/Annotations) 的 category name = FN concept 名
  → concept 与 FNM 条目的 fathomnet_concept 同链路
  → 框归属:
      latin 唯一命中条目  → 直接挂
      歧义类名            → concept → 唯一 FNM 条目回挂
      其余 (无 FNM 对应)  → 丢弃
  → 每条目图量上限 50 (图册图保留 + 框切图合计), 框按帧 round-robin 保证多样性
  → 最小框 32px 过滤, 裁剪加 10% margin

用法:
  python scripts/12a_crop_fathomnet_boxes.py --limit 5      # 小批量验证
  python scripts/12a_crop_fathomnet_boxes.py                # 全量裁剪到本地中转
  python scripts/12a_crop_fathomnet_boxes.py --sync         # 中转 → NAS
  python scripts/12a_crop_fathomnet_boxes.py --stats        # 查看统计
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
FN = Path('/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet')
IDX_PATH = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
WORK = Path('/tmp/fn_boxes')                      # 容器本地中转 (NFS 写慢)
NAS_BOX_DIR = ROOT / 'data/processed/fathomnet_expansion/boxes'
NAS_MANIFEST = ROOT / 'data/processed/fathomnet_expansion/boxes_manifest.json'
MIN_BOX = 32                                      # 框最小边长 (px)
LIMIT_PER_MSP = 50                                # 条目图片总数上限 (所有来源合计)
MARGIN = 0.10                                     # 裁剪外扩比例


def atomic_json(path: Path, obj):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_index():
    print('[1] 加载 morphospecies_index ...')
    idx = json.load(open(IDX_PATH))
    print(f'    条目 {len(idx["morphospecies"])}')
    return idx


def build_lookups(idx):
    by_latin = defaultdict(list)
    for m in idx['morphospecies']:
        by_latin[m['latin']].append(m)
    concept2fnm = {}
    for m in idx['morphospecies']:
        if m['morphospecies_id'].startswith('FNM'):
            for e in m.get('entries', []):
                c = e.get('fathomnet_concept') if isinstance(e, dict) else None
                if c:
                    concept2fnm[c] = m['morphospecies_id']
    return by_latin, concept2fnm


def stem_to_frame(idx):
    """stem → 原整帧路径 (同 stem 多 concept 副本取全部候选)。"""
    cands = defaultdict(list)
    for m in idx['morphospecies']:
        for rel in m.get('images', []):
            if 'fathomnet_expansion' in rel:
                p = ROOT / rel
                if p.exists():
                    cands[p.stem].append(p)
    return cands


def nonfn_image_count(m):
    """条目非 FN-整帧 的图量 (图册/GBIF/JEDI 等, 保留并计入 50 上限)。"""
    return sum(1 for rel in m.get('images', []) if 'fathomnet_expansion' not in rel)


def build_plan(idx, limit=0):
    print('[2] 构建 plans ...')
    by_latin, concept2fnm = build_lookups(idx)

    # 交集 stems
    exp_stems = set()
    for m in idx['morphospecies']:
        for rel in m.get('images', []):
            if 'fathomnet_expansion' in rel:
                exp_stems.add(Path(rel).stem)

    # COCO 框 → 归属
    msp_boxes = defaultdict(lambda: defaultdict(list))   # msp -> stem -> [(split, bbox, concept)]
    from collections import Counter
    drop = Counter()
    for split in ['train', 'val']:
        coco = json.load(open(FN / f'Annotations/fathomnet_{split}.json'))
        cats = {c['id']: c['name'] for c in coco['categories']}
        i2s = {im['id']: Path(im['file_name']).stem for im in coco['images']}
        for a in coco['annotations']:
            stem = i2s.get(a['image_id'])
            if stem is None or stem not in exp_stems:
                continue
            concept = cats[a['category_id']]
            owners = by_latin.get(concept, [])
            if len(owners) == 1:
                msp = owners[0]['morphospecies_id']
            elif len(owners) > 1 and concept in concept2fnm:
                msp = concept2fnm[concept]
            else:
                drop[concept] += 1
                continue
            msp_boxes[msp][stem].append((split, a['bbox'], concept))
    print(f'    归属完成: {sum(len(v) for d in msp_boxes.values() for v in d.values())} 框 -> '
          f'{len(msp_boxes)} 条目; 丢弃 {sum(drop.values())} 框 ({len(drop)} 类名)')

    # 50 上限 + round-robin by stem (帧多样性)
    msp2m = {m['morphospecies_id']: m for m in idx['morphospecies']}
    plans = {}
    tiny = total_kept = 0
    for msp, frames in msp_boxes.items():
        m = msp2m[msp]
        budget = LIMIT_PER_MSP - nonfn_image_count(m)
        if budget <= 0:
            continue
        # 过滤过小框
        kept_frames = {}
        for stem, boxes in frames.items():
            good = [b for b in boxes if min(b[1][2], b[1][3]) >= MIN_BOX]
            if good:
                kept_frames[stem] = good
        kept_n = sum(len(v) for v in kept_frames.values())
        # round-robin: 第 k 轮取每帧第 k 框
        take = []
        round_i = 0
        while len(take) < min(budget, kept_n):
            for stem, boxes in kept_frames.items():
                if round_i < len(boxes) and len(take) < budget:
                    take.append((stem, boxes[round_i]))
            round_i += 1
        if take:
            plans[msp] = [{'stem': s, 'split': b[0], 'bbox': b[1], 'concept': b[2]}
                          for s, b in take]
            total_kept += len(take)
    print(f'    plans: {len(plans)} 条目, {total_kept} 框 (tiny<{MIN_BOX}px 过滤 {tiny} 框)')
    return plans, drop


def stem_frame_map(idx):
    """stem → 实际存在的一个整帧路径。"""
    print('[3] 索引整帧路径 ...')
    cands = defaultdict(list)
    for m in idx['morphospecies']:
        for rel in m.get('images', []):
            if 'fathomnet_expansion' in rel:
                cands[Path(rel).stem].append(ROOT / rel)
    mapping = {}
    for stem, ps in cands.items():
        for p in ps:                     # 多副本取第一个存在者
            if p.exists():
                mapping[stem] = p
                break
    print(f'    {len(mapping)} 帧可用')
    return mapping


def run_crop(plans, mapping, limit=0):
    print('[4] 裁剪 ...')
    manifest = []
    if (WORK / 'boxes_manifest.json').exists():
        manifest = json.load(open(WORK / 'boxes_manifest.json'))
        print(f'    断点续跑: 已有 {len(manifest)} 框 manifest')
    done = {(r['msp_id'], Path(r['box_rel']).name) for r in manifest}

    # stem → [(msp, box_idx, b)] 索引 (避免 O(帧×框) 全扫)
    stem_index = defaultdict(list)
    for msp, boxes in plans.items():
        for j, b in enumerate(boxes):
            stem_index[b['stem']].append((msp, j, b))
    stems = sorted(stem_index.keys())
    if limit:
        stems = stems[:limit]
    print(f'    待裁剪 {len(stems)} 帧')
    t0 = time.time()
    n_new = 0
    for i, stem in enumerate(stems):
        fp = mapping.get(stem)
        if fp is None:
            continue
        try:
            with Image.open(fp) as im:
                W, H = im.size
                rgb = im.convert('RGB')
        except Exception as e:
            print(f'    [跳过] {stem}: {e}')
            continue
        for msp, j, b in stem_index[stem]:
            name = f"{b['split']}_{stem}_b{j}.jpg"
            key = (msp, name)
            if key in done:
                continue
            x, y, w, h = b['bbox']
            mx, my = int(w * MARGIN), int(h * MARGIN)
            x0 = max(0, int(x) - mx); y0 = max(0, int(y) - my)
            x1 = min(W, int(x + w) + mx); y1 = min(H, int(y + h) + my)
            if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
                continue
            crop = rgb.crop((x0, y0, x1, y1))
            outdir = WORK / 'boxes' / msp
            outdir.mkdir(parents=True, exist_ok=True)
            crop.save(outdir / name, 'JPEG', quality=95)
            manifest.append({'box_rel': f'data/processed/fathomnet_expansion/boxes/{msp}/{name}',
                             'msp_id': msp, 'concept': b['concept'], 'source_stem': stem,
                             'bbox_xywh': [round(x), round(y), round(w), round(h)],
                             'frame_rel': str(fp.relative_to(ROOT))})
            n_new += 1
            if n_new % 2000 == 0:
                atomic_json(WORK / 'boxes_manifest.json', manifest)
                print(f'    {n_new} 框 ... {time.time()-t0:.0f}s')
        rgb.close()
        if (i + 1) % 500 == 0:
            atomic_json(WORK / 'boxes_manifest.json', manifest)
            rate = (i + 1) / (time.time() - t0)
            eta = (len(stems) - i - 1) / max(rate, 0.1) / 60
            print(f'    帧 {i+1}/{len(stems)}, 框 {n_new}, {rate:.1f} 帧/s, ETA {eta:.0f}min')
    atomic_json(WORK / 'boxes_manifest.json', manifest)
    print(f'[4] 完成: 新增 {n_new} 框, manifest 共 {len(manifest)}')


def run_sync():
    print('[5] 中转 → NAS ...')
    if not (WORK / 'boxes_manifest.json').exists():
        print('    manifest 不存在, 先跑全量裁剪')
        return
    total = copied = skip = 0
    for r in json.load(open(WORK / 'boxes_manifest.json')):
        total += 1
        src = WORK / r['box_rel'].replace('data/processed/fathomnet_expansion/', '')
        dst = ROOT / r['box_rel']
        if dst.exists():
            skip += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    shutil.copy2(WORK / 'boxes_manifest.json', NAS_MANIFEST)
    print(f'    {total} 框: 新复制 {copied}, 已存在 {skip}; manifest → {NAS_MANIFEST}')


def run_stats():
    if not (WORK / 'boxes_manifest.json').exists():
        print('无 manifest')
        return
    from collections import Counter
    man = json.load(open(WORK / 'boxes_manifest.json'))
    per_msp = Counter(r['msp_id'] for r in man)
    per_concept = Counter(r['concept'] for r in man)
    print(f'框切图 manifest: {len(man)} 框 / {len(per_msp)} 条目')
    print(f'  条目框数: top5 {per_msp.most_common(5)}')
    print(f'  concept: top5 {per_concept.most_common(5)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='仅处理前 N 帧 (验证)')
    ap.add_argument('--sync', action='store_true')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--plan', action='store_true', help='仅构建并保存 plan')
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    if args.stats:
        run_stats(); return
    if args.sync:
        run_sync(); return

    idx = load_index()
    if (WORK / 'plans.json').exists() and not args.plan:
        plans = json.load(open(WORK / 'plans.json'))
        print(f'[2] plans 缓存: {len(plans)} 条目')
    else:
        plans, drop = build_plan(idx)
        atomic_json(WORK / 'plans.json', plans)
    if args.plan:
        return

    mapping = stem_frame_map(idx)
    run_crop(plans, mapping, limit=args.limit)
    run_stats()


if __name__ == '__main__':
    main()
