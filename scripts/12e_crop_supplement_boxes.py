#!/usr/bin/env python3
"""12e_crop_supplement_boxes.py — 方案补-2: 用 FN 全集标注为框稀少条目补框

与 12a 的差异:
  1. 归属顺序: concept→FNM 权威映射优先 → latin 唯一命中兜底
  2. 帧域: FN 本地全集标注帧, 不限 expansion 交集
  3. 去重: 框级 (stem + bbox 8px 粒度) 对已切 19,024 框去重
  4. 目标: 当前框数 <10 的条目, budget = 50 - 非FN图 - 已有框图
  5. b 索引接续: 同条目同帧已有切图时从最大索引 +1 起
  6. index 更新: 获得框图的条目若仍有 FN 整帧 → 移除整帧, 备份 .bak_sup
"""
import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
FN = Path('/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet')
IDX_PATH = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
WORK = Path('/tmp/fn_boxes')
NAS_MANIFEST = ROOT / 'data/processed/fathomnet_expansion/boxes_manifest.json'
SUP_MANIFEST_W = WORK / 'boxes_manifest_sup.json'
SUP_MANIFEST_N = ROOT / 'data/processed/fathomnet_expansion/boxes_manifest_sup.json'
MIN_BOX = 32
LIMIT = 50
MARGIN = 0.10


def atomic_json(path: Path, obj):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_all():
    print('[1] 加载 index + 原框 manifest ...')
    idx = json.load(open(IDX_PATH))
    man = json.load(open(NAS_MANIFEST)) if NAS_MANIFEST.exists() else []
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
    return idx, man, by_latin, concept2fnm


def current_stats(idx):
    fnbox, nonfn = defaultdict(int), defaultdict(int)
    for m in idx['morphospecies']:
        mid = m['morphospecies_id']
        for rel in m.get('images', []):
            if '/boxes/' in rel:
                fnbox[mid] += 1
            elif 'fathomnet_expansion' not in rel:
                nonfn[mid] += 1
    return fnbox, nonfn


def build_sup_plan(idx, man, by_latin, concept2fnm):
    print('[2] 构建补充 plans ...')
    fnbox, nonfn = current_stats(idx)
    cut = set()
    for r in man:
        b = r['bbox_xywh']
        cut.add((r['source_stem'], b[0] // 8, b[1] // 8, b[2] // 8, b[3] // 8))
    max_b = defaultdict(int)
    for r in man:
        k = Path(r['box_rel']).name.rsplit('_b', 1)
        if len(k) == 2 and k[1].endswith('.jpg'):
            key = (r['msp_id'], r['source_stem'])
            max_b[key] = max(max_b[key], int(k[1][:-4]) + 1)

    def owner(cname):
        if cname in concept2fnm:
            return concept2fnm[cname]
        owners = by_latin.get(cname, [])
        if len(owners) == 1:
            return owners[0]['morphospecies_id']
        return None

    msp_boxes = defaultdict(lambda: defaultdict(list))
    drop = 0
    for split in ['train', 'val']:
        coco = json.load(open(FN / f'Annotations/fathomnet_{split}.json'))
        cats = {c['id']: c['name'] for c in coco['categories']}
        i2s = {im['id']: Path(im['file_name']).stem for im in coco['images']}
        for a in coco['annotations']:
            stem = i2s.get(a['image_id'])
            if stem is None:
                continue
            b = a['bbox']
            if min(b[2], b[3]) < MIN_BOX:
                continue
            if (stem, b[0] // 8, b[1] // 8, b[2] // 8, b[3] // 8) in cut:
                continue
            msp = owner(cats[a['category_id']])
            if msp is None:
                drop += 1
                continue
            msp_boxes[msp][stem].append((split, b, cats[a['category_id']]))

    plans = {}
    total = 0
    for msp, frames in msp_boxes.items():
        if fnbox.get(msp, 0) >= 10:
            continue
        budget = LIMIT - nonfn.get(msp, 0) - fnbox.get(msp, 0)
        if budget <= 0:
            continue
        take = []
        round_i = 0
        while len(take) < budget:
            progressed = False
            for stem, boxes in frames.items():
                if round_i < len(boxes) and len(take) < budget:
                    take.append((stem, boxes[round_i]))
                    progressed = True
            if not progressed:
                break
            round_i += 1
        if take:
            plans[msp] = [{'stem': s, 'split': b[0], 'bbox': list(map(float, b[1])),
                           'concept': b[2], 'b_start': max_b[(msp, s)]}
                          for s, b in take]
            total += len(take)
    print(f'    plans: {len(plans)} 条目, {total} 框; 丢弃(不可归属) {drop} 框')
    return plans


def run_crop(plans, limit=0):
    print('[3] 裁剪 ...')
    manifest = []
    if SUP_MANIFEST_W.exists():
        manifest = json.load(open(SUP_MANIFEST_W))
        print(f'    断点续跑: 已有 {len(manifest)} 框')
    done = {r['box_rel'] for r in manifest}
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
        infos = stem_index[stem]
        split0 = infos[0][2]['split']
        fp = FN / ('Train' if split0 == 'train' else 'Val') / f'{stem}.png'
        if not fp.exists():
            continue
        try:
            with Image.open(fp) as im:
                W, H = im.size
                rgb = im.convert('RGB')
        except Exception as e:
            print(f'    [跳过] {stem}: {e}')
            continue
        for msp, j, b in infos:
            name = f"{b['split']}_{stem}_b{b['b_start'] + j}.jpg"
            box_rel = f'data/processed/fathomnet_expansion/boxes/{msp}/{name}'
            if box_rel in done:
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
            manifest.append({'box_rel': box_rel, 'msp_id': msp, 'concept': b['concept'],
                             'source_stem': stem,
                             'bbox_xywh': [round(x), round(y), round(w), round(h)],
                             'frame_rel': str(fp)})
            n_new += 1
            if n_new % 1000 == 0:
                atomic_json(SUP_MANIFEST_W, manifest)
                print(f'    {n_new} 框 ... {time.time()-t0:.0f}s')
        rgb.close()
        if (i + 1) % 500 == 0:
            atomic_json(SUP_MANIFEST_W, manifest)
            rate = (i + 1) / (time.time() - t0)
            print(f'    帧 {i+1}/{len(stems)}, 框 {n_new}, {rate:.1f} 帧/s')
    atomic_json(SUP_MANIFEST_W, manifest)
    print(f'[3] 完成: 新增 {n_new} 框, sup manifest 共 {len(manifest)}')


def run_sync():
    print('[4] 中转 → NAS ...')
    if not SUP_MANIFEST_W.exists():
        print('    sup manifest 不存在')
        return
    total = copied = skip = 0
    for r in json.load(open(SUP_MANIFEST_W)):
        total += 1
        src = WORK / r['box_rel'].replace('data/processed/fathomnet_expansion/', '')
        dst = ROOT / r['box_rel']
        if dst.exists():
            skip += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    shutil.copy2(SUP_MANIFEST_W, SUP_MANIFEST_N)
    print(f'    {total} 框: 新复制 {copied}, 已存在 {skip}; sup manifest → NAS')


def update_index():
    print('[5] index 更新: 框 rel 加入 / 获框条目整帧退出 ...')
    if not SUP_MANIFEST_W.exists():
        print('    sup manifest 不存在')
        return
    sup = json.load(open(SUP_MANIFEST_W))
    bak = IDX_PATH.with_suffix('.json.bak_sup')
    shutil.copy2(IDX_PATH, bak)
    idx = json.load(open(IDX_PATH))
    msp2m = {m['morphospecies_id']: m for m in idx['morphospecies']}
    new_by_msp = defaultdict(list)
    for r in sup:
        new_by_msp[r['msp_id']].append(r['box_rel'])
    n_add = n_frame_rm = 0
    for mid, rels in new_by_msp.items():
        m = msp2m[mid]
        imgs = m.get('images', [])
        rel_set = set(rels)
        imgs = [rel for rel in imgs if rel not in rel_set]
        if any(rel.startswith('data/processed/fathomnet_expansion/species/') for rel in imgs):
            n_frame_rm += sum(1 for rel in imgs
                              if rel.startswith('data/processed/fathomnet_expansion/species/'))
            imgs = [rel for rel in imgs
                    if not rel.startswith('data/processed/fathomnet_expansion/species/')]
        imgs += rels
        n_add += len(rels)
        m['images'] = imgs
        m['image_count'] = len(imgs)
        if len(imgs) > LIMIT:
            print(f'  [超限] {mid}: {len(imgs)}')
    n_zero = sum(1 for m in idx['morphospecies'] if m['image_count'] == 0)
    atomic_json(IDX_PATH, idx)
    with open(IDX_PATH) as f:
        chk = json.load(f)
    assert len(chk['morphospecies']) == len(idx['morphospecies']), '落盘失败'
    print(f'    新增框 rel {n_add}, 移除整帧 {n_frame_rm}, 0图条目 {n_zero} (应0)')
    print(f'    备份 .bak_sup; 落盘断言通过')


def run_stats():
    if not SUP_MANIFEST_W.exists():
        print('无 sup manifest')
        return
    from collections import Counter
    man = json.load(open(SUP_MANIFEST_W))
    per = Counter(r['msp_id'] for r in man)
    print(f'sup manifest: {len(man)} 框 / {len(per)} 条目; top5 {per.most_common(5)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sync', action='store_true')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--update-index', action='store_true')
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    if args.stats:
        run_stats(); return
    if args.sync:
        run_sync(); return
    if args.update_index:
        update_index(); return

    idx, man, by_latin, concept2fnm = load_all()
    if (WORK / 'sup_plans.json').exists() and not args.plan:
        plans = json.load(open(WORK / 'sup_plans.json'))
        print(f'[2] sup plans 缓存: {len(plans)} 条目')
    else:
        plans = build_sup_plan(idx, man, by_latin, concept2fnm)
        atomic_json(WORK / 'sup_plans.json', plans)
    if args.plan:
        return
    run_crop(plans, limit=args.limit)
    run_stats()


if __name__ == '__main__':
    main()
