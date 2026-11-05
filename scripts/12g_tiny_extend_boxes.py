#!/usr/bin/env python3
"""12g_tiny_extend_boxes.py — 小框扩域切图 (16-31px 框, 自适应 margin 扩到 64px 区域)

原理: 26px 框按 10% margin 切出 ~31px 再 resize 224 → 7.2x 放大全糊;
     扩域到 64px 区域再 resize → 3.5x, 目标占边长 ~40%, 可用性显著提升。
目标: 0 框 FNM 条目在全集标注中 16<=min(w,h)<32 的框 (从未切过)。
"""
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
FN = Path('/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet')
IDX_PATH = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
WORK = Path('/tmp/fn_boxes')
NAS_MANIFEST = ROOT / 'data/processed/fathomnet_expansion/boxes_manifest.json'
TINY_MANIFEST_W = WORK / 'boxes_manifest_tiny.json'
TINY_MANIFEST_N = ROOT / 'data/processed/fathomnet_expansion/boxes_manifest_tiny.json'
TINY_MIN, TINY_MAX, EXTEND = 16, 32, 64


def atomic_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    idx = json.load(open(IDX_PATH))
    msps = idx['morphospecies']
    concept2fnm = {}
    for m in msps:
        if m['morphospecies_id'].startswith('FNM'):
            for e in m.get('entries', []):
                c = e.get('fathomnet_concept') if isinstance(e, dict) else None
                if c:
                    concept2fnm[c] = m['morphospecies_id']
    fnbox = defaultdict(int)
    nonfn = defaultdict(int)
    for m in msps:
        for r in m.get('images', []):
            if '/boxes/' in r:
                fnbox[m['morphospecies_id']] += 1
            elif 'fathomnet_expansion' not in r:
                nonfn[m['morphospecies_id']] += 1
    targets = {}
    for m in msps:
        mid = m['morphospecies_id']
        if mid.startswith('FNM') and fnbox[mid] == 0:
            for e in m.get('entries', []):
                if isinstance(e, dict) and e.get('fathomnet_concept'):
                    targets[e['fathomnet_concept']] = mid
    print(f'目标条目: {len(targets)}')

    cut = set()
    man = json.load(open(NAS_MANIFEST))
    for r in man:
        b = r['bbox_xywh']
        cut.add((r['source_stem'], b[0] // 8, b[1] // 8, b[2] // 8, b[3] // 8))
    max_b = defaultdict(int)
    for r in man:
        k = Path(r['box_rel']).name.rsplit('_b', 1)
        if len(k) == 2 and k[1].endswith('.jpg'):
            max_b[(r['msp_id'], r['source_stem'])] = max(max_b[(r['msp_id'], r['source_stem'])],
                                                         int(k[1][:-4]) + 1)

    # 收集 tiny 框 (帧级 round-robin, budget=50-nonfn)
    per_msp = defaultdict(lambda: defaultdict(list))  # msp -> stem -> [(split, bbox)]
    for split in ['train', 'val']:
        coco = json.load(open(FN / f'Annotations/fathomnet_{split}.json'))
        cats = {c['id']: c['name'] for c in coco['categories']}
        i2s = {im['id']: Path(im['file_name']).stem for im in coco['images']}
        for a in coco['annotations']:
            cname = cats[a['category_id']]
            if cname not in targets:
                continue
            stem = i2s.get(a['image_id'])
            b = a['bbox']
            if stem is None or not (TINY_MIN <= min(b[2], b[3]) < TINY_MAX):
                continue
            if (stem, b[0] // 8, b[1] // 8, b[2] // 8, b[3] // 8) in cut:
                continue
            per_msp[targets[cname]][stem].append((split, b))
    manifest = []
    for mid, frames in per_msp.items():
        budget = 50 - nonfn.get(mid, 0)
        take, round_i = [], 0
        while len(take) < budget:
            prog = False
            for stem, boxes in frames.items():
                if round_i < len(boxes) and len(take) < budget:
                    take.append((stem, boxes[round_i]))
                    prog = True
            if not prog:
                break
            round_i += 1
        for stem, (split, b) in take:
            fp = FN / ('Train' if split == 'train' else 'Val') / f'{stem}.png'
            if not fp.exists():
                continue
            with Image.open(fp) as im:
                W, H = im.size
                rgb = im.convert('RGB')
            x, y, w, h = b
            mx = max(int(w * 0.10), int((EXTEND - w) / 2))
            my = max(int(h * 0.10), int((EXTEND - h) / 2))
            x0, y0 = max(0, int(x) - mx), max(0, int(y) - my)
            x1, y1 = min(W, int(x + w) + mx), min(H, int(y + h) + my)
            if x1 - x0 < TINY_MIN or y1 - y0 < TINY_MIN:
                rgb.close()
                continue
            name = f'{split}_{stem}_b{max_b[(mid, stem)] + sum(1 for mm in manifest if mm["source_stem"] == stem and mm["msp_id"] == mid)}.jpg'
            box_rel = f'data/processed/fathomnet_expansion/boxes/{mid}/{name}'
            outdir = WORK / 'boxes' / mid
            outdir.mkdir(parents=True, exist_ok=True)
            rgb.crop((x0, y0, x1, y1)).save(outdir / name, 'JPEG', quality=95)
            manifest.append({'box_rel': box_rel, 'msp_id': mid, 'concept': None,
                             'source_stem': stem, 'bbox_xywh': [round(x), round(y), round(w), round(h)],
                             'frame_rel': str(fp), 'tiny_extend': True})
            rgb.close()
    atomic_json(TINY_MANIFEST_W, manifest)
    print(f'切图完成: {len(manifest)} 框 / {len({r["msp_id"] for r in manifest})} 条目')

    # sync → NAS
    copied = 0
    for r in manifest:
        src = WORK / r['box_rel'].replace('data/processed/fathomnet_expansion/', '')
        dst = ROOT / r['box_rel']
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    shutil.copy2(TINY_MANIFEST_W, TINY_MANIFEST_N)
    print(f'sync: {copied} 新复制; manifest → NAS')

    # index 更新: 加框图, 获框条目整帧退出
    bak = IDX_PATH.with_suffix('.json.bak_tiny')
    shutil.copy2(IDX_PATH, bak)
    msp2m = {m['morphospecies_id']: m for m in msps}
    n_add = n_rm = 0
    for mid in {r['msp_id'] for r in manifest}:
        m = msp2m[mid]
        rels = [r['box_rel'] for r in manifest if r['msp_id'] == mid]
        imgs = [r for r in m.get('images', []) if r not in set(rels)]
        if any(r.startswith('data/processed/fathomnet_expansion/species/') for r in imgs):
            n_rm += sum(1 for r in imgs if r.startswith('data/processed/fathomnet_expansion/species/'))
            imgs = [r for r in imgs if not r.startswith('data/processed/fathomnet_expansion/species/')]
        imgs += rels
        n_add += len(rels)
        m['images'] = imgs
        m['image_count'] = len(imgs)
    n_zero = sum(1 for m in idx['morphospecies'] if m['image_count'] == 0)
    atomic_json(IDX_PATH, idx)
    with open(IDX_PATH) as f:
        assert len(json.load(f)['morphospecies']) == len(msps)
    print(f'index: +{n_add} 框 rel, -{n_rm} 整帧, 0图条目 {n_zero}; 备份 .bak_tiny')


if __name__ == '__main__':
    main()
