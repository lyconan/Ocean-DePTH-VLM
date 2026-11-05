#!/usr/bin/env python3
"""13a_crop_jedi_deepsea_boxes.py — JEDI/深海整帧 → YOLO 双路框切图 (二期)

数据链路:
  index 中 jedi_jodd_supplement + deepsea_expansion 整帧 (6,160 张 / 974 条目)
  → DualPathDetector (960+TTA ∪ 1920, conf=0.05) 检测
  → 框归属: 图挂在哪个 msp, 框挂回哪个 msp (一图一挂, 归属唯一)
  → ≥32px 直接切 (10% margin); 16-31px 自适应扩域至 64px 区域; <16px 丢弃
  → 50 上限: budget = 50 - 条目保留图量; 帧间 round-robin, 帧内 conf 降序

用法 (容器内, PYTHONPATH=项目根):
  python scripts/13a_crop_jedi_deepsea_boxes.py --detect --limit 20   # 小批量验证
  python scripts/13a_crop_jedi_deepsea_boxes.py --detect              # 全量推理 (断点续跑)
  python scripts/13a_crop_jedi_deepsea_boxes.py --crop                # 切图 → /tmp 中转
  python scripts/13a_crop_jedi_deepsea_boxes.py --sync                # 中转 → NAS
  python scripts/13a_crop_jedi_deepsea_boxes.py --stats
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
IDX_PATH = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
WEIGHTS = ROOT / 'runs/detect/yolo11m_class34-2/weights/best.pt'
WORK = Path('/tmp/jedi_ds_boxes')
CACHE = WORK / 'detect_cache.jsonl'
NAS_BOX_DIR = ROOT / 'data/processed/yolo_boxes'
NAS_MANIFEST = ROOT / 'data/processed/yolo_boxes_manifest.json'
MIN_BOX = 32
TINY_MIN = 16
EXTEND = 64
LIMIT = 50
MARGIN = 0.10
CONF = 0.05
BATCH = 32
JEDI_MARK = 'jedi_jodd_supplement'
DS_MARK = 'deepsea_expansion'


def atomic_json(path: Path, obj):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def is_frame(rel: str) -> bool:
    return JEDI_MARK in rel or DS_MARK in rel


def load_index():
    idx = json.load(open(IDX_PATH))
    rel2msp, keep = {}, defaultdict(int)
    for m in idx['morphospecies']:
        mid = m['morphospecies_id']
        for rel in m.get('images', []):
            if is_frame(rel):
                rel2msp[rel] = mid
            else:
                keep[mid] += 1
    print(f'index: 整帧 {len(rel2msp)} 张 / {len(set(rel2msp.values()))} 条目')
    return idx, rel2msp, keep


def run_detect(args, rel2msp):
    from scripts.utils.yolo_infer import DualPathDetector
    frames = sorted(f for f in rel2msp if (ROOT / f).exists())
    done = set()
    if CACHE.exists():
        with open(CACHE) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['frame_rel'])
                except Exception:
                    pass
    todo = [f for f in frames if f not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'[detect] 待推理 {len(todo)} 帧 (已完成 {len(done)})')
    if not todo:
        return
    WORK.mkdir(parents=True, exist_ok=True)
    det = DualPathDetector(str(WEIGHTS), device=args.device)
    t0 = time.time()
    n_done = 0
    with open(CACHE, 'a') as fout:
        for b0 in range(0, len(todo), BATCH):
            batch = todo[b0:b0 + BATCH]
            results = det.predict([str(ROOT / f) for f in batch], conf=CONF)
            for f, dets in zip(batch, results):
                fout.write(json.dumps({'frame_rel': f, 'dets': dets}, ensure_ascii=False) + '\n')
            fout.flush()
            n_done += len(batch)
            if (b0 // BATCH) % 10 == 0:
                rate = n_done / max(time.time() - t0, 1)
                eta = (len(todo) - n_done) / max(rate, 0.1) / 60
                print(f'  {n_done}/{len(todo)}, {rate:.1f} 帧/s, ETA {eta:.0f}min')
    print(f'[detect] 完成 {n_done} 帧, {time.time() - t0:.0f}s')


def load_cache():
    dets = {}
    with open(CACHE) as f:
        for line in f:
            try:
                r = json.loads(line)
                dets[r['frame_rel']] = r['dets']
            except Exception:
                pass
    return dets


def run_crop(args, rel2msp, keep):
    dets = load_cache()
    print(f'[crop] cache {len(dets)} 帧')
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = []
    if (WORK / 'boxes_manifest.json').exists():
        manifest = json.load(open(WORK / 'boxes_manifest.json'))
        print(f'  断点续跑: 已有 {len(manifest)} 框')
    done = {(r['msp_id'], Path(r['box_rel']).name) for r in manifest}

    # msp -> frame -> dets 过滤
    msp_frames = defaultdict(lambda: defaultdict(list))
    for f, dl in dets.items():
        mid = rel2msp.get(f)
        if mid is None:
            continue
        src = 'jedi' if JEDI_MARK in f else 'deepsea'
        with Image.open(ROOT / f) as im:
            W, H = im.size
        for d in dl:
            x1, y1, x2, y2 = d['box']
            w, h = x2 - x1, y2 - y1
            if min(w, h) < TINY_MIN:
                continue
            msp_frames[mid][f].append((src, [x1, y1, w, h], d.get('name', ''), float(d['conf'])))
    n_valid = sum(len(v) for d in msp_frames.values() for v in d.values())
    print(f'  有效框(≥{TINY_MIN}px): {n_valid} / {len(msp_frames)} 条目')

    # 50 上限 + 帧间 round-robin, 帧内 conf 降序
    plans = {}
    for mid, frames in msp_frames.items():
        budget = LIMIT - keep.get(mid, 0)
        if budget <= 0:
            continue
        for f in frames:
            frames[f].sort(key=lambda t: -t[3])
        take, round_i = [], 0
        while len(take) < min(budget, sum(len(v) for v in frames.values())):
            for f, boxes in frames.items():
                if round_i < len(boxes) and len(take) < budget:
                    take.append((f, boxes[round_i]))
            round_i += 1
        if take:
            plans[mid] = take
    n_planned = sum(len(v) for v in plans.values())
    print(f'  plans: {len(plans)} 条目, {n_planned} 框 (受 50 上限约束)')

    stems = sorted({Path(f).stem for take in plans.values() for f, _ in take})
    if args.limit:
        stems = stems[:args.limit]
    print(f'  待切 {len(stems)} 帧')
    # frame stem -> (frame_rel, [(msp, box)]) 索引
    stem_index = defaultdict(list)
    frame_of = {}
    for mid, take in plans.items():
        for f, box in take:
            stem_index[Path(f).stem].append((mid, box))
            frame_of[Path(f).stem] = f
    t0 = time.time()
    n_new = 0
    for i, stem in enumerate(stems):
        f = frame_of[stem]
        try:
            with Image.open(ROOT / f) as im:
                W, H = im.size
                rgb = im.convert('RGB')
        except Exception as e:
            print(f'  [跳过] {stem}: {e}')
            continue
        for mid, (src, b, name, conf) in stem_index[stem]:
            bname = f'{src}_{stem}_b{sum(1 for mm in manifest if mm["msp_id"] == mid and mm["source_stem"] == stem)}.jpg'
            key = (mid, bname)
            if key in done:
                continue
            x, y, w, h = b
            if min(w, h) < MIN_BOX:
                mx, my = max(int(w * MARGIN), int((EXTEND - w) / 2)), max(int(h * MARGIN), int((EXTEND - h) / 2))
            else:
                mx, my = int(w * MARGIN), int(h * MARGIN)
            x0, y0 = max(0, int(x) - mx), max(0, int(y) - my)
            x1, y1 = min(W, int(x + w) + mx), min(H, int(y + h) + my)
            if x1 - x0 < TINY_MIN or y1 - y0 < TINY_MIN:
                continue
            crop = rgb.crop((x0, y0, x1, y1))
            outdir = WORK / 'boxes' / mid
            outdir.mkdir(parents=True, exist_ok=True)
            crop.save(outdir / bname, 'JPEG', quality=95)
            manifest.append({'box_rel': f'data/processed/yolo_boxes/{mid}/{bname}',
                             'msp_id': mid, 'source': src, 'yolo_name': name, 'conf': conf,
                             'source_stem': stem, 'bbox_xywh': [round(x), round(y), round(w), round(h)],
                             'frame_rel': f})
            done.add(key)
            n_new += 1
        rgb.close()
        if (i + 1) % 300 == 0:
            atomic_json(WORK / 'boxes_manifest.json', manifest)
            print(f'  帧 {i + 1}/{len(stems)}, 框 {n_new}, {time.time() - t0:.0f}s')
    atomic_json(WORK / 'boxes_manifest.json', manifest)
    print(f'[crop] 完成: 新增 {n_new}, manifest 共 {len(manifest)}')


def run_sync():
    if not (WORK / 'boxes_manifest.json').exists():
        print('manifest 不存在, 先 --crop')
        return
    total = copied = 0
    for r in json.load(open(WORK / 'boxes_manifest.json')):
        total += 1
        src = WORK / 'boxes' / r['msp_id'] / Path(r['box_rel']).name
        dst = ROOT / r['box_rel']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)
            copied += 1
    atomic_json(NAS_MANIFEST, json.load(open(WORK / 'boxes_manifest.json')))
    print(f'[sync] {total} 框: 新复制 {copied}; manifest → {NAS_MANIFEST}')


def run_stats(rel2msp, keep):
    if not (WORK / 'boxes_manifest.json').exists():
        print('manifest 不存在')
        return
    man = json.load(open(WORK / 'boxes_manifest.json'))
    by_msp = defaultdict(int)
    for r in man:
        by_msp[r['msp_id']] += 1
    srcs = defaultdict(int)
    for r in man:
        srcs[r['source']] += 1
    tiny = sum(1 for r in man if min(r['bbox_xywh'][2], r['bbox_xywh'][3]) < MIN_BOX)
    print(f'manifest: {len(man)} 框 / {len(by_msp)} 条目; 来源 {dict(srcs)}; tiny(16-31px扩域) {tiny}')
    print('top5:', sorted(by_msp.items(), key=lambda kv: -kv[1])[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detect', action='store_true')
    ap.add_argument('--crop', action='store_true')
    ap.add_argument('--sync', action='store_true')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    idx, rel2msp, keep = load_index()
    if args.detect:
        run_detect(args, rel2msp)
    if args.crop:
        run_crop(args, rel2msp, keep)
    if args.sync:
        run_sync()
    if args.stats:
        run_stats(rel2msp, keep)


if __name__ == '__main__':
    main()
