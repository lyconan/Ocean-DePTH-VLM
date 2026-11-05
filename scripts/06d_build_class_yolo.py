#!/usr/bin/env python3
"""
06d_build_class_yolo.py — FathomNet 2553 细类 → 34 纲级 YOLO 检测数据集

数据源 (只读, 不修改原目录):
  /mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet/
    Train/*.png + *.txt (YOLO格式, 旧2553类id)   132,924 张
    Val/*.png + *.txt                             33,236 张
    class_mapping.json  {concept_name: old_id}
    Annotations/fathomnet_{train,val}.json (COCO, 用于框数统计)

折叠规则:
  1. concept → WoRMS 纲 (fathomnet_candidates.json 的 taxonomy_worms 链)
  2. 纲名合并: Teleostei→Actinopterygii, Stenolaemata→Gymnolaemata
  3. 框数 < 50 的纲 → 'Other' (Aves/Mammalia/Copepoda/Monoplacophora/Ostracoda/Petromyzonti/Solenogastres)
  4. 器械/垃圾类 (candidates pass=False 或无纲) → 丢弃
  5. 类别顺序: 按框数降序, Other 永远最后

输出: data/processed/fathomnet_class_yolo/
  images/{train,val}/*.png  → 符号链接指向 FathomNet 原图 (打包时 tar -h 解引用)
  labels/{train,val}/*.txt  → 重映射后的纲级标签
  data.yaml / class_table.json
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FN = Path('/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet')
CAND = ROOT / 'data/processed/fathomnet_expansion/fathomnet_candidates.json'
OUT = ROOT / 'data/processed/fathomnet_class_yolo'

MERGE = {
    'Teleostei': 'Actinopterygii',   # 索引用 Actinopterygii, FathomNet 用 Teleostei
    'Stenolaemata': 'Gymnolaemata',  # 苔藓虫统一
}
MIN_BOXES = 50  # 低于此框数的纲并入 Other


def concept_class(concept, cand):
    """concept → WoRMS 纲名 (含合并规则); 无法映射返回 None"""
    v = cand.get(concept)
    if not v or not v.get('pass'):
        return None
    tw = v.get('taxonomy_worms') or {}
    c = tw.get('class') or {}
    name = c.get('name') if isinstance(c, dict) else c
    if not name:
        return None
    return MERGE.get(name, name)


def main():
    cand = json.load(open(CAND))['concepts']
    cm = json.load(open(FN / 'class_mapping.json'))  # {concept: old_id}

    # ---- 1. 统计每个纲的框数 (COCO train+val), 决定小类 ----
    cls_boxes = Counter()
    for split in ['train', 'val']:
        coco = json.load(open(FN / f'Annotations/fathomnet_{split}.json'))
        cats = {c['id']: c['name'] for c in coco['categories']}
        for ann in coco['annotations']:
            cl = concept_class(cats[ann['category_id']], cand)
            if cl:
                cls_boxes[cl] += 1
    small = {c for c, n in cls_boxes.items() if n < MIN_BOXES}
    kept = [c for c, _ in cls_boxes.most_common() if c not in small]
    names = kept + ['Other']                      # Other 永远最后
    new_id = {c: i for i, c in enumerate(names)}
    print(f'纲级类别 {len(names)} 个 (含 Other); 并入 Other 的小纲: {sorted(small)}')

    # ---- 2. old_id → new_id 映射 ----
    id_map = {}
    for concept, old in cm.items():
        cl = concept_class(concept, cand)
        id_map[int(old)] = new_id['Other' if (cl and cl in small) else cl] if cl else None

    # ---- 3. 重写标签 + 建立符号链接 ----
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, src in [('train', FN / 'Train'), ('val', FN / 'Val')]:
        (OUT / f'images/{split}').mkdir(parents=True, exist_ok=True)
        (OUT / f'labels/{split}').mkdir(parents=True, exist_ok=True)
        box_cnt = Counter(); img_cnt = Counter(); n_img = n_lbl = 0; dropped = 0
        for txt in sorted(src.glob('*.txt')):
            lines_out = []
            for ln in txt.read_text().splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    continue
                old = int(parts[0])
                if old not in id_map or id_map[old] is None:
                    dropped += 1
                    continue
                x, y, w, h = (float(v) for v in parts[1:5])
                if w <= 0 or h <= 0 or w > 1 or h > 1:  # 退化框
                    dropped += 1
                    continue
                lines_out.append(f'{id_map[old]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
                box_cnt[names[id_map[old]]] += 1
            if not lines_out:      # 全部是非生物框的图 → 不入数据集
                continue
            stem = txt.stem
            (OUT / f'labels/{split}/{stem}.txt').write_text('\n'.join(lines_out) + '\n')
            img_link = OUT / f'images/{split}/{stem}.png'
            if not img_link.exists():
                os.symlink(src / f'{stem}.png', img_link)
            n_img += 1
            for ln in lines_out:
                img_cnt[names[int(ln.split()[0])]] += 1
            n_lbl += 1
        stats[split] = {'images': n_img, 'boxes': sum(box_cnt.values()),
                        'per_class': dict(box_cnt)}
        print(f'[{split}] 图 {n_img}, 框 {sum(box_cnt.values())}, 丢弃框 {dropped}')
        sys.stdout.flush()

    # ---- 4. data.yaml (path 留给训练脚本按运行环境改写绝对路径) ----
    yaml_lines = ['path: .', 'train: images/train', 'val: images/val', f'nc: {len(names)}', 'names:']
    yaml_lines += [f'  {new_id[c]}: {c}' for c in names]
    (OUT / 'data.yaml').write_text('\n'.join(yaml_lines) + '\n')

    table = {c: {'boxes': stats['train']['per_class'].get(c, 0) + stats['val']['per_class'].get(c, 0),
                 'train_imgs': stats['train']['per_class'].get(c, 0),
                 'val_imgs': stats['val']['per_class'].get(c, 0)} for c in names}
    table['_meta'] = {'n_classes': len(names), 'merged_small': sorted(small),
                      'merge_rules': MERGE, 'total_images': stats['train']['images'] + stats['val']['images']}
    json.dump(table, open(OUT / 'class_table.json', 'w'), ensure_ascii=False, indent=1)

    print(f'\n=== 完成: {OUT} ===')
    print(f"{'类别':<22}{'总框数':>9}{'train框':>9}{'val框':>8}")
    for c in names:
        print(f"  {c:<22}{table[c]['boxes']:>9}{table[c]['train_imgs']:>9}{table[c]['val_imgs']:>8}")


if __name__ == '__main__':
    main()
