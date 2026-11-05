#!/usr/bin/env python3
"""13h_verify_selection.py — 多框帧选优效果的独立验证 (自检索 + YOLO 类名双角度)

验证逻辑:
  A. 自检索一致性: 每个 JEDI/深海框在 FathomNet 框图库 (27k+ 已确认归属的框嵌入)
     中找 top-1 最近邻, 看邻居条目与框所属条目的谱系一致率 (genus/family/class/phylum)。
     对比【选中的框】(每帧 s_pos argmax) vs【被丢弃的框】(同帧落选者):
     若选优有效, 选中框的邻居一致率应显著更高。
  B. YOLO 类名角度: 选中框 vs 丢弃框的 yolo_name 与条目谱系的命中层级分布
     (Other 占比、最高命中阶元)。

输入: emb_yolo_shard{0,1}.npz (13c, 选中框嵌入) + WORK/boxes_manifest.json (选中)
     + WORK/boxes_manifest.json.bak_l2all (选优前全集 → 差集=丢弃框, 需现算嵌入)
     + FN 框图库嵌入 emb_boxes_shard{0,1}/emb_boxes_sup_shard{0,1}/emb_boxes_tiny.npz
输出: 终端报告 + WORK/verify_selection.json

用法 (容器内): PYTHONPATH=项目根 python3 scripts/13h_verify_selection.py [--device cuda:0]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
WORK = Path('/tmp/jedi_ds_boxes')
MAN = WORK / 'boxes_manifest.json'
BAK = WORK / 'boxes_manifest.json.bak_l2all'
EMB = ROOT / 'data/processed/bioclip25_embeddings'
IDX = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
BIOCLIP_NAME = 'hf-hub:imageomics/bioclip-2.5-vith14'

FN_NPZ = ['emb_boxes_shard0.npz', 'emb_boxes_shard1.npz',
          'emb_boxes_sup_shard0.npz', 'emb_boxes_sup_shard1.npz',
          'emb_boxes_tiny.npz']


def msp_of(rel):
    return Path(rel).parts[-2]


def lineage_of(m):
    lin = m.get('lineage') or {}
    return {k: lin.get(k) for k in ['phylum', 'class', 'order', 'family', 'genus']}


def agree_level(lin_a, lin_b):
    """两谱系最高一致阶元 (深→浅): genus>family>order>class>phylum>无"""
    for k in ['genus', 'family', 'order', 'class', 'phylum']:
        a, b = lin_a.get(k), lin_b.get(k)
        if a and b and a == b:
            return k
    return None


def load_lib():
    vs, mids = [], []
    for name in FN_NPZ:
        z = np.load(EMB / name, allow_pickle=True)
        vs.append(z['embeddings'].astype(np.float32))
        mids.extend(msp_of(p) for p in z['image_paths'])
    R = np.concatenate(vs)
    n = np.linalg.norm(R, axis=1, keepdims=True)
    R = R / np.maximum(n, 1e-8)
    return R, mids


def top1_agree(Q, R, r_mids, q_mids, m2m, topk=5):
    """返回每个查询的 (top1 agree_level, topk 多数 agree_level, top1 邻居 mid)"""
    out = []
    r_arr = np.array(r_mids)
    for b0 in range(0, len(Q), 512):
        S = Q[b0:b0 + 512] @ R.T
        idx1 = S.argmax(axis=1)
        idxk = np.argpartition(-S, topk, axis=1)[:, :topk]
        for i in range(len(S)):
            q_lin = lineage_of(m2m[q_mids[b0 + i]])
            n1 = lineage_of(m2m[r_mids[idx1[i]]])
            lv1 = agree_level(q_lin, n1)
            lvk = Counter(x for x in (agree_level(q_lin, lineage_of(m2m[m]))
                                      for m in r_arr[idxk[i]]) if x)
            lvk = lvk.most_common(1)[0][0] if lvk else None
            out.append((lv1, lvk, r_mids[idx1[i]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    idx = json.load(open(IDX))
    m2m = {m['morphospecies_id']: m for m in idx['morphospecies']}
    man = json.load(open(MAN))
    bak = json.load(open(BAK))
    sel_keys = {(r['msp_id'], Path(r['box_rel']).name) for r in man}
    dropped = [r for r in bak if (r['msp_id'], Path(r['box_rel']).name) not in sel_keys]
    print(f'[数据] 选中 {len(man)} / 丢弃 {len(dropped)}')

    # ---- A. 自检索 ----
    R, r_mids = load_lib()
    print(f'[库] FN 框图库 {R.shape[0]} 嵌入 / {len(set(r_mids))} 条目')
    sel_npz = [np.load(EMB / f'emb_yolo_shard{i}.npz', allow_pickle=True) for i in (0, 1)]
    Q_sel = np.concatenate([z['embeddings'].astype(np.float32) for z in sel_npz])
    sel_mids = [msp_of(p) for z in sel_npz for p in z['image_paths']]
    n = np.linalg.norm(Q_sel, axis=1, keepdims=True)
    Q_sel = Q_sel / np.maximum(n, 1e-8)
    assert len(sel_mids) == len(man), '嵌入数与 manifest 不一致'

    from scripts.utils.bioclip_embed import get_bioclip_model, extract_embeddings_batch
    model, processor = get_bioclip_model(device=args.device)
    paths = [str(WORK / 'boxes' / r['msp_id'] / Path(r['box_rel']).name) for r in dropped]
    Q_drop = extract_embeddings_batch(paths, model, processor, batch_size=64, device=args.device)
    Q_drop = Q_drop.numpy() if hasattr(Q_drop, 'numpy') else np.asarray(Q_drop)
    Q_drop = Q_drop.astype(np.float32)
    Q_drop = Q_drop / np.maximum(np.linalg.norm(Q_drop, axis=1, keepdims=True), 1e-8)

    RANK = ['genus', 'family', 'order', 'class', 'phylum', None]
    def agg(res):
        c1 = Counter(x[0] for x in res)
        ck = Counter(x[1] for x in res)
        n = len(res)
        return {f'top1_{k or "none"}': round(100 * c1.get(k, 0) / n, 1) for k in RANK} | \
               {f'top5maj_{k or "none"}': round(100 * ck.get(k, 0) / n, 1) for k in RANK}

    print('\n===== A. 自检索一致率 (FN 框图库 top-1 / top-5 多数) =====')
    a_sel = agg(top1_agree(Q_sel, R, r_mids, sel_mids, m2m))
    a_drop = agg(top1_agree(Q_drop, R, r_mids, [r['msp_id'] for r in dropped], m2m))
    keys = sorted(a_sel)
    print(f'{"指标":<22}{"选中框":>10}{"丢弃框":>10}')
    for k in keys:
        print(f'{k:<24}{a_sel[k]:>9}%{a_drop[k]:>9}%')

    # ---- B. YOLO 类名角度 ----
    from utils.module_loader import load_script_module
    m13d = load_script_module('13d_verify_yolo_boxes')
    def yolo_stat(rows):
        c = Counter()
        for r in rows:
            m = m2m[r['msp_id']]
            lin = m.get('lineage') or {}
            yn = r.get('yolo_name', 'Other')
            if yn == 'Other' or yn not in m13d.CL34:
                c['Other/未知'] += 1
                continue
            chain = {x for x in m13d.CL34[yn] if x}
            lv = ('class' if lin.get('class') and lin['class'] in chain
                  else 'phylum' if lin.get('phylum') and lin['phylum'] in chain
                  else '无')
            c[lv] += 1
        n = len(rows)
        return {k: round(100 * v / n, 1) for k, v in c.items()}
    print('\n===== B. YOLO 34 类名与条目谱系命中层级 (L1 内部) =====')
    b_sel, b_drop = yolo_stat(man), yolo_stat(dropped)
    for k in sorted(set(b_sel) | set(b_drop)):
        print(f'  {k:<10} 选中 {b_sel.get(k, 0)}% | 丢弃 {b_drop.get(k, 0)}%')

    report = {'selected': {'n': len(man), 'retrieval': a_sel, 'yolo': b_sel},
              'dropped': {'n': len(dropped), 'retrieval': a_drop, 'yolo': b_drop}}
    json.dump(report, open(WORK / 'verify_selection.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\n[完成] 报告 → {WORK / "verify_selection.json"}')


if __name__ == '__main__':
    main()
