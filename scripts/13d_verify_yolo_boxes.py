#!/usr/bin/env python3
"""13d_verify_yolo_boxes.py — YOLO 框两道校验 (二期入库前)

L1 纲级一致性: YOLO 34 类名上级链 {name, class, phylum} 与条目 lineage 求交,
   不相交 → reject_l1 (如珊瑚条目收到鱼框); 'Other'/未知类名不拒。
L2 BioCLIP zero-shot: 框图特征 vs 文本相似度,
   正类 = 条目谱系各级名 + latin ("a photo of a {x}")
   负类 = 海底背景/异物组 + 异类动物组 (剔除与条目同门/同纲的近亲项)
   s_pos > s_neg 才保留 → reject_l2。

输出: manifest 每行加 verdict/s_pos/s_neg; 保留子集写回 boxes_manifest.json
      (原始备份 .bak_raw), reject 框文件留在 /tmp 不 sync。

用法 (容器内): export PYTHONPATH=项目根; python3 scripts/13d_verify_yolo_boxes.py [--device cuda:0]
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
WORK = Path('/tmp/jedi_ds_boxes')
MAN = WORK / 'boxes_manifest.json'
IDX = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
BIOCLIP_NAME = 'hf-hub:imageomics/bioclip-2.5-vith14'
BATCH = 64

CL34 = {
    'Octocorallia': ('Anthozoa', 'Cnidaria'), 'Hexacorallia': ('Anthozoa', 'Cnidaria'),
    'Hydrozoa': (None, 'Cnidaria'), 'Scyphozoa': (None, 'Cnidaria'),
    'Malacostraca': (None, 'Arthropoda'), 'Thecostraca': (None, 'Arthropoda'),
    'Pycnogonida': (None, 'Arthropoda'),
    'Holothuroidea': (None, 'Echinodermata'), 'Echinoidea': (None, 'Echinodermata'),
    'Asteroidea': (None, 'Echinodermata'), 'Crinoidea': (None, 'Echinodermata'),
    'Ophiuroidea': (None, 'Echinodermata'),
    'Actinopterygii': (None, 'Chordata'), 'Elasmobranchii': (None, 'Chordata'),
    'Holocephali': (None, 'Chordata'), 'Myxini': (None, 'Chordata'),
    'Thaliacea': (None, 'Chordata'), 'Appendicularia': (None, 'Chordata'),
    'Ascidiacea': (None, 'Chordata'),
    'Polychaeta': (None, 'Annelida'), 'Clitellata': (None, 'Annelida'),
    'Demospongiae': (None, 'Porifera'), 'Hexactinellida': (None, 'Porifera'),
    'Cephalopoda': (None, 'Mollusca'), 'Bivalvia': (None, 'Mollusca'),
    'Gastropoda': (None, 'Mollusca'), 'Polyplacophora': (None, 'Mollusca'),
    'Tentaculata': (None, 'Ctenophora'), 'Nuda': (None, 'Ctenophora'),
    'Rhynchonellata': (None, 'Brachiopoda'), 'Sagittoidea': (None, 'Chaetognatha'),
    'Enteropneusta': (None, 'Hemichordata'), 'Gymnolaemata': (None, 'Bryozoa'),
}
NEG_BG = [
    'the empty seafloor sediment', 'a bare rock on the seafloor',
    'marine snow particles in water', 'murky water with no animal',
    'air bubbles in water', 'a rope or cable on the seafloor',
    'plastic debris on the seafloor', 'a blurred empty background',
]
NEG_ANIMALS = [
    ('a fish', 'Chordata'), ('a shark or ray', 'Chordata'), ('a tunicate', 'Chordata'),
    ('a crab', 'Arthropoda'), ('a shrimp', 'Arthropoda'), ('a lobster', 'Arthropoda'),
    ('a barnacle', 'Arthropoda'), ('an amphipod', 'Arthropoda'), ('an isopod', 'Arthropoda'),
    ('a sea star', 'Echinodermata'), ('a brittle star', 'Echinodermata'),
    ('a sea urchin', 'Echinodermata'), ('a sea cucumber', 'Echinodermata'),
    ('a feather star', 'Echinodermata'),
    ('a jellyfish', 'Cnidaria'), ('a coral', 'Cnidaria'),
    ('a sea anemone', 'Cnidaria'), ('a sea pen', 'Cnidaria'),
    ('a squid', 'Mollusca'), ('an octopus', 'Mollusca'),
    ('a clam or bivalve', 'Mollusca'), ('a snail', 'Mollusca'), ('a chiton', 'Mollusca'),
    ('a worm', 'Annelida'), ('a sponge', 'Porifera'),
    ('a bryozoan', 'Bryozoa'), ('a brachiopod', 'Brachiopoda'),
]
TPL = 'a photo of a {}'


def entry_names(m):
    vals = []
    lin = m.get('lineage') or {}
    for k in ['phylum', 'class', 'order', 'family', 'genus']:
        v = lin.get(k)
        if v:
            vals.append(v.strip())
    if m.get('latin'):
        vals.append(m['latin'].strip())
    return [v for v in vals if v]


def entry_kin(m):
    lin = m.get('lineage') or {}
    return {lin.get('phylum'), lin.get('class')} - {None, ''}


def l1_pass(yname, names):
    if yname == 'Other' or yname not in CL34:
        return True
    chain = {x for x in CL34[yname] if x}
    return bool(chain & set(names))


def neg_texts(m):
    kin = entry_kin(m)
    return NEG_BG + [t for t, ph in NEG_ANIMALS if ph not in kin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    man = json.load(open(MAN))
    idx = json.load(open(IDX))
    m2m = {m['morphospecies_id']: m for m in idx['morphospecies']}

    bak = MAN.with_suffix('.json.bak_raw')
    if not bak.exists():
        shutil.copy2(MAN, bak)
        print(f'[备份] {bak}')
    else:
        man = json.load(open(bak))          # 从原始重跑 (幂等)
        print('[断点重跑] 从 .bak_raw 恢复')

    # ---- L1 ----
    n_l1 = 0
    for r in man:
        m = m2m[r['msp_id']]
        names = entry_names(m)
        if not l1_pass(r.get('yolo_name', 'Other'), names):
            r['verdict'] = 'reject_l1'
            n_l1 += 1
        else:
            r['verdict'] = None
    print(f'[L1] 纲级拒识 {n_l1}/{len(man)}')

    # 单框帧采信 (用户方案): trusted 框跳过 L2, 直接 keep
    n_tr = 0
    for r in man:
        if r.get('trusted') and r['verdict'] is None:
            r['verdict'] = 'keep'
            n_tr += 1
    if n_tr:
        print(f'[采信] trusted 单框帧 {n_tr} 框跳过 L2')

    todo = [r for r in man if r['verdict'] is None]
    print(f'[L2] 待判别 {len(todo)} 框')

    from scripts.utils.bioclip_embed import get_bioclip_model, extract_embeddings_batch
    model, processor = get_bioclip_model(device=args.device)
    import open_clip
    tokenizer = open_clip.get_tokenizer(BIOCLIP_NAME)

    # ---- 文本编码 (唯一文本 cache) ----
    uniq = {}
    for r in todo:
        m = m2m[r['msp_id']]
        for t in entry_names(m):
            uniq.setdefault(TPL.format(t), None)
        for t in neg_texts(m):
            uniq.setdefault(t, None)
    texts = sorted(uniq)
    tvecs = []
    with torch.no_grad():
        for b0 in range(0, len(texts), 256):
            toks = tokenizer(texts[b0:b0 + 256]).to(args.device)
            e = model.encode_text(toks)
            e = e / e.norm(dim=-1, keepdim=True)
            tvecs.append(e.float().cpu().numpy())
    T = np.concatenate(tvecs)
    t2i = {t: i for i, t in enumerate(texts)}
    print(f'[L2] 文本 {len(texts)} 条编码完成')

    # per-entry 索引 (仅为 L2 待判别条目构建, 与文本编码集合一致)
    todo_mids = {r['msp_id'] for r in todo}
    pos_idx, neg_idx = {}, {}
    for mid in todo_mids:
        m = m2m[mid]
        pos_idx[mid] = [t2i[TPL.format(t)] for t in entry_names(m)]
        neg_idx[mid] = [t2i[t] for t in neg_texts(m)]

    # ---- 图像特征 + 判别 ----
    t0 = time.time()
    n_rej = 0
    for b0 in range(0, len(todo), BATCH):
        batch = todo[b0:b0 + BATCH]
        paths = [str(WORK / 'boxes' / r['msp_id'] / Path(r['box_rel']).name) for r in batch]
        Q = extract_embeddings_batch(paths, model, processor, batch_size=BATCH, device=args.device)
        Q = Q.numpy() if hasattr(Q, 'numpy') else np.asarray(Q)
        Q = Q.astype(np.float32)
        S = Q @ T.T
        for i, r in enumerate(batch):
            sp = float(S[i, pos_idx[r['msp_id']]].max())
            sn = float(S[i, neg_idx[r['msp_id']]].max())
            r['s_pos'], r['s_neg'] = round(sp, 4), round(sn, 4)
            r['verdict'] = 'keep' if sp > sn else 'reject_l2'
            n_rej += (r['verdict'] == 'reject_l2')
        if (b0 // BATCH) % 20 == 0:
            rate = (b0 + len(batch)) / max(time.time() - t0, 1)
            print(f'  {b0 + len(batch)}/{len(todo)}, {rate:.0f} 图/s, L2拒 {n_rej}')

    kept = [r for r in man if r['verdict'] == 'keep']
    print(f'[L2] 拒识 {n_rej}; 保留 {len(kept)}/{len(man)} '
          f'(L1 {n_l1} + L2 {n_rej} = {len(man) - len(kept)} 拒, {100 * (len(man) - len(kept)) / len(man):.1f}%)')
    tmp = MAN.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(kept, f, ensure_ascii=False)
    os.replace(tmp, MAN)
    check = json.load(open(MAN))
    assert all(r['verdict'] == 'keep' for r in check)
    print(f'[完成] manifest {len(check)} 框 (仅 keep), 断言通过')


if __name__ == '__main__':
    main()
