#!/usr/bin/env python3
"""12d_rebuild_kg_with_boxes.py — Neo4j EntryImage 库增量重建 (方案A)

变更:
  1. 删除 FN 整帧 EntryImage (image_id 前缀 data/processed/fathomnet_expansion/species/)
  2. MERGE 新框切图 EntryImage (嵌入来自 emb_boxes_shard{0,1}.npz) + HAS_IMAGE
  3. QA 审计 (条目/图/索引)
幂等, 可重跑。前置: 12b 已更新 index (整帧退出/框图加入), 12c 已提嵌入。

用法: python scripts/12d_rebuild_kg_with_boxes.py [--dry-run]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

ROOT = Path('/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM')
IDX_PATH = ROOT / 'data/processed/atlas_lookup/morphospecies_index.json'
EMB_DIR = ROOT / 'data/processed/bioclip25_embeddings'
FRAME_PREFIX = 'data/processed/fathomnet_expansion/species/'
NEO4J_URI = 'bolt://172.17.0.1:7687'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = 'password'
BATCH = 500


def load_box_embeddings():
    embs = {}
    npz_files = [EMB_DIR / f'emb_boxes_shard{i}.npz' for i in [0, 1]]
    npz_files += [EMB_DIR / f'emb_boxes_sup_shard{i}.npz' for i in [0, 1]]
    npz_files += [EMB_DIR / 'emb_boxes_tiny.npz']
    npz_files += [EMB_DIR / f'emb_yolo_shard{i}.npz' for i in [0, 1]]
    npz_files += [EMB_DIR / 'emb_manual_fallback.npz']
    for npz in npz_files:
        if not npz.exists():
            print(f'警告: {npz} 不存在')
            continue
        z = np.load(npz)
        for p, e in zip(z['image_paths'].tolist(), z['embeddings']):
            embs[p] = e.astype(np.float32)
    print(f'框嵌入: {len(embs)}')
    return embs


def q(session, cypher, **params):
    return session.run(cypher, **params).data()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    idx = json.load(open(IDX_PATH))
    msps = idx['morphospecies']
    embs = load_box_embeddings()

    # index 全部引用 rel (差集删除的保留集合)
    valid_rels = [rel for m in msps for rel in m.get('images', [])]
    n_frames_kept = sum(1 for r in valid_rels if r.startswith(FRAME_PREFIX))
    box_rels = {rel for m in msps for rel in m.get('images', [])
                if rel.startswith('data/processed/fathomnet_expansion/boxes/')}
    n_box_emb = sum(1 for r in box_rels if r in embs)
    print(f'index: {len(msps)} 条目; 引用整帧 {n_frames_kept} (无框替代条目, 保留); '
          f'框图 {len(box_rels)}, 有嵌入 {n_box_emb}')

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            before = q(session, "MATCH (ei:EntryImage) RETURN count(ei) AS c")[0]['c']
            print(f'现状 EntryImage: {before}')

            # ---- 1. 删除 index 不再引用的 EntryImage (差集, 保留整帧引用) ----
            t0 = time.time()
            if not args.dry_run:
                q(session, """
                    MATCH (ei:EntryImage)
                    WHERE NOT ei.image_id IN $valid
                    DETACH DELETE ei
                """, valid=valid_rels)
            orphans = q(session, """
                MATCH (ei:EntryImage) WHERE NOT ei.image_id IN $valid
                RETURN count(ei) AS c
            """, valid=valid_rels)[0]['c']
            print(f'孤儿 EntryImage (index 不引用): 删除后残留 {orphans} ({time.time()-t0:.0f}s)')

            # ---- 2. MERGE 框切图 EntryImage ----
            t0 = time.time()
            img_rows = []
            for m in msps:
                for rel in m.get('images', []):
                    emb = embs.get(rel)
                    if emb is not None:
                        img_rows.append({'msp_id': m['morphospecies_id'],
                                         'image_id': rel, 'embedding': emb.tolist()})
            if not args.dry_run:
                for b0 in range(0, len(img_rows), BATCH):
                    batch = img_rows[b0:b0 + BATCH]
                    q(session, """
                        UNWIND $rows AS row
                        MATCH (ms:Morphospecies {morphospecies_id: row.msp_id})
                        MERGE (ei:EntryImage {image_id: row.image_id})
                        SET ei.msp_id = row.msp_id, ei.embedding = row.embedding
                        MERGE (ms)-[:HAS_IMAGE]->(ei)
                    """, rows=batch)
                    if (b0 // BATCH) % 10 == 0:
                        print(f'  框 EntryImage {b0+len(batch)}/{len(img_rows)} ({time.time()-t0:.0f}s)')
            print(f'框 EntryImage: {len(img_rows)} ({time.time()-t0:.0f}s)')

            # ---- 3. QA ----
            print('\n===== QA 审计 =====')
            for cy, label, extra in [
                ("MATCH (ei:EntryImage) RETURN count(ei) AS c", "EntryImage 总数", {}),
                ("MATCH (ei:EntryImage) WHERE ei.image_id STARTS WITH 'data/processed/fathomnet_expansion/boxes/' RETURN count(ei) AS c", "其中框切图", {}),
                ("MATCH (ei:EntryImage) WHERE ei.image_id STARTS WITH $pfx RETURN count(ei) AS c", "其中保留整帧", {'pfx': FRAME_PREFIX}),
                ("MATCH (ei:EntryImage) WHERE ei.embedding IS NULL RETURN count(ei) AS c", "无嵌入图(应0)", {}),
                ("MATCH (ei:EntryImage) WHERE NOT ei.image_id IN $valid RETURN count(ei) AS c", "孤儿(index不引用,应0)", {'valid': valid_rels}),
            ]:
                print(f'  {label}: {q(session, cy, **extra)[0]["c"]}')
            for r in q(session, "SHOW INDEXES YIELD name, type, state WHERE type='VECTOR' RETURN name, state"):
                print(f'  向量索引: {r["name"]} ({r["state"]})')
    finally:
        driver.close()


if __name__ == '__main__':
    main()
