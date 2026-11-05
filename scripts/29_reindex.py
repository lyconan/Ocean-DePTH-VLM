#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
29_reindex.py — 将 LoRA 微调后的全库嵌入写回 Neo4j (在 docker_default 网络内运行)
输入: /exp/meta.jsonl (id 顺序) + /exp/emb_ft.f32 (36147 x 1024 f32)
写入: MATCH (e:EntryImage {image_id: row.id}) SET e.embedding = row.emb  (批量 UNWIND, 500/批)
注: Neo4j 5.20 向量索引在属性写入时自动增量更新, 无需显式重建索引;
    完成后用 db.index.vector.queryNodes 抽样验证索引已反映新嵌入。
"""
import json
import time

import numpy as np
from neo4j import GraphDatabase

DIM = 1024
BATCH = 500
URI = "bolt://depth_vlm_neo4j:7687"
AUTH = ("neo4j", "password")


def main():
    meta = [json.loads(l) for l in open("/exp/meta.jsonl")]
    mat = np.fromfile("/exp/emb_ft2.f32", dtype=np.float32).reshape(-1, DIM)
    assert len(mat) == len(meta), f"{len(mat)} vs {len(meta)}"
    print(f"载入 {len(meta)} 条嵌入, 开始写回...", flush=True)

    drv = GraphDatabase.driver(URI, auth=AUTH)
    t0 = time.time()
    with drv.session() as s:
        for k in range(0, len(meta), BATCH):
            # 注意: 全局索引 = k + j (此前 enumerate 从 0 起算曾导致每批错写前 500 行)
            rows = [{"id": meta[k + j]["id"], "emb": mat[k + j].tolist()}
                    for j in range(min(BATCH, len(meta) - k))]
            s.run("""
                UNWIND $rows AS row
                MATCH (e:EntryImage {image_id: row.id})
                SET e.embedding = row.emb
            """, rows=rows)
            done = min(k + BATCH, len(meta))
            print(f"  {done}/{len(meta)}  {time.time()-t0:.0f}s", flush=True)

        # 验证
        rec = s.run("""
            MATCH (e:EntryImage) WHERE e.embedding IS NOT NULL
            RETURN count(e) AS n
        """).single()
        print(f"带嵌入节点总数: {rec['n']}", flush=True)

        # 向量索引抽样验证: 取一个查询图, 用索引检索, 确认 top1 是新嵌入空间的最近邻
        qid = meta[0]["id"]
        qv = mat[0].tolist()
        recs = s.run("""
            CALL db.index.vector.queryNodes('entryimage_embeddings', 3, $qv)
            YIELD node, score
            RETURN node.image_id AS id, score
        """, qv=qv).data()
        print("向量索引抽样 (查询", qid, "):", flush=True)
        for r in recs:
            # 与新嵌入矩阵的直接最近邻对照
            j = [m["id"] for m in meta].index(r["id"]) if r["id"] in id_set else -1
            print(f"  {r['id']}  score={r['score']:.4f}", flush=True)
    drv.close()
    print(f"完成, 总耗时 {time.time()-t0:.0f}s", flush=True)


id_set = set()

if __name__ == "__main__":
    with open("/exp/meta.jsonl") as f:
        id_set = {json.loads(l)["id"] for l in f}
    main()
