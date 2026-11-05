#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
35_build_centroids.py — 从 Neo4j (v2 嵌入, 权威来源) 构建 mid 质心矩阵
输入: EntryImage.embedding (36,147 x 1024, LoRA v2 重提后)
输出: /exp/centroids_v2.npz  { C: (n_mid,1024) f32 L2归一化, mids: str数组 }
用途: 查询端融合重排 score(mid) = 0.5*max图相似 + 0.5*(查询·质心)
      部署位置: data/experiments/ft_20261102/centroids_v2.npz (宿主侧拷贝)
      库变更(增删EntryImage/换模型)后需重跑本脚本刷新
运行: docker run --rm --network docker_default -v /tmp/exp:/exp -v $WS:/work:ro \
        yolo34_train:v1 env PYTHONPATH=/exp/pylibs \
        /root/miniconda3/envs/vllm/bin/python /work/scripts/35_build_centroids.py
"""
import numpy as np
from neo4j import GraphDatabase

DIM = 1024
URI = "bolt://depth_vlm_neo4j:7687"
AUTH = ("neo4j", "password")
OUT = "/exp/centroids_v2.npz"


def main():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    mids_all, vecs = [], []
    with drv.session() as s:
        # 分批拉取 (SKIP/OFFSET 稳定: 事务内不写入)
        total = s.run(
            "MATCH (e:EntryImage) WHERE e.embedding IS NOT NULL "
            "RETURN count(e) AS n").single()["n"]
        print(f"带嵌入 EntryImage: {total}", flush=True)
        off = 0
        while off < total:
            rows = s.run(
                "MATCH (e:EntryImage) WHERE e.embedding IS NOT NULL "
                "RETURN e.msp_id AS mid, e.embedding AS emb "
                "ORDER BY e.image_id SKIP $off LIMIT 2000", off=off).data()
            if not rows:
                break
            for r in rows:
                mids_all.append(r["mid"])
                vecs.append(np.asarray(r["emb"], dtype=np.float32))
            off += len(rows)
            print(f"  拉取 {off}/{total}", flush=True)
    drv.close()

    mat = np.stack(vecs)
    assert mat.shape == (total, DIM), f"{mat.shape} vs {(total, DIM)}"
    mids_arr = np.array(mids_all)
    uniq = sorted(set(mids_all))
    idx_of = {m: i for i, m in enumerate(uniq)}
    C = np.zeros((len(uniq), DIM), dtype=np.float32)
    cnt = np.zeros(len(uniq), dtype=np.float32)
    for i, u in enumerate(uniq):
        rows = mat[mids_arr == u]
        C[i] = rows.mean(axis=0)
        cnt[i] = len(rows)
    C /= np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12)
    print(f"质心: {C.shape[0]} 个 mid (单图中位 {int(np.median(cnt))} 张, "
          f"最大 {int(cnt.max())} 张)", flush=True)
    np.savez(OUT, C=C, mids=np.array(uniq), cnt=cnt)
    print(f"已保存 {OUT}", flush=True)


if __name__ == "__main__":
    main()
