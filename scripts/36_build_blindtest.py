#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36_build_blindtest.py — 构建盲测集: 将抽中的 val 源图所有框从检索库物理移除
目的: 消除"查询命中库里自己/同源图"的开卷效应 — 盲测查询在库中不存在任何同源图

选择规则 (seed=42 可复现):
  1. FN val 框按源图 uuid 分组; 每个 mid 优先从"单 mid 源图"中随机抽 1 个 uuid
  2. 该 uuid 名下所有框 (任意 mid) 全部抽为盲测
  3. 安全环: 若某 mid 因此在库中清零, 其 uuid 退回画廊 (该 mid 不可测但不破坏库)
  4. 删除 EntryImage 节点 -> 重建向量索引; 删除前备份嵌入 (可 --restore 一键恢复)

输入: $DEPTH_WORK/data/processed/fathomnet_expansion/boxes/, $DEPTH_EXP/meta.jsonl + emb_ft2.f32
输出: $DEPTH_EXP/blindtest/manifest.json + removed_backup.npz
运行: docker run --rm --gpus '"device=0"' --network docker_default --shm-size 8g \
        -v /tmp/exp:/exp -v $WS:/work:ro \
        -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
        yolo34_train:v1 env HF_HUB_OFFLINE=1 PYTHONPATH=/exp/pylibs \
        /root/miniconda3/envs/vllm/bin/python /work/scripts/36_build_blindtest.py
环境变量: NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD, DEPTH_WORK, DEPTH_EXP (均有原值作默认)
"""
import os
import json
import random
import numpy as np
from neo4j import GraphDatabase

WORK = os.environ.get("DEPTH_WORK", "/work")
EXP = os.environ.get("DEPTH_EXP", "/exp")
URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password"))
BOX_DIR = os.path.join(WORK, "data/processed/fathomnet_expansion/boxes")
OUT_DIR = os.path.join(EXP, "blindtest")
DIM = 1024
SEED = 42


def scan_boxes():
    """boxes/<MID>/val_<uuid>_b<k>.jpg -> [(mid, uuid, image_id)]"""
    recs = []
    for mid in sorted(os.listdir(BOX_DIR)):
        d = os.path.join(BOX_DIR, mid)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not (fn.startswith("val_") and fn.endswith(".jpg")):
                continue
            uuid = fn[len("val_"):].rsplit("_b", 1)[0]
            recs.append((mid, uuid, f"data/processed/fathomnet_expansion/boxes/{mid}/{fn}"))
    return recs


def main(restore=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    drv = GraphDatabase.driver(URI, auth=AUTH)

    if restore:
        z = np.load(f"{OUT_DIR}/removed_backup.npz", allow_pickle=False)
        ids, embs = list(z["ids"]), z["embs"]
        mids = list(z["mids"])
        with drv.session() as s:
            for k in range(0, len(ids), 500):
                rows = [{"id": ids[k + j], "mid": mids[k + j],
                         "emb": embs[k + j].tolist()}
                        for j in range(min(500, len(ids) - k))]
                s.run("""
                    UNWIND $rows AS row
                    MATCH (m:Morphospecies {morphospecies_id: row.mid})
                    MERGE (e:EntryImage {image_id: row.id})
                    SET e.msp_id = row.mid, e.embedding = row.emb
                    MERGE (m)-[:HAS_IMAGE]->(e)
                """, rows=rows)
            n = s.run("MATCH (e:EntryImage) RETURN count(e) AS n").single()["n"]
        print(f"已恢复 {len(ids)} 节点, EntryImage 总数 {n}")
        drv.close()
        return

    # 1. 扫描框 + 库清单
    recs = scan_boxes()
    print(f"FN val 框: {len(recs)} 张", flush=True)
    meta = [json.loads(l) for l in open(os.path.join(EXP, "meta.jsonl"))]
    lib_ids = [m["id"] for m in meta]
    lib_set = set(lib_ids)
    missing = [r[2] for r in recs if r[2] not in lib_set]
    print(f"库中缺失的 val 框: {len(missing)} (应为0)", flush=True)

    # 2. uuid 分组 + 每 mid 抽 1 个 uuid (优先单 mid 源图)
    uuid_mids, uuid_boxes = {}, {}
    for mid, uuid, img in recs:
        uuid_mids.setdefault(uuid, set()).add(mid)
        uuid_boxes.setdefault(uuid, []).append(img)
    mid_uuids = {}
    for mid, uuid, img in recs:
        mid_uuids.setdefault(mid, set()).add(uuid)

    rng = random.Random(SEED)
    selected = set()
    MAX_UUIDS_PER_MID = 2  # v2: 每 mid 最多抽 2 个源图 (扩框不扩类, 画廊保持非空)
    for mid in sorted(mid_uuids):
        elig = sorted(mid_uuids[mid])
        single = [u for u in elig if uuid_mids[u] == {mid}]
        pool = single if single else elig
        take = min(MAX_UUIDS_PER_MID, len(pool))
        chosen = rng.sample(pool, take)
        # 若 single 不足且 elig 更多, 从剩余 elig 补齐
        if len(chosen) < MAX_UUIDS_PER_MID and set(elig) - set(chosen):
            rest = sorted(set(elig) - set(chosen))
            chosen += rng.sample(rest, min(MAX_UUIDS_PER_MID - len(chosen), len(rest)))
        selected.update(chosen)

    def blind_ids(sel):
        return {b for u in sel for b in uuid_boxes[u]}

    # 3. 安全环: mid 库中清零 -> 退回
    mid_all_lib = {}
    for m in meta:
        mid_all_lib.setdefault(m["mid"], []).append(m["id"])
    changed = True
    while changed:
        changed = False
        bl = blind_ids(selected)
        for mid in list(mid_uuids):
            remain = [i for i in mid_all_lib.get(mid, []) if i not in bl]
            if not remain:
                back = [u for u in selected if mid in uuid_mids[u]]
                if back:
                    selected.discard(back[0])
                    changed = True

    bl = sorted(blind_ids(selected))
    n_scene = len(selected)
    n_mid = len({m for m, u, i in recs
                 if i in set(bl)})
    print(f"盲测: {n_scene} 个源图场景 / {len(bl)} 个框 / 覆盖 {n_mid} mids", flush=True)

    # 4. 备份嵌入 (可恢复)
    idx_of = {m["id"]: i for i, m in enumerate(meta)}
    mat = np.fromfile(os.path.join(EXP, "emb_ft2.f32"),
                      dtype=np.float32).reshape(-1, DIM)
    keep = [i for i, x in enumerate(lib_ids) if x in set(bl)]
    np.savez(f"{OUT_DIR}/removed_backup.npz",
             ids=np.array([lib_ids[i] for i in keep]),
             mids=np.array([meta[i]["mid"] for i in keep]),
             embs=mat[keep])
    print(f"备份嵌入: {len(keep)} 条", flush=True)

    # 5. 删除节点
    with drv.session() as s:
        n0 = s.run("MATCH (e:EntryImage) RETURN count(e) AS n").single()["n"]
        for k in range(0, len(bl), 500):
            s.run("""
                UNWIND $ids AS id
                MATCH (e:EntryImage {image_id: id}) DETACH DELETE e
            """, ids=bl[k:k + 500])
        n1 = s.run("MATCH (e:EntryImage) RETURN count(e) AS n").single()["n"]
        print(f"库: {n0} -> {n1} (删除 {n0 - n1}, 应={len(bl)})", flush=True)

        # 6. 重建向量索引 (大规模删除后防不收敛)
        s.run("DROP INDEX entryimage_embeddings IF EXISTS")
        s.run("""CREATE VECTOR INDEX entryimage_embeddings IF NOT EXISTS
                 FOR (e:EntryImage) ON (e.embedding)
                 OPTIONS {indexConfig: {`vector.dimensions`: 1024,
                                        `vector.similarity_function`: 'cosine'}}""")
        s.run("CALL db.awaitIndexes(300)")
        # 抽验: 盲测 id 已不可见
        c = s.run("MATCH (e:EntryImage) WHERE e.image_id IN $ids RETURN count(e) AS n",
                  ids=bl[:200]).single()["n"]
        print(f"抽验盲测节点可见性: {c}/200 (应=0)", flush=True)
    drv.close()

    # 7. manifest
    scenes = {}
    for u in sorted(selected):
        boxes = [{"image_id": b,
                  "mid": b.split("/boxes/")[1].split("/")[0]}
                 for b in uuid_boxes[u]]
        scenes[u] = {"boxes": boxes}
    manifest = {
        "seed": SEED,
        "stats": {"n_scenes": n_scene, "n_boxes": len(bl),
                  "n_mids": n_mid, "lib_before": n0, "lib_after": n1},
        "blind_ids": bl,
        "scenes": scenes,
    }
    json.dump(manifest, open(f"{OUT_DIR}/manifest.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"manifest -> {OUT_DIR}/manifest.json", flush=True)


if __name__ == "__main__":
    import sys
    main(restore="--restore" in sys.argv)
