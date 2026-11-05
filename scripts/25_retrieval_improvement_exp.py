#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_retrieval_improvement_exp.py — 检索能力提升实验 (宿主编排端)
两阶段:
  A. 宿主: 从 Neo4j 流式导出全库 EntryImage 嵌入 -> /tmp/exp/emb.f32 + meta.jsonl
     并抽样查询集 (FN/YOLO/图集, 每层多图条目, 文件须在盘上) -> queries.json
  B. 容器 (yolo34_train:v1, GPU0): 运行 scripts/25b_exp_container_runner.py
     对比 查询侧预处理变体 x (原图NN检索 / 条目中心检索) 的 top-1/top-5
"""
import subprocess
import sys
import os
import re
import json
import random
import struct
from collections import defaultdict

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "/tmp/exp"
DIM = 1024
STRATA = {
    "FN框图": "data/processed/fathomnet_expansion/boxes/",
    "YOLO框图": "data/processed/yolo_boxes/",
    "图集插图": "data/processed/atlas_lookup/",
}
SAMPLE_PER_STRATUM = 400


def phase_a():
    os.makedirs(EXP, exist_ok=True)
    f_bin = open(f"{EXP}/emb.f32", "wb")
    f_meta = open(f"{EXP}/meta.jsonl", "w")
    cmd = ["docker", "exec", "-i", "-e", "LANG=C.UTF-8", "depth_vlm_neo4j",
           "cypher-shell", "-u", "neo4j", "-p", "password", "--format", "plain"]
    cy = "MATCH (e:EntryImage) WHERE e.embedding IS NOT NULL " \
         "RETURN e.image_id AS iid, e.msp_id AS mid, e.embedding AS emb"
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    p.stdin.write(cy.encode())
    p.stdin.close()
    n = 0
    row_re = re.compile(r'^"([^"]+)",\s*"([^"]+)",\s*\[(.*)\]\s*$')
    for raw in p.stdout:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        m = row_re.match(line)
        if not m:
            continue
        iid, mid, arr = m.groups()
        try:
            floats = [float(x) for x in arr.split(",")]
        except ValueError:
            continue
        if len(floats) != DIM:
            print(f"!! 维度异常 {iid}: {len(floats)}", flush=True)
            continue
        f_bin.write(struct.pack(f"<{DIM}f", *floats))
        f_meta.write(json.dumps({"i": n, "id": iid, "mid": mid}) + "\n")
        n += 1
        if n % 5000 == 0:
            print(f"  dumped {n}", flush=True)
    p.wait()
    f_bin.close()
    f_meta.close()
    print(f"导出完成: {n} 条", flush=True)

    # 抽样查询集
    by_strat_mid = defaultdict(lambda: defaultdict(list))
    with open(f"{EXP}/meta.jsonl") as f:
        for line in f:
            r = json.loads(line)
            for name, prefix in STRATA.items():
                if r["id"].startswith(prefix):
                    by_strat_mid[name][r["mid"]].append(r["id"])
    random.seed(42)
    queries = []
    for name, mids in by_strat_mid.items():
        # 只取多图条目(同层内 >=2 张)且文件在盘上
        cand = []
        for mid, ids in mids.items():
            ok = [i for i in ids if os.path.exists(os.path.join(WS, i))]
            if len(ok) >= 2:
                cand.extend((mid, i) for i in ok)
        random.shuffle(cand)
        cand = cand[:SAMPLE_PER_STRATUM]
        queries.extend({"stratum": name, "qid": i, "mid": mid} for mid, i in cand)
        print(f"[{name}] 候选 {sum(len(v) for v in mids.values())} 张, 采样 {len(cand)}", flush=True)
    with open(f"{EXP}/queries.json", "w") as f:
        json.dump(queries, f, ensure_ascii=False, indent=1)
    print(f"查询集 {len(queries)} 条 -> {EXP}/queries.json", flush=True)


def phase_b():
    script = os.path.join(WS, "scripts/25b_exp_container_runner.py")
    cmd = ["docker", "run", "--rm", "--gpus", '"device=0"', "--shm-size", "8g",
           "-v", f"{EXP}:/exp",
           "-v", f"{WS}:/work:ro",
           "-v", "/root/.cache/huggingface:/root/.cache/huggingface:ro",
           "yolo34_train:v1", "python", "/work/scripts/25b_exp_container_runner.py"]
    print("启动容器实验...", flush=True)
    r = subprocess.run(cmd)
    sys.exit(r.returncode)


if __name__ == "__main__":
    if "--container" not in sys.argv:
        if os.path.exists(f"{EXP}/queries.json"):
            print("已存在导出结果, 跳过 phase A", flush=True)
        else:
            phase_a()
        phase_b()
    # 容器内逻辑全部在 25b
