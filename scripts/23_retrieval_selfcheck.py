#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_retrieval_selfcheck.py — 检索库快模式自检评测（纯服务端, 无需驱动）
原理: 抽样库内多图条目的 EntryImage, 用向量索引检索 top-K,
      看非自身前5是否命中同 msp_id 的图。
分层: FN框图 / YOLO框图 / 图集插图 / 其他, 每层最多500, 共≤2000
产出: 总体与分层 top-1/top-5 命中率, 平均 top-1 相似度
注: 查询抽样使用 rand() 随机取样 — 早期版本按存储序 LIMIT 500 会集中到
    先创建的图册条目, 曾造成 YOLO 层 2% 的抽样偏差。
"""
import subprocess
import sys
from collections import defaultdict

STRATA = [
    ("FN框图", "data/processed/fathomnet_expansion/boxes/"),
    ("YOLO框图", "data/processed/yolo_boxes/"),
    ("图集插图", "data/processed/atlas_lookup/"),
]

QUERY = """MATCH (e:EntryImage) WHERE e.image_id STARTS WITH '%s'
WITH e WHERE (size(e.image_id) > 0) AND exists((:Morphospecies {morphospecies_id: e.msp_id}))
WITH e, e.msp_id AS mid, count(*) AS c
"""

# 需要多图条目: 先算每 mid 的图数, 再在图数>=2 的 mid 中抽样
def build_cypher(prefix: str, limit: int) -> str:
    return f"""
MATCH (e:EntryImage) WHERE e.image_id STARTS WITH '{prefix}'
WITH e.msp_id AS mid, count(*) AS cnt WHERE cnt >= 2
MATCH (q:EntryImage {{msp_id: mid}}) WHERE q.image_id STARTS WITH '{prefix}'
WITH q, rand() AS r ORDER BY r LIMIT {limit}
CALL {{ WITH q CALL db.index.vector.queryNodes('entryimage_embeddings', 10, q.embedding) YIELD node, score RETURN node, score }}
WITH q, [h IN collect({{mid: node.msp_id, id: node.image_id, s: score}}) WHERE h.id <> q.image_id] AS hits
RETURN q.image_id AS qid, q.msp_id AS qmid, hits
"""


def run_cypher(cy: str) -> str:
    p = subprocess.run(
        ["docker", "exec", "-i", "-e", "LANG=C.UTF-8", "depth_vlm_neo4j",
         "cypher-shell", "-u", "neo4j", "-p", "password", "--format", "plain"],
        input=cy.encode("utf-8"), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=1800)
    return p.stdout.decode("utf-8", "replace")


def parse_row(line: str):
    """plain 格式: "qid", "qmid", [{mid: "...", id: "...", s: 0.99}, ...] — 键不带引号, 用正则解析"""
    import re
    m = re.match(r'^"([^"]+)",\s*"([^"]+)",\s*(\[.*\])\s*$', line)
    if not m:
        return None
    qid, qmid, hits_raw = m.groups()
    hits = [{"mid": a, "id": b, "s": float(c)} for a, b, c in re.findall(
        r'\{mid:\s*"([^"]*)",\s*id:\s*"([^"]*)",\s*s:\s*([0-9.eE+-]+)\}', hits_raw)]
    return qid, qmid, hits


def evaluate(name: str, prefix: str, limit: int, stats: dict):
    out = run_cypher(build_cypher(prefix, limit))
    lines = [l for l in out.splitlines()
             if l.startswith('"') and 'qid' not in l[:10]]
    n = t1 = t5 = 0
    sims = []
    for l in lines:
        r = parse_row(l)
        if not r:
            continue
        _, qmid, hits = r
        if not hits:
            continue
        n += 1
        sims.append(hits[0]["s"])
        h5 = [h for h in hits[:5]]
        if h5 and h5[0]["mid"] == qmid:
            t1 += 1
        if any(h["mid"] == qmid for h in h5):
            t5 += 1
    if n:
        stats[name] = {"n": n, "top1": t1 / n, "top5": t5 / n,
                       "sim": sum(sims) / len(sims)}
    print(f"{name}: 查询 {n} | top-1 {t1}/{n} ({t1/n*100 if n else 0:.1f}%) "
          f"| top-5 {t5}/{n} ({t5/n*100 if n else 0:.1f}%) "
          f"| 平均top-1相似度 {stats.get(name, {}).get('sim', 0):.4f}")
    return n


def main():
    print("=== 检索库快模式自检 (向量索引, 库内抽样) ===")
    stats = {}
    total = 0
    for name, prefix in STRATA:
        total += evaluate(name, prefix, 500, stats)
    if total:
        n = sum(s["n"] for s in stats.values())
        t1 = sum(s["top1"] * s["n"] for s in stats.values())
        t5 = sum(s["top5"] * s["n"] for s in stats.values())
        sim = sum(s["sim"] * s["n"] for s in stats.values())
        print(f"\n总体(加权): 查询 {n} | top-1 {t1/n*100:.1f}% | top-5 {t5/n*100:.1f}% | 平均相似度 {sim/n:.4f}")
    print("\n注: 仅统计图像数>=2的条目(单图条目无法被'其他图'命中); 检索含自身, 已排除后取前5。")


if __name__ == "__main__":
    sys.exit(main())
