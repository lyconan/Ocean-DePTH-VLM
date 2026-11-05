#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24_deep_error_analysis.py — 检索误差深度分析
对快模式自检的漏检样本追到 top-50 深处, 回答三个问题:
1) 漏检是"差一点"(正确答案在 6-50 名) 还是"彻底错"(50 名内都没有)?
2) 属级/科级命中率多少? (下游 VLM 推理只需要候选范围缩小到属)
3) 嵌入空间同类/跨类相似度分布 — 细粒度区分能力的天花板在哪
产出: 分层 top-K 命中曲线(同条目/同属/同科) + 相似度分布
"""
import subprocess
import sys
import re
import json
from collections import defaultdict

STRATA = [
    ("FN框图", "data/processed/fathomnet_expansion/boxes/"),
    ("YOLO框图", "data/processed/yolo_boxes/"),
    ("图集插图", "data/processed/atlas_lookup/"),
]
K = 50
SAMPLE = 500


def run_cypher(cy: str, timeout=3600) -> str:
    p = subprocess.run(
        ["docker", "exec", "-i", "-e", "LANG=C.UTF-8", "depth_vlm_neo4j",
         "cypher-shell", "-u", "neo4j", "-p", "password", "--format", "plain"],
        input=cy.encode("utf-8"), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout)
    return p.stdout.decode("utf-8", "replace")


def parse_hits(raw: str):
    return [{"mid": a, "id": b, "s": float(c)} for a, b, c in re.findall(
        r'\{mid:\s*"([^"]*)",\s*id:\s*"([^"]*)",\s*s:\s*([0-9.eE+-]+)\}', raw)]


def build_cypher(prefix: str, limit: int) -> str:
    return f"""
MATCH (e:EntryImage) WHERE e.image_id STARTS WITH '{prefix}'
WITH e.msp_id AS mid, count(*) AS cnt WHERE cnt >= 2
MATCH (q:EntryImage {{msp_id: mid}}) WHERE q.image_id STARTS WITH '{prefix}'
WITH q LIMIT {limit}
CALL {{ WITH q CALL db.index.vector.queryNodes('entryimage_embeddings', {K}, q.embedding) YIELD node, score RETURN node, score }}
WITH q, [h IN collect({{mid: node.msp_id, id: node.image_id, s: score}}) WHERE h.id <> q.image_id] AS hits
RETURN q.image_id AS qid, q.msp_id AS qmid, hits
"""


def fetch_taxonomy(mids):
    """分批取 morphospecies_id -> (genus, family)"""
    tax = {}
    mids = list(mids)
    for i in range(0, len(mids), 400):
        chunk = mids[i:i + 400]
        lst = "[" + ",".join(f'"{m}"' for m in chunk) + "]"
        out = run_cypher(
            f"MATCH (m:Morphospecies) WHERE m.morphospecies_id IN {lst} "
            f"RETURN m.morphospecies_id AS mid, m.genus AS g, m.family_name AS f, m.latin AS lat")
        for line in out.splitlines():
            m = re.match(r'^"([^"]+)",\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\s*$', line)
            if m:
                mid, g, f, lat = m.groups()
                tax[mid] = {"genus": g, "family": f, "latin": lat}
    return tax


def rank_stats(sims_same, ks=(1, 5, 10, 20, 50)):
    """sims_same: 每条查询的正确条目最佳排名(1-based), 无则 None"""
    n = len(sims_same)
    out = {}
    for k in ks:
        out[f"top{k}"] = sum(1 for r in sims_same if r is not None and r <= k) / n if n else 0
    out["miss50"] = sum(1 for r in sims_same if r is None) / n if n else 0
    return out


def fmt(name, n, rs, label):
    return (f"{name} {label}: n={n} | "
            + " | ".join(f"{k} {v*100:.1f}%" for k, v in rs.items()))


def analyze(name, rows, tax):
    # 同条目最佳排名
    best_rank = []
    genus_best_rank = []
    family_best_rank = []
    same_sims, inter_sims = [], []
    for qid, qmid, hits in rows:
        if not hits:
            continue
        qtax = tax.get(qmid, {})
        qgen, qfam = qtax.get("genus", ""), qtax.get("family", "")
        r_mid = r_gen = r_fam = None
        for i, h in enumerate(hits, 1):
            htax = tax.get(h["mid"], {})
            if r_mid is None and h["mid"] == qmid:
                r_mid = i
                same_sims.append(h["s"])
            if r_gen is None and qgen and htax.get("genus") == qgen and h["mid"] != qmid:
                r_gen = i
            if r_fam is None and qfam and htax.get("family") == qfam and h["mid"] != qmid:
                r_fam = i
        if r_mid is None:
            inter_sims.extend(h["s"] for h in hits)
        best_rank.append(r_mid)
        genus_best_rank.append(r_gen)
        family_best_rank.append(r_fam)

    n = len(best_rank)
    rm = rank_stats(best_rank)
    rg = rank_stats(genus_best_rank)
    rf = rank_stats(family_best_rank)
    print(f"\n== {name} (n={n}) ==")
    print(fmt("同条目", n, rm, ""))
    print(fmt("同属(非同条目)", n, rg, ""))
    print(fmt("同科(非同条目)", n, rf, ""))
    if same_sims:
        same_sims.sort()
        print(f"同类相似度: 中位 {same_sims[len(same_sims)//2]:.4f} / p25 {same_sims[len(same_sims)//4]:.4f} / n={len(same_sims)}")
    if inter_sims:
        inter_sims.sort()
        print(f"漏检查询的top命中(跨类)相似度: 中位 {inter_sims[len(inter_sims)//2]:.4f} / p25 {inter_sims[len(inter_sims)//4]:.4f} / n={len(inter_sims)}")
    return {"n": n, "mid": rm, "genus": rg, "family": rf,
            "same_sim_med": sorted(same_sims)[len(same_sims)//2] if same_sims else None,
            "inter_sim_med": sorted(inter_sims)[len(inter_sims)//2] if inter_sims else None}


def main():
    all_results = {}
    for name, prefix in STRATA:
        out = run_cypher(build_cypher(prefix, SAMPLE))
        rows = []
        mids = set()
        for line in out.splitlines():
            m = re.match(r'^"([^"]+)",\s*"([^"]+)",\s*(\[.*\])\s*$', line)
            if not m:
                continue
            qid, qmid, hits = m.group(1), m.group(2), parse_hits(m.group(3))
            rows.append((qid, qmid, hits))
            mids.add(qmid)
            mids.update(h["mid"] for h in hits)
        print(f"[{name}] 查询行 {len(rows)}, 涉及 mid {len(mids)}, 取分类信息...")
        tax = fetch_taxonomy(mids)
        all_results[name] = analyze(name, rows, tax)

    with open("/tmp/deep_error_analysis.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n已存 /tmp/deep_error_analysis.json")


if __name__ == "__main__":
    sys.exit(main())
