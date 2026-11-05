#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09f_backfill_lineage.py — 回填 Morphospecies 纲/目/科属性 (修复 09c 键名不匹配 bug)

Bug: 09c 建图时按 m.lineage['class_name']/'order_name'/'family_name' 取值,
     但 index 实际键为 lineage['class']/['order']/['family'], 导致图谱中
     class_name/order_name/family_name 全为空 (0/2641)。

回填优先级: lineage.{class,order,family} -> worms.chain_direct.{class,order,family}
幂等可重跑。用法:
  docker exec -e PYTHONUNBUFFERED=1 GPU_USFM_Paper python scripts/09f_backfill_lineage.py
"""
import json
import os
import time

from neo4j import GraphDatabase

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password")


def norm_rank(v):
    """chain_direct 值可能是 {name:..} 或裸字符串。"""
    if isinstance(v, dict):
        return v.get("name") or ""
    return v or ""


def main():
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp

    rows = []
    for m in msp:
        lin = m.get("lineage") or {}
        cd = (m.get("worms") or {}).get("chain_direct") or {}
        tw = m.get("taxonomy_worms") or {}

        def pick(prop, lin_key, cd_key, tw_key):
            v = lin.get(lin_key) or norm_rank(cd.get(cd_key))
            if not v and tw_key:
                v = norm_rank((tw.get(tw_key) or {}).get("name"))
            return v or ""

        rows.append({
            "mid": m["morphospecies_id"],
            "class_name": pick("class_name", "class", "class", "class"),
            "order_name": pick("order_name", "order", "order", "order"),
            "family_name": pick("family_name", "family", "family", "family"),
            "genus": lin.get("genus") or m.get("genus") or norm_rank(cd.get("genus")) or norm_rank((tw.get("genus") or {}).get("name")) or "",
        })
    n_cls = sum(1 for r in rows if r["class_name"])
    n_ord = sum(1 for r in rows if r["order_name"])
    n_fam = sum(1 for r in rows if r["family_name"])
    print(f"回填可用: class {n_cls}, order {n_ord}, family {n_fam} / {len(rows)}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        driver.verify_connectivity()
        t0 = time.time()
        with driver.session() as s:
            for b0 in range(0, len(rows), 500):
                s.run("""
                    UNWIND $rows AS row
                    MATCH (ms:Morphospecies {morphospecies_id: row.mid})
                    SET ms.class_name = row.class_name,
                        ms.order_name = row.order_name,
                        ms.family_name = row.family_name,
                        ms.genus = row.genus
                """, rows=rows[b0:b0 + 500]).consume()
        print(f"回填完成 ({time.time()-t0:.1f}s)")
        with driver.session() as s:
            for label, cql in (
                ("class_name", "MATCH (ms:Morphospecies) WHERE ms.class_name <> '' RETURN count(ms) AS c"),
                ("order_name", "MATCH (ms:Morphospecies) WHERE ms.order_name <> '' RETURN count(ms) AS c"),
                ("family_name", "MATCH (ms:Morphospecies) WHERE ms.family_name <> '' RETURN count(ms) AS c"),
                ("genus", "MATCH (ms:Morphospecies) WHERE ms.genus <> '' RETURN count(ms) AS c"),
            ):
                print(f"  {label} 非空: {s.run(cql).single()['c']}/{len(rows)}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
