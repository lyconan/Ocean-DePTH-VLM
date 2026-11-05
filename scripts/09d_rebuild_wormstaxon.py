#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09d_rebuild_wormstaxon.py — WoRMSTaxon 全量重建 + Morphospecies WORMS_MATCHED (事故恢复)

背景:
  2026-09-20 Neo4j 数据目录误置 /tmp 易失挂载, docker restart 后从空库启动,
  WoRMSTaxon 263,609 + BELONGS_TO 259,100 + UnmatchedSpecies + 老 WORMS_MATCHED 全部丢失。
  本脚本从 09b 同源源数据完整重建:

    [1] 约束+索引: aphia_id 唯一约束, taxon_id/rank/phylum/family 索引
    [2] WoRMSTaxon 节点 ← data/raw/worms_inherited/apsma_worms_enhanced.jsonl
    [3] BELONGS_TO      ← data/raw/worms_inherited/apsma_worms_relationship_subset.csv
    [4] WORMS_MATCHED   ← data/processed/atlas_lookup/morphospecies_index.json
        (:Morphospecies)-[:WORMS_MATCHED {mapping_type, matched_rank}]->(:WoRMSTaxon)
        aphia_id = worms.valid_aphia_id or worms.aphia_id (与 09c Morphospecies.aphia_id 语义一致)
        注意: 图谱中 WoRMSTaxon.aphia_id 为字符串, 这里统一 str() 归一化
        (09c 原实现用 int 匹配 str 节点会静默 0 命中, 本脚本修正)
    [5] QA 审计

  幂等: 全部 MERGE, 可重复执行。

用法 (在带 neo4j 驱动且挂载 /mnt 的容器内):
  docker exec GPU_USFM_Paper python /mnt/.../Ocean-DePTH-VLM/scripts/09d_rebuild_wormstaxon.py
  可选: --skip-nodes 跳过节点/关系导入, 只补 WORMS_MATCHED
"""
import argparse
import csv
import json
import os
import sys
import time

from neo4j import GraphDatabase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL_PATH = os.path.join(ROOT, "data/raw/worms_inherited/apsma_worms_enhanced.jsonl")
REL_CSV = os.path.join(ROOT, "data/raw/worms_inherited/apsma_worms_relationship_subset.csv")
MSP_INDEX = os.path.join(ROOT, "data/processed/atlas_lookup/morphospecies_index.json")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password")

NODE_BATCH = 1000
EDGE_BATCH = 2000


def norm_record(obj: dict) -> dict:
    """JSONL 记录 → Cypher 参数 (与 09b _normalize_record 字段一致)。"""
    taxonomy = obj.get("taxonomy", {}) or {}
    features = obj.get("features", {}) or {}
    min_depth = features.get("minDepthInMeters")
    max_depth = features.get("maxDepthInMeters")
    return {
        "aphia_id": str(obj.get("aphiaID", "")),
        "taxon_id": obj.get("taxonID", ""),
        "scientific_name": obj.get("scientificName", ""),
        "rank": obj.get("rank", ""),
        "kingdom": taxonomy.get("kingdom", ""),
        "phylum": taxonomy.get("phylum", ""),
        "class": taxonomy.get("class", "") or "",
        "order_name": taxonomy.get("order", "") or "",
        "family": taxonomy.get("family", "") or "",
        "genus": taxonomy.get("genus", "") or "",
        "authority": obj.get("authority", "") or "",
        "status": obj.get("status", ""),
        "source": obj.get("source", ""),
        "min_depth": float(min_depth) if min_depth not in (None, "") else None,
        "max_depth": float(max_depth) if max_depth not in (None, "") else None,
        "marine": str(features.get("marine", "")),
        "brackish": str(features.get("brackish", "")),
    }


def run_batches(session, cypher: str, rows: list, label: str, batch_size: int, log_every: int = 20000):
    total = 0
    t0 = time.time()
    for b0 in range(0, len(rows), batch_size):
        session.run(cypher, rows=rows[b0:b0 + batch_size]).consume()
        total = min(b0 + batch_size, len(rows))
        if total % log_every == 0 or total == len(rows):
            print(f"  [{label}] {total}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nodes", action="store_true", help="跳过节点/BELONGS_TO导入, 只补 WORMS_MATCHED")
    args = ap.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        driver.verify_connectivity()
        with driver.session() as s:
            # ---- [1] 约束 + 索引 ----
            for cql in (
                "CREATE CONSTRAINT wormstaxon_aphia_id_unique IF NOT EXISTS "
                "FOR (t:WoRMSTaxon) REQUIRE t.aphia_id IS UNIQUE",
                "CREATE INDEX wormstaxon_taxon_id IF NOT EXISTS FOR (t:WoRMSTaxon) ON (t.taxon_id)",
                "CREATE INDEX wormstaxon_rank IF NOT EXISTS FOR (t:WoRMSTaxon) ON (t.rank)",
                "CREATE INDEX wormstaxon_phylum IF NOT EXISTS FOR (t:WoRMSTaxon) ON (t.phylum)",
                "CREATE INDEX wormstaxon_family IF NOT EXISTS FOR (t:WoRMSTaxon) ON (t.family)",
                "CREATE INDEX wormstaxon_sciname_phylum IF NOT EXISTS "
                "FOR (t:WoRMSTaxon) ON (t.scientific_name, t.phylum)",
            ):
                s.run(cql).consume()
            print("[1] 约束+索引就绪", flush=True)

            if not args.skip_nodes:
                # ---- [2] WoRMSTaxon 节点 ----
                t0 = time.time()
                records = []
                with open(JSONL_PATH, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(norm_record(json.loads(line)))
                print(f"[2] JSONL 解析完成: {len(records)} 条 ({time.time()-t0:.0f}s)", flush=True)
                run_batches(s, """
                    UNWIND $rows AS row
                    MERGE (t:WoRMSTaxon {aphia_id: row.aphia_id})
                    SET t.taxon_id = row.taxon_id,
                        t.scientific_name = row.scientific_name,
                        t.rank = row.rank,
                        t.kingdom = row.kingdom,
                        t.phylum = row.phylum,
                        t.class = row.class,
                        t.order_name = row.order_name,
                        t.family = row.family,
                        t.genus = row.genus,
                        t.authority = row.authority,
                        t.status = row.status,
                        t.source = row.source,
                        t.min_depth = row.min_depth,
                        t.max_depth = row.max_depth,
                        t.marine = row.marine,
                        t.brackish = row.brackish
                """, records, "WoRMSTaxon", NODE_BATCH)

                # ---- [3] BELONGS_TO ----
                edges = []
                with open(REL_CSV, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        start_id = (row.get(":START_ID") or "").strip()
                        end_id = (row.get(":END_ID") or "").strip()
                        if start_id and end_id:
                            edges.append({"start_id": start_id, "end_id": end_id})
                print(f"[3] 关系 CSV 解析完成: {len(edges)} 条", flush=True)
                run_batches(s, """
                    UNWIND $rows AS row
                    MATCH (child:WoRMSTaxon {taxon_id: row.start_id})
                    MATCH (parent:WoRMSTaxon {taxon_id: row.end_id})
                    MERGE (child)-[:BELONGS_TO]->(parent)
                """, edges, "BELONGS_TO", EDGE_BATCH)

            # ---- [4] WORMS_MATCHED ----
            idx = json.load(open(MSP_INDEX, encoding="utf-8"))
            msp = idx["morphospecies"]
            msp = list(msp.values()) if isinstance(msp, dict) else msp
            link_rows, no_aphia = [], 0
            for m in msp:
                w = m.get("worms") or {}
                aid = w.get("valid_aphia_id") or w.get("aphia_id")
                if aid:
                    link_rows.append({"msp_id": m["morphospecies_id"], "aphia_id": str(aid)})
                else:
                    no_aphia += 1
            print(f"[4] 待建 WORMS_MATCHED: {len(link_rows)} 条 (无aphia_id条目 {no_aphia})", flush=True)
            run_batches(s, """
                UNWIND $rows AS row
                MATCH (ms:Morphospecies {morphospecies_id: row.msp_id})
                MATCH (wt:WoRMSTaxon {aphia_id: row.aphia_id})
                MERGE (ms)-[r:WORMS_MATCHED]->(wt)
                SET r.mapping_type = 'WORMS_EXACT', r.matched_rank = 'accepted'
            """, link_rows, "WORMS_MATCHED", 500, log_every=1000)

            # ---- [5] QA ----
            print("=== QA ===", flush=True)
            for label, cql in (
                ("WoRMSTaxon", "MATCH (t:WoRMSTaxon) RETURN count(t) AS c"),
                ("BELONGS_TO", "MATCH (:WoRMSTaxon)-[:BELONGS_TO]->() RETURN count(*) AS c"),
                ("WORMS_MATCHED", "MATCH (:Morphospecies)-[:WORMS_MATCHED]->() RETURN count(*) AS c"),
                ("有aphia但未匹配", "MATCH (ms:Morphospecies) WHERE ms.aphia_id IS NOT NULL "
                                   "AND NOT (ms)-[:WORMS_MATCHED]->() RETURN count(ms) AS c"),
                ("Morphospecies", "MATCH (ms:Morphospecies) RETURN count(ms) AS c"),
                ("EntryImage", "MATCH (ei:EntryImage) RETURN count(ei) AS c"),
            ):
                rec = s.run(cql).single()
                print(f"  {label}: {rec['c']}", flush=True)
            for r in s.run("SHOW INDEXES YIELD name, type, state WHERE type='VECTOR' "
                           "RETURN name, state"):
                print(f"  向量索引: {r['name']} ({r['state']})", flush=True)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
