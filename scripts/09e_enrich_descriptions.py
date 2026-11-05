#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09e_enrich_descriptions.py — 为 Morphospecies 挂接描述文本 (节点属性方案, 只挂确定种)

策略 (2026-09-21 用户确认):
  1. 图集原文描述: morphospecies_index.morphotype_ids → apsma_knowledge_validated.jsonl
     features.description (条目自己的文献描述, 不存在张冠李戴, 有则直接挂)
  2. 物种介绍: knowledge_base descriptions/ 的 markdown 文件按拉丁名精确命中
     (只挂确定种; 属级文件/同属代表描述一律不挂, 宁缺毋滥)
  3. 图集出处: index.entries 的 source 引用 → ms.atlas_source (全部条目)

节点新增属性:
  description  主描述文本 (atlas 优先, 其次 knowledge_base)
  desc_source  'atlas' / 'knowledge_base'
  atlas_source 图集出处 (逗号连接)
  kb_path      knowledge_base md 全文路径 (desc_source=knowledge_base 时)

幂等可重跑。用法:
  docker exec GPU_USFM_Paper python /mnt/.../scripts/09e_enrich_descriptions.py
"""
import json
import os
import sys
import time

from neo4j import GraphDatabase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSP_JSON = os.path.join(ROOT, "data/processed/atlas_lookup/morphospecies_index.json")
APSMA_JSONL = os.path.join(ROOT, "data/processed/apsma_parsed/apsma_knowledge_validated.jsonl")
KB_DIR = os.path.join(ROOT, "data/processed/species_knowledge_base/descriptions")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password")


def build_name_index():
    """知识库描述文件: stem(下划线/空格归一) → (path, snippet)。"""
    idx = {}
    for dirpath, _, files in os.walk(KB_DIR):
        for f in files:
            if not f.endswith(".md"):
                continue
            stem = os.path.splitext(f)[0]
            key = stem.replace("_", " ").strip()
            p = os.path.join(dirpath, f)
            snippet = ""
            try:
                with open(p, encoding="utf-8") as fh:
                    lines = [l.strip() for l in fh.read().splitlines()
                             if l.strip() and not l.strip().startswith(("#", "![", "!["))]
                    snippet = " ".join(lines)[:600]
            except OSError:
                continue
            idx[key] = (p, snippet)
    return idx


def main():
    # ---- 数据准备 ----
    atlas_desc = {}
    with open(APSMA_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = (r.get("features") or {}).get("description") or ""
            if d.strip():
                # morphotype_id (APSMA_V1_...) 与 original_morphotype_id (ACT_001 风格)
                # 双键注册 — 新图谱 morphospecies_index.morphotype_ids 用后者
                for key in (r.get("morphotype_id"), r.get("original_morphotype_id")):
                    if key:
                        atlas_desc[key] = d.strip()
    print(f"图集原文描述: {len(atlas_desc)} 条 (morphotype 级)")

    kb = build_name_index()
    print(f"知识库描述文件: {len(kb)} 个")

    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp

    rows = []
    n_atlas = n_kb = n_src_only = 0
    for m in msp:
        mid = m["morphospecies_id"]
        # 1) 图集出处 (全部条目)
        atlas_source = ",".join(sorted({e.get("source", "") for e in m.get("entries", [])
                                        if isinstance(e, dict) and e.get("source")}))
        # 2) 图集原文描述 (morphotype_ids join)
        desc, src, kb_path = "", "", ""
        for mt in (m.get("morphotype_ids") or []):
            if mt in atlas_desc:
                desc, src = atlas_desc[mt], "atlas"
                break
        # 3) 物种介绍 (拉丁名精确命中, 仅确定种)
        if not desc:
            lat = (m.get("latin") or "").strip()
            if lat and lat in kb:  # 完整科学名命中 = 确定种
                kb_path, desc = kb[lat]
                src = "knowledge_base"
        if desc and src == "atlas":
            n_atlas += 1
        elif desc and src == "knowledge_base":
            n_kb += 1
        elif atlas_source:
            n_src_only += 1
        rows.append({"mid": mid, "desc": desc, "src": src,
                     "atlas_source": atlas_source, "kb_path": kb_path})

    print(f"匹配结果: 图集原文 {n_atlas} | 物种介绍 {n_kb} | 仅出处 {n_src_only} | "
          f"完全无 {len(rows) - n_atlas - n_kb - n_src_only}")

    # ---- 写入 Neo4j ----
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        driver.verify_connectivity()
        t0 = time.time()
        with driver.session() as s:
            for b0 in range(0, len(rows), 500):
                s.run("""
                    UNWIND $rows AS row
                    MATCH (ms:Morphospecies {morphospecies_id: row.mid})
                    SET ms.description = row.desc,
                        ms.desc_source = CASE WHEN row.desc = '' THEN '' ELSE row.src END,
                        ms.atlas_source = row.atlas_source,
                        ms.kb_path = row.kb_path
                """, rows=rows[b0:b0 + 500]).consume()
        print(f"写入完成 ({time.time()-t0:.1f}s)")

        # ---- QA ----
        with driver.session() as s:
            for label, cql in (
                ("description非空", "MATCH (ms:Morphospecies) WHERE ms.description <> '' RETURN count(ms) AS c"),
                ("  其中atlas", "MATCH (ms:Morphospecies) WHERE ms.desc_source='atlas' RETURN count(ms) AS c"),
                ("  其中knowledge_base", "MATCH (ms:Morphospecies) WHERE ms.desc_source='knowledge_base' RETURN count(ms) AS c"),
                ("atlas_source非空", "MATCH (ms:Morphospecies) WHERE ms.atlas_source <> '' RETURN count(ms) AS c"),
            ):
                print(f"  {label}: {s.run(cql).single()['c']}")
            sample = s.run("MATCH (ms:Morphospecies) WHERE ms.description <> '' "
                           "RETURN ms.morphospecies_id AS id, ms.desc_source AS src, "
                           "left(ms.description, 120) AS d LIMIT 3").data()
            for r in sample:
                print(f"  样例 [{r['id']}/{r['src']}]: {r['d']}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
