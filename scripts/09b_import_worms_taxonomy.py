"""WoRMS 分类树 Neo4j 导入 — 263K 节点 + 259K 关系 + 图片/描述关联 + APSMA 对齐 (v1.0)

将 opc-52 继承的 WoRMS 21 门分类子集导入 Neo4j，与现有 APSMA 形态型 KG 共存。

数据模型:
  WoRMSTaxon  — 统一标签，rank 属性区分层级
  BELONGS_TO  — WoRMS 分类层级关系
  WORMS_MATCHED — APSMA Morphotype → WoRMSTaxon 对齐关系

执行顺序:
  1a. 预创建索引 (aphia_id 唯一约束 + taxon_id 索引)
  1b. 导入 WoRMSTaxon 节点 (263,609)
  2.  导入 BELONGS_TO 关系 (259,100)
  3.  关联物种图片信息 (has_images, image_count, image_paths_json)
  4.  关联物种描述信息 (has_description, description_path)
  5.  导入双名法未匹配物种 (UnmatchedSpecies 标签, ~4833)
  6.  创建 APSMA Morphotype → WoRMSTaxon 对齐关系 (WORMS_MATCHED)
  7.  验证 + 统计报告

用法:
  python scripts/09b_import_worms_taxonomy.py
  python scripts/09b_import_worms_taxonomy.py --skip-nodes  # 跳过节点导入(仅更新属性)
  python scripts/09b_import_worms_taxonomy.py --skip-relationships
"""
import os
import csv
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

from neo4j import GraphDatabase


def get_neo4j_driver(uri="bolt://localhost:7687", user="neo4j", password="password"):
    return GraphDatabase.driver(uri, auth=(user, password))


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_worms_data(jsonl_path: str) -> List[dict]:
    """加载 WoRMS enhanced JSONL，返回标准化记录列表。"""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append(_normalize_record(obj))
    return records


def _normalize_record(obj: dict) -> dict:
    """将 JSONL 记录转为 Cypher 参数格式。"""
    taxonomy = obj.get("taxonomy", {})
    features = obj.get("features", {})

    min_depth = features.get("minDepthInMeters")
    max_depth = features.get("maxDepthInMeters")

    return {
        "aphia_id":        str(obj.get("aphiaID", "")),
        "taxon_id":        obj.get("taxonID", ""),
        "scientific_name": obj.get("scientificName", ""),
        "rank":            obj.get("rank", ""),
        "kingdom":         taxonomy.get("kingdom", ""),
        "phylum":          taxonomy.get("phylum", ""),
        "class":           taxonomy.get("class", "") or "",
        "order_name":      taxonomy.get("order", "") or "",
        "family":          taxonomy.get("family", "") or "",
        "genus":           taxonomy.get("genus", "") or "",
        "authority":       obj.get("authority", "") or "",
        "status":          obj.get("status", ""),
        "source":          obj.get("source", ""),
        "min_depth":       float(min_depth) if min_depth not in (None, "") else None,
        "max_depth":       float(max_depth) if max_depth not in (None, "") else None,
        "marine":          str(features.get("marine", "")),
        "brackish":        str(features.get("brackish", "")),
    }


def load_image_index(index_path: str) -> Dict[str, dict]:
    """加载图片索引，构建 'scientificName' → info 映射。"""
    with open(index_path) as f:
        data = json.load(f)

    name_map = {}
    for key, paths in data.get("worms_species", {}).items():
        parts = key.split("/")
        if len(parts) == 2:
            phylum = parts[0]
            # 图片库用 _ 分隔，WoRMS 用空格
            scientific_name = parts[1].replace("_", " ")
            name_map[scientific_name] = {
                "phylum": phylum,
                "image_count": len(paths),
                "image_paths_json": json.dumps(paths),
            }
    return name_map


def load_description_index(index_path: str) -> Dict[str, dict]:
    """加载描述索引，构建 'scientificName' → info 映射。"""
    with open(index_path) as f:
        data = json.load(f)

    name_map = {}
    for key, info in data.get("descriptions", {}).items():
        parts = key.split("/")
        if len(parts) == 2:
            phylum = parts[0]
            # 描述库已经是空格分隔
            scientific_name = parts[1]
            name_map[scientific_name] = {
                "phylum": phylum,
                "description_path": info.get("path", ""),
            }
    return name_map


def load_alignment_csv(csv_path: str) -> List[dict]:
    """加载 APSMA-WoRMS 对齐矩阵。"""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aphia_id = row.get("worms_aphia_id", "")
            if aphia_id:
                rows.append({
                    "apsma_morphotype_id": row["apsma_morphotype_id"],
                    "worms_aphia_id": str(aphia_id),
                    "mapping_type": row.get("mapping_type", ""),
                    "matched_rank": row.get("matched_rank", ""),
                })
    return rows


# ═══════════════════════════════════════════════════════════════
# 批量执行工具
# ═══════════════════════════════════════════════════════════════

def _execute_batch(driver, query, batch, label=""):
    """执行单批次事务，失败时重试一次。"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            with driver.session() as session:
                session.run(query, batch=batch)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [{label}] 批次失败 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
            else:
                print(f"  [{label}] 批次最终失败: {e}")
                raise
    return False


# ═══════════════════════════════════════════════════════════════
# 步骤 1a: 预创建索引
# ═══════════════════════════════════════════════════════════════

def pre_create_indexes(driver):
    """创建导入所需的关键索引（在节点/关系导入前运行）。"""
    with driver.session() as session:
        # 唯一性约束 — aphia_id 是 WoRMSTaxon 的主键
        session.run(
            "CREATE CONSTRAINT wormstaxon_aphia_id_unique IF NOT EXISTS "
            "FOR (t:WoRMSTaxon) REQUIRE t.aphia_id IS UNIQUE"
        )
        # B-tree 索引 — 关系导入步骤需要 taxon_id MATCH
        session.run(
            "CREATE INDEX wormstaxon_taxon_id_idx IF NOT EXISTS "
            "FOR (t:WoRMSTaxon) ON (t.taxon_id)"
        )
    print("[1a] 预创建索引完成: aphia_id 唯一约束 + taxon_id 索引")


# ═══════════════════════════════════════════════════════════════
# 步骤 1b: 导入 WoRMSTaxon 节点
# ═══════════════════════════════════════════════════════════════

def import_worms_nodes_batch(driver, records, batch_size=2000):
    """批量导入 WoRMSTaxon 节点。"""
    cypher = """
    UNWIND $batch AS row
    MERGE (t:WoRMSTaxon {aphia_id: row.aphia_id})
    ON CREATE SET
        t.taxon_id         = row.taxon_id,
        t.scientific_name  = row.scientific_name,
        t.rank             = row.rank,
        t.kingdom          = row.kingdom,
        t.phylum           = row.phylum,
        t.class            = row.class,
        t.order_name       = row.order_name,
        t.family           = row.family,
        t.genus            = row.genus,
        t.authority        = row.authority,
        t.status           = row.status,
        t.source           = row.source,
        t.min_depth        = row.min_depth,
        t.max_depth        = row.max_depth,
        t.marine           = row.marine,
        t.brackish         = row.brackish,
        t.has_images       = false,
        t.image_count      = 0,
        t.has_description  = false
    """

    total = len(records)
    imported = 0
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        _execute_batch(driver, cypher, batch, label="节点导入")
        imported += len(batch)
        if imported % 20000 == 0 or imported == total:
            print(f"  [1b] 节点导入: {imported}/{total} ({imported/total*100:.1f}%)")

    print(f"[1b] WoRMSTaxon 节点导入完成: {total} 个")


# ═══════════════════════════════════════════════════════════════
# 步骤 2: 创建中间索引 + 导入 BELONGS_TO 关系
# ═══════════════════════════════════════════════════════════════

def create_intermediate_indexes(driver):
    """创建图片/描述关联所需的复合索引。"""
    with driver.session() as session:
        session.run(
            "CREATE INDEX wormstaxon_name_phylum_idx IF NOT EXISTS "
            "FOR (t:WoRMSTaxon) ON (t.scientific_name, t.phylum)"
        )
    print("[2a] 中间索引创建完成: scientific_name+phylum 复合索引")


def import_worms_relationships_batch(driver, csv_path, batch_size=5000):
    """从 CSV 批量导入 BELONGS_TO 关系。"""
    cypher = """
    UNWIND $batch AS row
    MATCH (child:WoRMSTaxon {taxon_id: row.start_id})
    MATCH (parent:WoRMSTaxon {taxon_id: row.end_id})
    MERGE (child)-[:BELONGS_TO]->(parent)
    """

    batch = []
    total = 0
    skipped = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = row.get(":START_ID", "").strip()
            end_id = row.get(":END_ID", "").strip()
            if not start_id or not end_id:
                skipped += 1
                continue
            batch.append({
                "start_id": start_id,
                "end_id": end_id,
            })
            if len(batch) >= batch_size:
                _execute_batch(driver, cypher, batch, label="关系导入")
                total += len(batch)
                batch = []
                if total % 50000 == 0:
                    print(f"  [2b] BELONGS_TO: {total} 条")
        if batch:
            _execute_batch(driver, cypher, batch, label="关系导入")
            total += len(batch)

    print(f"[2b] BELONGS_TO 关系导入完成: {total} 条" +
          (f" (跳过 {skipped} 条空ID)" if skipped else ""))


# ═══════════════════════════════════════════════════════════════
# 步骤 3: 关联物种图片信息
# ═══════════════════════════════════════════════════════════════

def enrich_worms_nodes_with_images(driver, image_map, batch_size=2000):
    """为匹配到的 WoRMSTaxon 节点关联图片信息。"""
    cypher = """
    UNWIND $batch AS row
    MATCH (t:WoRMSTaxon {scientific_name: row.name, phylum: row.phylum})
    SET t.has_images      = true,
        t.image_count     = row.image_count,
        t.image_paths_json = row.image_paths_json
    """

    # 构建 Cypher 批量参数
    batch = []
    for name, info in image_map.items():
        batch.append({
            "name": name,
            "phylum": info["phylum"],
            "image_count": info["image_count"],
            "image_paths_json": info["image_paths_json"],
        })

    total = len(batch)
    matched = 0
    for i in range(0, total, batch_size):
        sub_batch = batch[i:i + batch_size]
        with driver.session() as session:
            result = session.run(cypher, batch=sub_batch)
            # 统计实际匹配数
            summary = result.consume()
            matched += summary.counters.properties_set
        print(f"  [3] 图片关联: {min(i+batch_size, total)}/{total}")

    print(f"[3] 图片关联完成: {total} 个物种名, {matched} 个属性更新")


# ═══════════════════════════════════════════════════════════════
# 步骤 4: 关联物种描述信息
# ═══════════════════════════════════════════════════════════════

def enrich_worms_nodes_with_descriptions(driver, desc_map, batch_size=2000):
    """为匹配到的 WoRMSTaxon 节点关联描述信息。"""
    cypher = """
    UNWIND $batch AS row
    MATCH (t:WoRMSTaxon {scientific_name: row.name, phylum: row.phylum})
    SET t.has_description  = true,
        t.description_path = row.description_path
    """

    batch = []
    for name, info in desc_map.items():
        batch.append({
            "name": name,
            "phylum": info["phylum"],
            "description_path": info["description_path"],
        })

    total = len(batch)
    for i in range(0, total, batch_size):
        sub_batch = batch[i:i + batch_size]
        with driver.session() as session:
            session.run(cypher, batch=sub_batch)
        print(f"  [4] 描述关联: {min(i+batch_size, total)}/{total}")

    print(f"[4] 描述关联完成: {total} 个物种名")


# ═══════════════════════════════════════════════════════════════
# 步骤 5: 导入双名法未匹配物种 (UnmatchedSpecies)
# ═══════════════════════════════════════════════════════════════

def import_unmatched_species(driver, image_map, desc_map, worms_names_set, batch_size=2000):
    """为图片库中无法匹配到 WoRMS 的双名法物种创建独立节点。

    这些物种有图片和/或描述，但不在 WoRMS 21 门子集中。
    使用 UnmatchedSpecies 标签，不创建 BELONGS_TO 关系。
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (s:UnmatchedSpecies {scientific_name: row.scientific_name, phylum: row.phylum})
    ON CREATE SET
        s.has_images      = row.has_images,
        s.image_count     = row.image_count,
        s.image_paths_json = row.image_paths_json,
        s.has_description = row.has_description,
        s.description_path = row.description_path
    """

    batch = []
    # 合并图片库和描述库中未匹配的物种
    all_keys = set()
    for name in image_map:
        if name not in worms_names_set:
            all_keys.add(name)
    for name in desc_map:
        if name not in worms_names_set:
            all_keys.add(name)

    for name in all_keys:
        # 仅导入双名法（属+种，至少两个词且种加词小写开头）
        words = name.split()
        if len(words) < 2:
            continue
        # 检查第二个词首字母是否小写（种加词特征）
        # 但有些属名+亚属名也是双词，宽容处理：只要有两个词就导入
        img_info = image_map.get(name, {})
        desc_info = desc_map.get(name, {})

        phylum = img_info.get("phylum") or desc_info.get("phylum", "")
        if not phylum:
            continue

        batch.append({
            "scientific_name": name,
            "phylum": phylum,
            "has_images": name in image_map,
            "image_count": img_info.get("image_count", 0),
            "image_paths_json": img_info.get("image_paths_json", "[]"),
            "has_description": name in desc_map,
            "description_path": desc_info.get("description_path", ""),
        })

    total = len(batch)
    if total == 0:
        print("[5] 无双名法未匹配物种需要导入")
        return

    imported = 0
    for i in range(0, total, batch_size):
        sub_batch = batch[i:i + batch_size]
        _execute_batch(driver, cypher, sub_batch, label="未匹配物种")
        imported += len(sub_batch)
        if imported % 5000 == 0 or imported == total:
            print(f"  [5] 未匹配物种: {imported}/{total}")

    print(f"[5] 未匹配物种导入完成: {total} 个 (UnmatchedSpecies 标签)")


# ═══════════════════════════════════════════════════════════════
# 步骤 6: 创建 APSMA-WoRMS 对齐关系
# ═══════════════════════════════════════════════════════════════

def create_worms_alignment_relations(driver, alignment_data):
    """创建 APSMA Morphotype → WoRMSTaxon 的 WORMS_MATCHED 对齐关系。"""
    cypher = """
    UNWIND $batch AS row
    MATCH (m:Morphotype {morphotype_id: row.apsma_morphotype_id})
    MATCH (t:WoRMSTaxon {aphia_id: row.worms_aphia_id})
    MERGE (m)-[r:WORMS_MATCHED]->(t)
    ON CREATE SET
        r.mapping_type     = row.mapping_type,
        r.matched_rank     = row.matched_rank,
        r.alignment_source = 'apsma_alignment_csv'
    """

    if not alignment_data:
        print("[6] 无对齐数据，跳过")
        return

    created = 0
    failed = 0
    with driver.session() as session:
        result = session.run(cypher, batch=alignment_data)
        summary = result.consume()
        created = summary.counters.relationships_created

    print(f"[6] APSMA-WoRMS 对齐关系创建: {created} 条 WORMS_MATCHED")


# ═══════════════════════════════════════════════════════════════
# 步骤 7: 创建剩余索引
# ═══════════════════════════════════════════════════════════════

def create_post_import_indexes(driver):
    """导入完成后创建其余索引（导入期间不建以避免拖慢速度）。"""
    with driver.session() as session:
        # rank 过滤
        session.run(
            "CREATE INDEX wormstaxon_rank_idx IF NOT EXISTS "
            "FOR (t:WoRMSTaxon) ON (t.rank)"
        )
        # 门级过滤
        session.run(
            "CREATE INDEX wormstaxon_phylum_idx IF NOT EXISTS "
            "FOR (t:WoRMSTaxon) ON (t.phylum)"
        )
        # 科级过滤
        session.run(
            "CREATE INDEX wormstaxon_family_idx IF NOT EXISTS "
            "FOR (t:WoRMSTaxon) ON (t.family)"
        )
        # 图片标记
        session.run(
            "CREATE INDEX wormstaxon_has_images_idx IF NOT EXISTS "
            "FOR (t:WoRMSTaxon) ON (t.has_images)"
        )
        # 全文索引
        session.run(
            "CREATE FULLTEXT INDEX wormstaxon_text_index IF NOT EXISTS "
            "FOR (t:WoRMSTaxon) ON EACH [t.scientific_name, t.authority]"
        )
        # UnmatchedSpecies 索引
        session.run(
            "CREATE INDEX unmatched_phylum_idx IF NOT EXISTS "
            "FOR (s:UnmatchedSpecies) ON (s.phylum)"
        )

    print("[7] 剩余索引创建完成")


# ═══════════════════════════════════════════════════════════════
# 步骤 8: 验证 + 统计报告
# ═══════════════════════════════════════════════════════════════

def verify_import(driver) -> dict:
    """导入后验证 + 统计报告。"""
    report = {}

    with driver.session() as session:
        # 总节点数
        total_nodes = session.run(
            "MATCH (t:WoRMSTaxon) RETURN count(t) AS cnt"
        ).single()["cnt"]
        report["total_worms_taxa"] = total_nodes

        # 按 rank 分布
        rank_stats = session.run("""
            MATCH (t:WoRMSTaxon)
            RETURN t.rank AS rank, count(t) AS cnt
            ORDER BY cnt DESC
        """)
        report["rank_distribution"] = [dict(r) for r in rank_stats]

        # BELONGS_TO 关系数
        rel_count = session.run(
            "MATCH (:WoRMSTaxon)-[r:BELONGS_TO]->(:WoRMSTaxon) RETURN count(r) AS cnt"
        ).single()["cnt"]
        report["belongs_to_count"] = rel_count

        # WORMS_MATCHED 关系数
        align_count = session.run(
            "MATCH (:Morphotype)-[r:WORMS_MATCHED]->(:WoRMSTaxon) RETURN count(r) AS cnt"
        ).single()["cnt"]
        report["worms_matched_count"] = align_count

        # 有图片/描述的节点
        media_stats = session.run("""
            MATCH (t:WoRMSTaxon)
            RETURN count(t) AS total,
                   sum(CASE WHEN t.has_images = true THEN 1 ELSE 0 END) AS with_images,
                   sum(CASE WHEN t.has_description = true THEN 1 ELSE 0 END) AS with_desc,
                   sum(CASE WHEN t.has_images = true AND t.has_description = true THEN 1 ELSE 0 END) AS with_both
        """)
        report["media_stats"] = dict(media_stats.single())

        # Phylum 数量
        phylum_count = session.run(
            "MATCH (t:WoRMSTaxon {rank: 'Phylum'}) RETURN count(t) AS cnt"
        ).single()["cnt"]
        report["phylum_count"] = phylum_count

        # 孤立节点（无 BELONGS_TO 入边也无出边，且非 Phylum 顶级）
        isolated = session.run("""
            MATCH (t:WoRMSTaxon)
            WHERE NOT (t)-[:BELONGS_TO]->(:WoRMSTaxon)
              AND NOT (:WoRMSTaxon)-[:BELONGS_TO]->(t)
              AND t.rank <> 'Phylum'
            RETURN count(t) AS cnt
        """).single()["cnt"]
        report["isolated_worms_taxa"] = isolated

        # UnmatchedSpecies 数量
        try:
            unmatched_count = session.run(
                "MATCH (s:UnmatchedSpecies) RETURN count(s) AS cnt"
            ).single()["cnt"]
            report["unmatched_species_count"] = unmatched_count
        except Exception:
            report["unmatched_species_count"] = 0

    # 打印报告
    print("\n" + "=" * 60)
    print("WoRMS 导入验证报告")
    print("=" * 60)
    print(f"  WoRMSTaxon 总节点数: {report['total_worms_taxa']:,}")
    print(f"  BELONGS_TO 关系数:   {report['belongs_to_count']:,}")
    print(f"  WORMS_MATCHED 关系数: {report['worms_matched_count']}")
    print(f"  Phylum 数量:         {report['phylum_count']}")
    print(f"  孤立节点:            {report['isolated_worms_taxa']}")
    print(f"  UnmatchedSpecies:    {report['unmatched_species_count']}")

    ms = report["media_stats"]
    print(f"\n  媒体统计:")
    print(f"    有图片:   {ms['with_images']:,} / {ms['total']:,}")
    print(f"    有描述:   {ms['with_desc']:,} / {ms['total']:,}")
    print(f"    图片+描述: {ms['with_both']:,}")

    print(f"\n  Rank 分布 (Top 10):")
    for r in report["rank_distribution"][:10]:
        print(f"    {r['rank']}: {r['cnt']:,}")

    print("=" * 60)
    return report


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="导入 WoRMS 分类树到 Neo4j 知识图谱"
    )
    parser.add_argument(
        "--jsonl",
        default="data/raw/worms_inherited/apsma_worms_enhanced.jsonl",
        help="WoRMS 增强 JSONL 路径"
    )
    parser.add_argument(
        "--relationships",
        default="data/raw/worms_inherited/apsma_worms_relationship_subset.csv",
        help="WoRMS belong_to 关系 CSV 路径"
    )
    parser.add_argument(
        "--image-index",
        default="data/processed/species_image_library/index.json",
        help="物种图片索引 JSON 路径"
    )
    parser.add_argument(
        "--desc-index",
        default="data/processed/species_knowledge_base/index.json",
        help="物种描述索引 JSON 路径"
    )
    parser.add_argument(
        "--alignment-csv",
        default="data/processed/apsma_parsed/worms_apsma_alignment.csv",
        help="APSMA-WoRMS 对齐 CSV 路径"
    )
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="节点批量大小")
    parser.add_argument("--rel-batch-size", type=int, default=5000,
                        help="关系批量大小")
    parser.add_argument("--skip-nodes", action="store_true",
                        help="跳过节点导入")
    parser.add_argument("--skip-relationships", action="store_true",
                        help="跳过关系导入")
    parser.add_argument("--skip-images", action="store_true",
                        help="跳过图片关联")
    parser.add_argument("--skip-descriptions", action="store_true",
                        help="跳过描述关联")
    parser.add_argument("--skip-unmatched", action="store_true",
                        help="跳过未匹配物种导入")
    parser.add_argument("--skip-alignment", action="store_true",
                        help="跳过 APSMA 对齐")
    parser.add_argument("--skip-audit", action="store_true",
                        help="跳过验证审计")
    parser.add_argument("--skip-indexes", action="store_true",
                        help="跳过索引创建（需已存在）")
    args = parser.parse_args()

    t0 = time.time()
    driver = get_neo4j_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    try:
        # ── 步骤 1a: 预创建索引 ──
        if not args.skip_indexes and not args.skip_nodes:
            pre_create_indexes(driver)

        # ── 步骤 1b: 导入 WoRMSTaxon 节点 ──
        if not args.skip_nodes:
            print(f"\n加载 WoRMS 数据: {args.jsonl}")
            records = load_worms_data(args.jsonl)
            print(f"  共 {len(records):,} 条记录")
            import_worms_nodes_batch(driver, records, args.batch_size)

        # ── 步骤 2a: 创建中间索引 ──
        if not args.skip_indexes and not args.skip_relationships:
            create_intermediate_indexes(driver)

        # ── 步骤 2b: 导入 BELONGS_TO 关系 ──
        if not args.skip_relationships:
            print(f"\n导入 BELONGS_TO 关系: {args.relationships}")
            import_worms_relationships_batch(
                driver, args.relationships, args.rel_batch_size
            )

        # ── 步骤 3: 关联图片信息 ──
        if not args.skip_images:
            print(f"\n关联物种图片: {args.image_index}")
            image_map = load_image_index(args.image_index)
            print(f"  图片库物种数: {len(image_map):,}")
            enrich_worms_nodes_with_images(driver, image_map)

        # ── 步骤 4: 关联描述信息 ──
        if not args.skip_descriptions:
            print(f"\n关联物种描述: {args.desc_index}")
            desc_map = load_description_index(args.desc_index)
            print(f"  描述库物种数: {len(desc_map):,}")
            enrich_worms_nodes_with_descriptions(driver, desc_map)

        # ── 步骤 5: 导入双名法未匹配物种 ──
        if not args.skip_unmatched:
            print(f"\n导入双名法未匹配物种")
            image_map = load_image_index(args.image_index)
            desc_map = load_description_index(args.desc_index)
            # 构建 WoRMS 名称集合
            worms_names = set()
            with open(args.jsonl, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    worms_names.add(d["scientificName"])
            import_unmatched_species(driver, image_map, desc_map, worms_names)

        # ── 步骤 6: 创建 APSMA 对齐关系 ──
        if not args.skip_alignment:
            print(f"\n创建 APSMA-WoRMS 对齐: {args.alignment_csv}")
            alignment_data = load_alignment_csv(args.alignment_csv)
            print(f"  对齐记录数: {len(alignment_data)}")
            create_worms_alignment_relations(driver, alignment_data)

        # ── 步骤 7: 创建剩余索引 ──
        if not args.skip_indexes:
            create_post_import_indexes(driver)

        # ── 步骤 8: 验证 ──
        if not args.skip_audit:
            verify_import(driver)

        elapsed = time.time() - t0
        print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    finally:
        driver.close()
