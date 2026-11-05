"""Neo4j知识图谱构建 — 分类体系树 + 多视图嵌入 + 反范式化属性 + QA审计 (v3.1.2)"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

from neo4j import GraphDatabase
import torch


def _safe_list_get(lst, idx, default=None):
    """安全获取列表元素，处理None/空列表/索引越界。"""
    if not lst or not isinstance(lst, list):
        return default
    if idx < len(lst):
        return lst[idx]
    return default


def get_neo4j_driver(uri: str = "bolt://localhost:7687",
                     user: str = "neo4j",
                     password: str = "password"):
    return GraphDatabase.driver(uri, auth=(user, password))


def build_kg_from_jsonl(
    jsonl_path: str,
    driver,
    insitu_embeddings_map: Dict = None,
):
    """从APSMA JSONL记录构建完整知识图谱。
    
    v3.1.2特性:
    - 反范式化门/纲/目属性存储在Morphotype节点上
    - insitu_embedding_primary作为向量索引目标
    - class_name替代class(Cypher/Python保留字)
    """
    if insitu_embeddings_map is None:
        insitu_embeddings_map = {}

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.module_loader import load_script_module

    # 动态加载clip嵌入模块(文件名以数字开头，需用module_loader)
    _extract_clip_module = None
    def _get_extract_clip():
        nonlocal _extract_clip_module
        if _extract_clip_module is None:
            _extract_clip_module = load_script_module('07_extract_clip_embeddings')
        return _extract_clip_module

    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    
    print(f"构建KG: {len(records)} 条APSMA形态型记录")
    
    with driver.session() as session:
        for i, record in enumerate(records):
            taxonomy = record["taxonomy"]
            features = record["features"]
            image = record["image"]
            
            # 提取ex-situ嵌入
            exsitu_embedding = None
            img_path = image.get("path", "")
            if img_path and os.path.exists(img_path):
                try:
                    emb = _get_extract_clip().extract_single_clip_embedding(img_path)
                    exsitu_embedding = emb.tolist()
                except Exception as e:
                    print(f"  嵌入提取失败 [{record['morphotype_id']}]: {e}")
            
            # 获取in-situ嵌入
            insitu_data = insitu_embeddings_map.get(record["morphotype_id"], {})
            insitu_emb_list = []
            if isinstance(insitu_data, dict):
                for e in insitu_data.get("embeddings", []):
                    if isinstance(e, torch.Tensor):
                        insitu_emb_list.append(e.tolist())
                    elif isinstance(e, list):
                        insitu_emb_list.append(e)
            insitu_paths = insitu_data.get("paths", []) if isinstance(insitu_data, dict) else []
            
            # v3.1.2: insitu_embedding_primary — 向量索引目标
            insitu_primary = insitu_emb_list[0] if insitu_emb_list else None
            insitu_emb_json = json.dumps(insitu_emb_list)
            
            # 创建节点和关系
            # v3.1.3 修复A: MERGE 属性映射中不允许 null 值(Neo4j 抛 SemanticError
            # "Cannot merge node because of null property value")。将除 morphotype_id
            # (唯一非null匹配键)外的所有 Morphotype 属性移到 ON CREATE SET — Neo4j
            # 对 ON CREATE SET 的 null 值静默忽略(等同不设置该属性), 不报错。使无原位嵌入
            # (insitu_embedding_primary=None) 或 BioCLIP 提取失败(exsitu_embedding=None)
            # 的形态型仍能正常入库; 向量索引后续只索引非null属性值。
            # v3.1.3 修复B: Class/Order 用复合键避免跨门合并。原 MERGE (c:Class {name:...})
            # 仅按 name 全局匹配, 导致不同门下同名 class(如 "unclassified" 被 6 门共享、
            # "Actiniaria" 被 Cnidaria+Porifera 共享)被错误合并为单节点同时连多个门,
            # QA 判定 1804 个 taxonomy_conflicts(零容忍)。改为 Class{name, phylum_name}
            # 和 Order{name, class_name, phylum_name} 复合键, 使每个 Class/Order 唯一归属
            # 其父级, 层级语义正确。class_name/order/phylum 均为非空字符串, 可安全用作 MERGE 键。
            session.run("""
                MERGE (p:Phylum {name: $phylum})
                MERGE (c:Class {name: $class_name, phylum_name: $phylum})
                MERGE (o:Order {name: $order, class_name: $class_name, phylum_name: $phylum})
                MERGE (m:Morphotype {morphotype_id: $mid})
                ON CREATE SET
                    m.name = $name,
                    m.description = $desc,
                    m.exsitu_image_path = $img_path,
                    m.exsitu_embedding = $exsitu_embedding,
                    m.insitu_embeddings_json = $insitu_emb_json,
                    m.insitu_embedding_primary = $insitu_primary,
                    m.insitu_image_paths = $insitu_paths,
                    m.size_min_cm = $min_size,
                    m.size_max_cm = $max_size,
                    m.substrate = $substrate,
                    m.depth_min_m = $depth_min,
                    m.depth_max_m = $depth_max,
                    m.phylum_name = $phylum,
                    m.class_name = $class_name,
                    m.order_name = $order
                MERGE (p)-[:HAS_CLASS]->(c)
                MERGE (c)-[:HAS_ORDER]->(o)
                MERGE (o)-[:HAS_MORPHOTYPE]->(m)
            """,
                phylum=taxonomy.get("phylum", ""),
                class_name=taxonomy.get("class", ""),
                order=taxonomy.get("order", ""),
                mid=record["morphotype_id"],
                name=taxonomy.get("morphotype", ""),
                desc=features.get("description", ""),
                img_path=img_path,
                exsitu_embedding=exsitu_embedding,
                insitu_emb_json=insitu_emb_json,
                insitu_primary=insitu_primary,
                insitu_paths=insitu_paths,
                min_size=_safe_list_get(features.get("size_range_cm"), 0),
                max_size=_safe_list_get(features.get("size_range_cm"), 1),
                substrate=features.get("substrate", ""),
                depth_min=_safe_list_get(features.get("depth_range_m"), 0),
                depth_max=_safe_list_get(features.get("depth_range_m"), 1),
            )
            
            # 存储地理分布
            for area in record.get("occurrence", {}).get("ccz_area", []):
                session.run("""
                    MATCH (m:Morphotype {morphotype_id: $mid})
                    MERGE (a:CCZArea {name: $area})
                    MERGE (m)-[:OCCURS_IN]->(a)
                """, mid=record["morphotype_id"], area=area)
            
            if (i + 1) % 50 == 0:
                print(f"  处理 {i+1}/{len(records)}")
    
    print(f"KG构建完成: {len(records)} 形态型节点")


def create_kg_indexes(driver):
    """创建索引和约束 — v3.2.0 含 WoRMS 层级索引。
    
    必须在KG构建后运行一次！包括:
    - 回填遗留节点属性
    - 唯一性约束
    - B-tree索引
    - 全文索引
    - 向量索引
    - WoRMSTaxon 索引（如已导入）
    """
    with driver.session() as session:
        # 回填反范式化属性
        session.run("""
            MATCH (p:Phylum)-[:HAS_CLASS]->(c:Class)-[:HAS_ORDER]->(o:Order)-[:HAS_MORPHOTYPE]->(m:Morphotype)
            WHERE m.phylum_name IS NULL
            SET m.phylum_name = p.name,
                m.class_name = c.name,
                m.order_name = o.name
        """)
        
        # 回填insitu_embedding_primary (Python侧处理，避免依赖apoc插件)
        result = session.run("""
            MATCH (m:Morphotype)
            WHERE m.insitu_embedding_primary IS NULL
              AND m.insitu_embeddings_json IS NOT NULL
              AND m.insitu_embeddings_json <> '[]'
            RETURN m.morphotype_id AS id, m.insitu_embeddings_json AS emb_json
        """)
        for r in result:
            try:
                emb_list = json.loads(r["emb_json"])
                if emb_list:
                    session.run(
                        "MATCH (m:Morphotype {morphotype_id: $mid}) "
                        "SET m.insitu_embedding_primary = $first_emb",
                        mid=r["id"], first_emb=emb_list[0]
                    )
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  回填insitu_embedding_primary失败 [{r['id']}]: {e}")
        
        # 唯一性约束
        session.run(
            "CREATE CONSTRAINT morphotype_id_unique IF NOT EXISTS "
            "FOR (m:Morphotype) REQUIRE m.morphotype_id IS UNIQUE"
        )
        
        # B-tree索引
        for prop in ("phylum_name", "class_name", "order_name"):
            session.run(
                f"CREATE INDEX morphotype_{prop}_idx IF NOT EXISTS "
                f"FOR (m:Morphotype) ON (m.{prop})"
            )
        
        # 全文索引
        session.run(
            "CREATE FULLTEXT INDEX morphotype_text_index IF NOT EXISTS "
            "FOR (m:Morphotype) ON EACH [m.name, m.description, m.substrate]"
        )
        
        # 向量索引 - exsitu
        session.run("""
            CREATE VECTOR INDEX morphotype_exsitu_embeddings IF NOT EXISTS
            FOR (m:Morphotype) ON (m.exsitu_embedding)
            OPTIONS { indexConfig: {
                `vector.dimensions`: 512,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        
        # 向量索引 - insitu
        session.run("""
            CREATE VECTOR INDEX morphotype_insitu_embeddings IF NOT EXISTS
            FOR (m:Morphotype) ON (m.insitu_embedding_primary)
            OPTIONS { indexConfig: {
                `vector.dimensions`: 512,
                `vector.similarity_function`: 'cosine'
            }}
        """)
    
    print("[KG] 索引创建 + 反范式化属性回填完成")

    # ── WoRMSTaxon 索引（如果 WoRMS 分类树已导入）──
    try:
        with driver.session() as session:
            # 检查 WoRMSTaxon 节点是否存在
            has_worms = session.run(
                "MATCH (t:WoRMSTaxon) RETURN count(t) > 0 AS has"
            ).single()["has"]
    except Exception:
        has_worms = False

    if has_worms:
        with driver.session() as session:
            # 唯一性约束
            session.run(
                "CREATE CONSTRAINT wormstaxon_aphia_id_unique IF NOT EXISTS "
                "FOR (t:WoRMSTaxon) REQUIRE t.aphia_id IS UNIQUE"
            )
            # B-tree 索引
            session.run(
                "CREATE INDEX wormstaxon_taxon_id_idx IF NOT EXISTS "
                "FOR (t:WoRMSTaxon) ON (t.taxon_id)"
            )
            session.run(
                "CREATE INDEX wormstaxon_rank_idx IF NOT EXISTS "
                "FOR (t:WoRMSTaxon) ON (t.rank)"
            )
            session.run(
                "CREATE INDEX wormstaxon_phylum_idx IF NOT EXISTS "
                "FOR (t:WoRMSTaxon) ON (t.phylum)"
            )
            session.run(
                "CREATE INDEX wormstaxon_family_idx IF NOT EXISTS "
                "FOR (t:WoRMSTaxon) ON (t.family)"
            )
            session.run(
                "CREATE INDEX wormstaxon_has_images_idx IF NOT EXISTS "
                "FOR (t:WoRMSTaxon) ON (t.has_images)"
            )
            # 复合索引
            session.run(
                "CREATE INDEX wormstaxon_name_phylum_idx IF NOT EXISTS "
                "FOR (t:WoRMSTaxon) ON (t.scientific_name, t.phylum)"
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
        print("[KG] WoRMSTaxon 索引创建完成")
    else:
        print("[KG] WoRMSTaxon 节点未导入，跳过 WoRMS 索引")


def run_kg_quality_audit(driver):
    """KG质量审计 — 孤立节点、断裂层级、缺失属性、分类冲突、统计异常、文件引用。"""
    report = {}
    
    with driver.session() as session:
        # 检查1: 孤立节点
        orphans = session.run("""
            MATCH (m:Morphotype)
            WHERE NOT (m)-[:WORMS_MATCHED]->(:WoRMSTaxon)
            AND NOT (m)-[:SIMILAR_TO]->(:Morphotype)
            AND m.exsitu_embedding IS NULL
            AND (m.insitu_embeddings_json IS NULL OR m.insitu_embeddings_json = '[]')
            RETURN m.morphotype_id AS id, m.name AS name
        """)
        report["orphan_nodes"] = [dict(r) for r in orphans]
        
        # 检查2: 断裂层级
        broken_phylum = session.run("MATCH (p:Phylum) WHERE NOT (p)-[:HAS_CLASS]->(:Class) RETURN p.name AS phylum")
        broken_class = session.run("MATCH (c:Class) WHERE NOT (c)-[:HAS_ORDER]->(:Order) RETURN c.name AS class")
        broken_order = session.run("MATCH (o:Order) WHERE NOT (o)-[:HAS_MORPHOTYPE]->(:Morphotype) RETURN o.name AS order_name")
        report["broken_hierarchy"] = {
            "phylum_no_class": [dict(r) for r in broken_phylum],
            "class_no_order": [dict(r) for r in broken_class],
            "order_no_morphotype": [dict(r) for r in broken_order],
        }
        
        # 检查3: 缺失属性
        missing_desc = session.run("MATCH (m:Morphotype) WHERE m.description IS NULL OR size(m.description) < 20 RETURN m.morphotype_id AS id")
        missing_img = session.run("MATCH (m:Morphotype) WHERE m.exsitu_image_path IS NULL OR m.exsitu_image_path = '' RETURN m.morphotype_id AS id")
        zero_emb = session.run("MATCH (m:Morphotype) WHERE m.exsitu_embedding IS NULL OR size(m.exsitu_embedding) = 0 RETURN m.morphotype_id AS id")
        report["missing_attributes"] = {
            "description_short": [dict(r) for r in missing_desc],
            "missing_image_path": [dict(r) for r in missing_img],
            "zero_embedding": [dict(r) for r in zero_emb],
        }
        
        # 检查4: 分类冲突
        conflicts = session.run("""
            MATCH (m:Morphotype)<-[:HAS_MORPHOTYPE]-(o1:Order)<-[:HAS_ORDER]-(c1:Class)<-[:HAS_CLASS]-(p1:Phylum)
            MATCH (m)<-[:HAS_MORPHOTYPE]-(o2:Order)<-[:HAS_ORDER]-(c2:Class)<-[:HAS_CLASS]-(p2:Phylum)
            WHERE p1.name <> p2.name
            RETURN m.morphotype_id AS id, p1.name AS phylum1, p2.name AS phylum2
        """)
        report["taxonomy_conflicts"] = [dict(r) for r in conflicts]
        
        # 检查5: 稀疏门
        sparse = session.run("""
            MATCH (p:Phylum)-[:HAS_CLASS]->()-[:HAS_ORDER]->()-[:HAS_MORPHOTYPE]->(m:Morphotype)
            WITH p, count(m) AS cnt WHERE cnt < 3
            RETURN p.name AS phylum, cnt AS morphotype_count
        """)
        report["sparse_phyla"] = [dict(r) for r in sparse]
        
        # 检查6: 图像文件存在性
        img_refs = session.run("MATCH (m:Morphotype) WHERE m.exsitu_image_path IS NOT NULL RETURN m.morphotype_id AS id, m.exsitu_image_path AS path")
        missing_files = [dict(r) for r in img_refs if not os.path.exists(r["path"])]
        report["missing_image_files"] = missing_files
        
        # ═══ WoRMS 审计（v3.2.0 新增）═══
        try:
            has_worms = session.run(
                "MATCH (t:WoRMSTaxon) RETURN count(t) > 0 AS has"
            ).single()["has"]
        except Exception:
            has_worms = False

        if has_worms:
            # W1: WoRMSTaxon 节点按 rank 分布
            worms_rank_stats = session.run("""
                MATCH (t:WoRMSTaxon)
                RETURN t.rank AS rank, count(t) AS cnt
                ORDER BY cnt DESC
            """)
            report["worms_rank_distribution"] = [dict(r) for r in worms_rank_stats]

            # W2: 孤立 WoRMSTaxon
            isolated_worms = session.run("""
                MATCH (t:WoRMSTaxon)
                WHERE NOT (t)-[:BELONGS_TO]->(:WoRMSTaxon)
                  AND NOT (:WoRMSTaxon)-[:BELONGS_TO]->(t)
                  AND t.rank <> 'Phylum'
                RETURN count(t) AS cnt
            """)
            report["isolated_worms_taxa"] = isolated_worms.single()["cnt"]

            # W3: BELONGS_TO 关系数
            belongs_to_count = session.run(
                "MATCH (:WoRMSTaxon)-[r:BELONGS_TO]->(:WoRMSTaxon) RETURN count(r) AS cnt"
            )
            report["belongs_to_count"] = belongs_to_count.single()["cnt"]

            # W4: APSMA 对齐覆盖率
            alignment_coverage = session.run("""
                MATCH (m:Morphotype) WITH count(m) AS total
                OPTIONAL MATCH (m2:Morphotype)-[:WORMS_MATCHED]->(:WoRMSTaxon)
                WITH total, count(m2) AS aligned
                RETURN total, aligned, toFloat(aligned)/toFloat(total) * 100 AS pct
            """)
            report["worms_alignment_coverage"] = dict(alignment_coverage.single())

            # W5: 有图片/描述的 WoRMSTaxon 统计
            media_stats = session.run("""
                MATCH (t:WoRMSTaxon)
                RETURN count(t) AS total,
                       sum(CASE WHEN t.has_images = true THEN 1 ELSE 0 END) AS with_images,
                       sum(CASE WHEN t.has_description = true THEN 1 ELSE 0 END) AS with_desc
            """)
            report["worms_media_stats"] = dict(media_stats.single())

            # W6: Phylum 节点数
            phylum_count = session.run(
                "MATCH (t:WoRMSTaxon {rank: 'Phylum'}) RETURN count(t) AS cnt"
            )
            report["worms_phylum_count"] = phylum_count.single()["cnt"]
    
    total = (len(report["orphan_nodes"]) +
             sum(len(v) for v in report["broken_hierarchy"].values()) +
             sum(len(v) for v in report["missing_attributes"].values()) +
             len(report["taxonomy_conflicts"]) +
             len(report["sparse_phyla"]) +
             len(report["missing_image_files"]))
    
    print(f"KG QA报告: {total} 个问题")
    for key, val in report.items():
        if isinstance(val, list):
            print(f"  - {key}: {len(val)}")
        elif isinstance(val, dict):
            try:
                print(f"  - {key}: {sum(len(v) for v in val.values())}")
            except TypeError:
                print(f"  - {key}: {val}")
        elif isinstance(val, (int, float)):
            pass  # 标量值在后续摘要中打印

    # 打印 WoRMS 审计摘要
    if report.get("worms_rank_distribution"):
        print(f"\n  WoRMS 审计:")
        print(f"    Phylum 数: {report.get('worms_phylum_count', 'N/A')}")
        print(f"    BELONGS_TO: {report.get('belongs_to_count', 'N/A'):,}")
        print(f"    孤立节点: {report.get('isolated_worms_taxa', 'N/A')}")
        ac = report.get("worms_alignment_coverage", {})
        if ac:
            print(f"    APSMA 对齐: {ac.get('aligned', 0)}/{ac.get('total', 0)} ({ac.get('pct', 0):.1f}%)")
        ms = report.get("worms_media_stats", {})
        if ms:
            print(f"    有图片: {ms.get('with_images', 0):,} / {ms.get('total', 0):,}")
            print(f"    有描述: {ms.get('with_desc', 0):,} / {ms.get('total', 0):,}")
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建Neo4j知识图谱")
    parser.add_argument("--jsonl", default="data/processed/apsma_parsed/apsma_knowledge_validated.jsonl")
    parser.add_argument("--insitu-embeddings", default=None, help="In-situ嵌入pt文件路径")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()
    
    driver = get_neo4j_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    try:
        insitu_map = {}
        if args.insitu_embeddings and os.path.exists(args.insitu_embeddings):
            insitu_map = torch.load(args.insitu_embeddings, weights_only=False)

        build_kg_from_jsonl(args.jsonl, driver, insitu_map)
        create_kg_indexes(driver)

        if not args.skip_audit:
            run_kg_quality_audit(driver)
    finally:
        driver.close()
