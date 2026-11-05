"""将 species_knowledge_base/structured/ 中的物种描述导入 Neo4j。

对每个 JSON 文件:
1. 解析 phylum(从目录名)和 species(从文件名)
2. 构造 Neo4j morphotype_id: WORMS_{Phylum}_{Species}
3. 提取 sections 中的描述信息
4. 更新 Neo4j Morphotype 节点的 description 字段

用法:
    docker exec -e PYTHONUNBUFFERED=1 -w /project GPU_USFM_Paper \
        conda run --no-capture-output -n qwen2.5 \
        python -u scripts/utils/import_worms_descriptions.py [--dry-run] [--batch-size 200]
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STRUCTURED_DIR = PROJECT_ROOT / "data/processed/species_knowledge_base/structured"

# Neo4j 配置(与 graphrag_inference.yaml 一致)
NEO4J_URI = "bolt://172.17.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def extract_description_from_json(json_path: Path) -> dict:
    """从结构化 JSON 中提取描述信息。

    Returns:
        {
            "description": str,     # 合并后的描述(sections 中的 description/diagnosis)
            "distribution": str,    # 分布信息
            "habitat": str,         # 栖息地信息
            "remarks": str,         # 备注
            "summary": str,         # 前 500 字符摘要(兜底)
        }
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    sections = data.get("sections", {})
    summary = data.get("summary", "")

    # 优先从 sections 提取
    description_parts = []
    for key in ["description", "diagnosis"]:
        parts = sections.get(key, [])
        if parts:
            # 取最长的片段(通常是最完整的)
            best = max(parts, key=len)
            if len(best) > 30:  # 过滤太短的垃圾
                description_parts.append(best.strip())

    distribution_parts = sections.get("distribution", [])
    habitat_parts = sections.get("habitat", [])
    remarks_parts = sections.get("remarks", [])

    # 合并描述
    description = "\n\n".join(description_parts) if description_parts else ""
    distribution = max(distribution_parts, key=len).strip() if distribution_parts else ""
    habitat = max(habitat_parts, key=len).strip() if habitat_parts else ""
    remarks = max(remarks_parts, key=len).strip() if remarks_parts else ""

    # 如果 sections 没有有效描述,尝试从 summary 提取(但跳过图片hash垃圾)
    if not description and summary:
        import re
        # 去除图片引用标记和哈希碎片
        clean = re.sub(r'!\[\]\(images/[a-f0-9]+\.jpg\)', '', summary)
        clean = re.sub(r'[a-f0-9]{20,}\.jpg\)', '', clean)
        clean = re.sub(r'[a-f0-9]{20,}', '', clean)  # 残留的长哈希
        # 按行分割,过滤掉太短的行和标题行
        lines = [l.strip() for l in clean.split('\n')
                 if l.strip()
                 and not l.strip().startswith('#')
                 and len(l.strip()) > 20]  # 跳过太短的行(通常是页码/标题)
        # 找包含生物学术语的行
        bio_lines = [l for l in lines if any(kw in l.lower() for kw in
            ['description', 'diagnosis', 'morphology', 'character', '形态', '特征', '描述',
             'body', 'tentacle', 'polyp', 'colony', 'spine', 'tube', 'shell',
             'segment', 'setae', 'cirri', 'radial', 'oral', 'specimen',
             'diameter', 'length', 'width', 'color', 'surface'])]
        if bio_lines:
            description = ' '.join(bio_lines[:3])[:500].strip()
        # 没有生物学术语的行很可能是 OCR 垃圾,不兜底提取

    # 质量检查: 过滤纯 OCR 垃圾(全大写无空格的文本)
    if description and _is_ocr_garbage(description):
        description = ""

    return {
        "description": description[:2000] if description else "",  # 截断防止过长
        "distribution": distribution[:1000] if distribution else "",
        "habitat": habitat[:1000] if habitat else "",
        "remarks": remarks[:1000] if remarks else "",
    }


def _is_ocr_garbage(text: str) -> bool:
    """检测是否为 OCR 垃圾文本。"""
    if not text:
        return False
    # 去除常见标记后检查
    clean = text.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    clean = clean.replace("!", "").replace("(", "").replace(")", "")
    # 1. 如果超过 80% 是大写字母且没有小写,很可能是 OCR 垃圾
    upper = sum(1 for c in clean if c.isupper())
    alpha = sum(1 for c in clean if c.isalpha())
    if alpha > 50 and upper / alpha > 0.8:
        return True
    # 2. 包含大量图片哈希引用(40+位hex字符串)
    import re
    hash_count = len(re.findall(r'[a-f0-9]{40,}', text))
    if hash_count >= 2:
        return True
    # 3. 描述太短(< 50 有效字符)且包含OCR常见噪声词
    noise_words = ['fuf', 'muf', 'muco', 'proftati', 'extremite', 'HISTOIRE',
                   'NATURELLE', 'planches', 'expedition', 'lat.S', 'long.E']
    effective_len = len(re.sub(r'\s+', '', clean))
    if effective_len < 80 and sum(1 for w in noise_words if w in text) >= 2:
        return True
    # 4. 几乎没有英文/中文字符(全是非ASCII符号或罕见拉丁语)
    en_cn_chars = sum(1 for c in text if c.isascii() and c.isalpha() and c.islower())
    if alpha > 100 and en_cn_chars / alpha < 0.15:
        return True
    return False


def scan_structured_files():
    """扫描所有结构化 JSON 文件,返回 (phylum, species, json_path) 列表。"""
    results = []
    for phylum_dir in STRUCTURED_DIR.iterdir():
        if not phylum_dir.is_dir():
            continue
        phylum = phylum_dir.name
        for json_file in phylum_dir.iterdir():
            if not json_file.name.endswith(".json"):
                continue
            species = json_file.stem  # 文件名去掉 .json
            results.append((phylum, species, json_file))
    return results


def main():
    parser = argparse.ArgumentParser(description="导入 WoRMS 物种描述到 Neo4j")
    parser.add_argument("--dry-run", action="store_true", help="只扫描不写入")
    parser.add_argument("--batch-size", type=int, default=200, help="批量提交大小")
    parser.add_argument("--min-desc-len", type=int, default=30, help="描述最短长度(低于此跳过)")
    args = parser.parse_args()

    print("=" * 70)
    print("WoRMS 物种描述导入 Neo4j")
    print("=" * 70)

    # 1. 扫描所有结构化文件
    all_files = scan_structured_files()
    print(f"\n扫描到 {len(all_files)} 个结构化 JSON 文件")

    # 2. 解析描述
    parsed = []
    skipped_no_desc = 0
    skipped_garbage = 0
    by_phylum = defaultdict(int)
    has_sections = 0

    for phylum, species, json_path in all_files:
        result = extract_description_from_json(json_path)
        if result is None:
            continue

        desc = result["description"]
        if not desc or len(desc) < args.min_desc_len:
            skipped_no_desc += 1
            continue

        if _is_ocr_garbage(desc):
            skipped_garbage += 1
            continue

        # 构造 Neo4j morphotype_id
        morphotype_id = f"WORMS_{phylum}_{species}"
        parsed.append({
            "morphotype_id": morphotype_id,
            "species": species,
            "phylum": phylum,
            "description": desc,
            "distribution": result["distribution"],
            "habitat": result["habitat"],
            "remarks": result["remarks"],
        })
        by_phylum[phylum] += 1
        if result.get("distribution") or result.get("habitat"):
            has_sections += 1

    print(f"\n解析结果:")
    print(f"  有效描述: {len(parsed):,}")
    print(f"  无描述(跳过): {skipped_no_desc:,}")
    print(f"  OCR垃圾(跳过): {skipped_garbage:,}")
    print(f"  有分布/生境信息: {has_sections:,}")

    print(f"\n按门级分布(前15):")
    for phylum, count in sorted(by_phylum.items(), key=lambda x: -x[1])[:15]:
        print(f"  {phylum:25s}: {count:5,}")

    if args.dry_run:
        print(f"\n[Dry-run] 不写入 Neo4j")
        # 打印几个样本
        for p in parsed[:3]:
            print(f"\n  样本: {p['morphotype_id']}")
            print(f"    desc: {p['description'][:120]}...")
        return

    # 3. 批量写入 Neo4j
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    updated = 0
    not_found = 0
    errors = 0

    with driver.session() as session:
        # 同时匹配 Morphotype 和 WoRMSTaxon 节点
        # Morphotype: morphotype_id = WORMS_{phylum}_{species}
        # WoRMSTaxon: scientific_name = {species} AND phylum = {phylum}
        existing_ids = set()
        matched_taxon = {}  # scientific_name -> WoRMSTaxon properties

        # 先查 Morphotype
        for i in range(0, len(parsed), 1000):
            batch = parsed[i:i+1000]
            ids = [p["morphotype_id"] for p in batch]
            result = session.run(
                "UNWIND $ids AS mid MATCH (m:Morphotype {morphotype_id: mid}) RETURN m.morphotype_id AS id",
                ids=ids
            )
            for row in result:
                existing_ids.add(row["id"])

        # 再查 WoRMSTaxon (通过 scientific_name + phylum 匹配)
        for i in range(0, len(parsed), 1000):
            batch = parsed[i:i+1000]
            name_phylum = [{"name": p["species"], "phylum": p["phylum"]} for p in batch]
            result = session.run("""
                UNWIND $rows AS row
                MATCH (t:WoRMSTaxon {scientific_name: row.name, phylum: row.phylum})
                RETURN t.scientific_name AS name, t.phylum AS phylum
            """, rows=name_phylum)
            for row in result:
                key = f"{row['phylum']}_{row['name']}"
                matched_taxon[key] = True

        # 合并匹配结果
        for p in parsed:
            key = f"{p['phylum']}_{p['species']}"
            if key in matched_taxon:
                existing_ids.add(p["morphotype_id"])

        print(f"\nNeo4j 中匹配到 {len(existing_ids)} / {len(parsed)} 个节点")

        # 批量更新
        batch_data = []
        for p in parsed:
            if p["morphotype_id"] in existing_ids:
                batch_data.append(p)

        for i in range(0, len(batch_data), args.batch_size):
            batch = batch_data[i:i+args.batch_size]
            try:
                # 更新 Morphotype 节点
                result = session.run("""
                    UNWIND $batch AS row
                    MATCH (m:Morphotype {morphotype_id: row.morphotype_id})
                    SET m.description = row.description,
                        m.distribution = row.distribution,
                        m.habitat = row.habitat,
                        m.remarks = row.remarks
                    RETURN count(m) AS updated
                """, batch=batch)
                count = result.single()["updated"]
                updated += count

                # 同时更新 WoRMSTaxon 节点 (通过 scientific_name + phylum 匹配)
                name_phylum_batch = [
                    {"name": p["species"], "phylum": p["phylum"],
                     "description": p["description"], "distribution": p["distribution"],
                     "habitat": p["habitat"], "remarks": p["remarks"]}
                    for p in batch
                ]
                result2 = session.run("""
                    UNWIND $rows AS row
                    MATCH (t:WoRMSTaxon {scientific_name: row.name, phylum: row.phylum})
                    SET t.description = row.description,
                        t.distribution = row.distribution,
                        t.habitat = row.habitat,
                        t.remarks = row.remarks,
                        t.has_description = true
                    RETURN count(t) AS updated
                """, rows=name_phylum_batch)
                count2 = result2.single()["updated"]
                updated += count2
            except Exception as e:
                errors += 1
                print(f"  批次 {i//args.batch_size} 错误: {e}")

            if (i // args.batch_size + 1) % 10 == 0:
                print(f"  已更新: {updated:,} / {len(batch_data):,}")

    driver.close()

    not_found = len(parsed) - len(batch_data)

    print(f"\n{'='*70}")
    print(f"导入完成!")
    print(f"  更新: {updated:,}")
    print(f"  Neo4j中不存在: {not_found:,}")
    print(f"  错误: {errors}")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
