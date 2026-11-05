"""从 opc-52 WoRMS 数据中筛选 APSMA 门级子集并整合到本项目。

应用场景:
    opc-52/WoRMS_download_2025-10-01/ 包含完整的 WoRMS 分类数据:
    - taxon_node.csv (589,562条全球海洋分类节点)
    - taxonrelationship.csv (589,562条分类层级关系)
    - speciesprofile.txt (物种生态属性)

    本项目(DePTH-VLM)关注西太平洋深海生物,只需 APSMA 15门相关物种。
    本脚本完成以下筛选:
    A1: 从 taxon_node.csv 筛选 APSMA 门级分类节点子集
    A2: 从 taxonrelationship.csv 筛选对应关系
    A3: 从 speciesprofile.txt 筛选物种属性
    A4: 合并为增强 JSONL (兼容 09_build_knowledge_graph.py)
"""
import os
import csv
import json
import argparse
from pathlib import Path
from typing import Set, Dict, List

# APSMA 有效门级 (含 CCZ 深渊 + APSMA 图集扩展 + 西太平洋深海新增6门)
# 原15门: APSMA标准门级
APSMA_PHYLA_15 = {
    "Porifera", "Echinodermata", "Cnidaria", "Annelida",
    "Arthropoda", "Xenophyophorea", "Mollusca", "Brachiopoda",
    "Hemichordata", "Chordata", "Nematoda", "Foraminifera",
    "Bryozoa", "Ctenophora", "Nemertea",
}
# 扩展21门: 增加6个西太平洋深海代表性门级
APSMA_PHYLA = APSMA_PHYLA_15 | {
    "Platyhelminthes",   # 扁形动物门 — 深海涡虫/吸虫/绦虫
    "Sipuncula",         # 星虫动物门 — 中国动物志第46卷
    "Echiura",           # 螠虫动物门 — 与星虫同卷
    "Priapulida",        # 曳鳃动物门 — 深海小型底栖
    "Kinorhyncha",       # 动吻动物门 — 深海小型底栖
    "Loricifera",        # 铠甲动物门 — 极深海(深渊区)特有
    "Chaetognatha",      # 毛颚动物门 — 中国动物志第38卷
}


def filter_taxon_nodes(
    input_csv: str,
    output_csv: str,
    phyla: Set[str] = APSMA_PHYLA,
):
    """A1: 从 WoRMS taxon_node.csv 筛选 APSMA 门级分类节点子集。

    只保留:
    - phylum 字段属于 APSMA_PHYLA 的行
    - taxonomicStatus == "accepted" 的行 (排除 unaccepted/synonym)
    """
    print(f"读取: {input_csv}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    kept_ids: Set[str] = set()
    total = 0
    kept = 0
    status_counts = {}

    with open(input_csv, encoding="utf-8-sig") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            phylum = row.get("phylum", "").strip()
            status = row.get("taxonomicStatus", "").strip()

            if phylum in phyla and status == "accepted":
                writer.writerow(row)
                kept_ids.add(row["taxonID"])
                kept += 1
                status_counts[phylum] = status_counts.get(phylum, 0) + 1

            if total % 100000 == 0:
                print(f"  已处理 {total} 行, 保留 {kept} 条...")

    print(f"\n筛选完成: {total} → {kept} 条 (accepted APSMA门级)")
    print("各门统计:")
    for phylum, count in sorted(status_counts.items()):
        print(f"  {phylum}: {count}")

    print(f"输出: {output_csv}")
    return kept_ids


def filter_relationships(
    input_csv: str,
    output_csv: str,
    kept_ids: Set[str],
):
    """A2: 从 WoRMS taxonrelationship.csv 筛选对应关系。

    只保留 :START_ID 和 :END_ID 都在 kept_ids 中的 belong_to 关系。
    """
    print(f"\n读取关系: {input_csv}")
    kept = 0
    total = 0

    with open(input_csv, encoding="utf-8-sig") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            start_id = row.get(":START_ID", "")
            end_id = row.get(":END_ID", "")

            if start_id in kept_ids and (end_id in kept_ids or end_id == ""):
                writer.writerow(row)
                kept += 1

            if total % 100000 == 0:
                print(f"  已处理 {total} 行, 保留 {kept} 条...")

    print(f"关系筛选: {total} → {kept} 条")
    print(f"输出: {output_csv}")
    return kept


def filter_species_profile(
    input_txt: str,
    output_csv: str,
    kept_ids: Set[str],
):
    """A3: 从 WoRMS speciesprofile.txt 筛选物种属性。

    speciesprofile.txt 是 DwC-A 格式(tab分隔),包含:
    - taxonID, lifeStage, sex, establishmentMeans, etc.
    关键字段: minDepthInMeters, maxDepthInMeters (深度范围)
    """
    print(f"\n读取物种属性: {input_txt}")

    # 先读取列名
    with open(input_txt, encoding="utf-8") as f:
        header_line = f.readline().strip()
        columns = header_line.split("\t")

    print(f"列名: {columns[:20]}...")

    kept = 0
    total = 0

    with open(input_txt, encoding="utf-8") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=columns)
        writer.writeheader()

        for line in fin:
            total += 1
            values = line.strip().split("\t")
            row = dict(zip(columns, values))

            taxon_id = row.get("taxonID", "")
            if taxon_id in kept_ids:
                writer.writerow(row)
                kept += 1

    print(f"物种属性筛选: {total} → {kept} 条")
    print(f"输出: {output_csv}")
    return kept


def build_enhanced_jsonl(
    taxon_csv: str,
    relationship_csv: str,
    profile_csv: str,
    output_jsonl: str,
    alignment_csv: str = None,
):
    """A4: 合并为增强 JSONL (兼容 09_build_knowledge_graph.py)。

    将 WoRMS 分类节点 + 关系 + 属性 合并为一条条 JSONL 记录,
    增强形态型的 taxonomy 和 features 字段。
    """
    print(f"\n合并为增强 JSONL...")

    # 1. 读取所有分类节点
    taxon_by_id: Dict[str, dict] = {}
    with open(taxon_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            taxon_by_id[row["taxonID"]] = row
    print(f"分类节点: {len(taxon_by_id)}")

    # 2. 读取层级关系,构建 parent_map
    parent_map: Dict[str, str] = {}
    with open(relationship_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = row.get(":START_ID", "")
            end_id = row.get(":END_ID", "")
            if end_id and start_id:
                parent_map[start_id] = end_id
    print(f"关系: {len(parent_map)}")

    # 3. 读取物种属性
    profile_by_id: Dict[str, dict] = {}
    if os.path.exists(profile_csv):
        with open(profile_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile_by_id[row.get("taxonID", "")] = row
        print(f"物种属性: {len(profile_by_id)}")
    else:
        print(f"物种属性文件不存在: {profile_csv}")

    # 4. 构建增强 JSONL
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    records: List[dict] = []

    for taxon_id, taxon in taxon_by_id.items():
        phylum = taxon.get("phylum", "")
        class_name = taxon.get("class", "")
        order_name = taxon.get("order", "")
        family = taxon.get("family", "")
        genus = taxon.get("genus", "")
        scientific_name = taxon.get("scientificName", "")
        rank = taxon.get("taxonRank", "")

        # 提取 AphiaID (从 taxonID: urn:lsid:marinespecies.org:taxname:558 → 558)
        aphia_id = taxon_id.split(":")[-1] if ":" in taxon_id else taxon_id

        # 物种属性 (深度范围等)
        profile = profile_by_id.get(taxon_id, {})
        min_depth = profile.get("minDepthInMeters", None)
        max_depth = profile.get("maxDepthInMeters", None)
        marine = profile.get("marine", "")
        brackish = profile.get("brackish", "")

        # 父节点
        parent_id = parent_map.get(taxon_id, "")
        parent_name = taxon_by_id.get(parent_id, {}).get("scientificName", "")

        record = {
            "taxonID": taxon_id,
            "aphiaID": aphia_id,
            "scientificName": scientific_name,
            "rank": rank,
            "taxonomy": {
                "kingdom": taxon.get("kingdom", ""),
                "phylum": phylum,
                "class": class_name or "",
                "order": order_name or "",
                "family": family or "",
                "genus": genus or "",
            },
            "parent": {
                "parentID": parent_id,
                "parentName": parent_name,
            },
            "features": {
                "minDepthInMeters": float(min_depth) if min_depth and min_depth not in ("", "None") else None,
                "maxDepthInMeters": float(max_depth) if max_depth and max_depth not in ("", "None") else None,
                "marine": marine,
                "brackish": brackish,
            },
            "status": taxon.get("taxonomicStatus", ""),
            "authority": taxon.get("scientificNameAuthorship", ""),
            "source": "WoRMS_2025-10-01",
        }
        records.append(record)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"增强 JSONL: {len(records)} 条 → {output_jsonl}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 opc-52 WoRMS 数据筛选 APSMA 门级子集"
    )
    parser.add_argument(
        "--worms-dir",
        default="/mnt/oneocean_200TBnas_feitian/opc-52/WoRMS_download_2025-10-01",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/worms_inherited",
    )
    args = parser.parse_args()

    worms_dir = Path(args.worms_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # A1: 筛选分类节点
    kept_ids = filter_taxon_nodes(
        input_csv=str(worms_dir / "taxon_node.csv"),
        output_csv=str(output_dir / "apsma_worms_taxon_subset.csv"),
    )

    # A2: 筛选关系
    filter_relationships(
        input_csv=str(worms_dir / "taxonrelationship.csv"),
        output_csv=str(output_dir / "apsma_worms_relationship_subset.csv"),
        kept_ids=kept_ids,
    )

    # A3: 筛选物种属性
    filter_species_profile(
        input_txt=str(worms_dir / "speciesprofile.txt"),
        output_csv=str(output_dir / "apsma_worms_species_profile.csv"),
        kept_ids=kept_ids,
    )

    # A4: 合并为增强 JSONL
    build_enhanced_jsonl(
        taxon_csv=str(output_dir / "apsma_worms_taxon_subset.csv"),
        relationship_csv=str(output_dir / "apsma_worms_relationship_subset.csv"),
        profile_csv=str(output_dir / "apsma_worms_species_profile.csv"),
        output_jsonl=str(output_dir / "apsma_worms_enhanced.jsonl"),
    )

    print("\n✅ 所有筛选任务完成!")
