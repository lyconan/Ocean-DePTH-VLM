"""补全 Unknown 物种的门级分类。

从 opc-52 完整 WoRMS taxon_node.csv 查询 Unknown 物种的真实门级:
1. 精确匹配 scientificName
2. 如果 phylum 在 APSMA 21门中 → 使用该 phylum
3. 如果是 unaccepted 同物异名 → 查找 acceptedNameUsage 对应记录的 phylum
4. 如果 phylum 不在 21门中(硅藻/植物) → 保留为 Unknown

然后将物种的符号链接从 Unknown/ 移动到正确的 {Phylum}/ 目录,
同时更新知识库 descriptions/Unknown/ 下的 MD 链接。

输入:
    data/processed/species_image_library/Unknown/{species}/  (1,638个)
    data/processed/species_knowledge_base/descriptions/Unknown/{species}.md
    /mnt/oneocean_200TBnas_feitian/opc-52/WoRMS_download_2025-10-01/taxon_node.csv

输出:
    data/processed/species_image_library/{Phylum}/{species}/  (重新分类)
    data/processed/species_knowledge_base/descriptions/{Phylum}/{species}.md
"""
import csv
import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import sys

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data" / "processed" / "species_image_library"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "species_knowledge_base"
TAXON_CSV = Path("/mnt/oneocean_200TBnas_feitian/opc-52/WoRMS_download_2025-10-01/taxon_node.csv")

# APSMA 扩展21门
APSMA_PHYLA = {
    "Porifera", "Echinodermata", "Cnidaria", "Annelida",
    "Arthropoda", "Xenophyophorea", "Mollusca", "Brachiopoda",
    "Hemichordata", "Chordata", "Nematoda", "Foraminifera",
    "Bryozoa", "Ctenophora", "Nemertea",
    "Platyhelminthes", "Sipuncula", "Echiura",
    "Priapulida", "Kinorhyncha", "Loricifera", "Chaetognatha",
}


def load_unknown_species():
    """加载 Unknown 目录下的物种列表。"""
    unknown_img_dir = IMAGE_LIBRARY_DIR / "Unknown"
    unknown_desc_dir = KNOWLEDGE_BASE_DIR / "descriptions" / "Unknown"

    # 图片库 Unknown 物种
    img_species = set()
    if unknown_img_dir.exists():
        for d in unknown_img_dir.iterdir():
            if d.is_dir():
                # 目录名格式: {Species_Name} (下划线代替空格)
                species = d.name.replace("_", " ")
                img_species.add((d.name, species))

    # 知识库 Unknown 物种
    desc_species = set()
    if unknown_desc_dir.exists():
        for f in unknown_desc_dir.iterdir():
            if f.is_symlink() or f.is_file():
                species = f.stem.replace("_", " ")
                desc_species.add((f.name, species))

    return img_species, desc_species


def build_worms_lookup():
    """从 opc-52 完整 taxon_node.csv 构建 scientificName → (phylum, status, acceptedNameUsage) 映射。

    同时构建 acceptedNameUsage → phylum 映射,用于查询同物异名的门级。
    """
    print(f"加载 WoRMS 完整数据: {TAXON_CSV}")
    print(f"  (512MB, 预计1-2分钟)")

    # name → list of (phylum, status, acceptedNameUsage)
    name_to_info = {}
    # accepted name → phylum (用于查找同物异名的门级)
    accepted_to_phylum = {}

    with open(TAXON_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        total = 0
        for row in reader:
            total += 1
            sci_name = row.get("scientificName", "").strip()
            phylum = row.get("phylum", "").strip()
            status = row.get("taxonomicStatus", "").strip()
            accepted = row.get("acceptedNameUsage", "").strip()

            if not sci_name:
                continue

            # 记录每条记录
            if sci_name not in name_to_info:
                name_to_info[sci_name] = []
            name_to_info[sci_name].append({
                "phylum": phylum,
                "status": status,
                "accepted": accepted,
            })

            # 如果是 accepted,记录到 accepted_to_phylum
            if status == "accepted" and phylum:
                accepted_to_phylum[sci_name] = phylum

    print(f"  加载 {total} 条记录, {len(name_to_info)} 个唯一学名")
    return name_to_info, accepted_to_phylum


def resolve_species_phylum(species_name, name_to_info, accepted_to_phylum):
    """查询物种的真实门级。

    Args:
        species_name: 物种名(空格分隔,如 "Holothuria scabra")
        name_to_info: scientificName → [{phylum, status, accepted}] 映射
        accepted_to_phylum: accepted学名 → phylum 映射

    Returns:
        (resolved_phylum, resolution_method)
        resolved_phylum: 门级名(在21门中) 或 "" (无法解析)
        resolution_method: "direct" / "via_accepted" / "genus" / ""
    """
    # 1. 精确匹配 scientificName
    if species_name in name_to_info:
        for info in name_to_info[species_name]:
            phylum = info["phylum"]
            if phylum in APSMA_PHYLA:
                return phylum, "direct"

        # 2. 如果精确匹配但门级不在21门中,且是 unaccepted → 查找 acceptedNameUsage
        for info in name_to_info[species_name]:
            accepted = info["accepted"]
            if accepted and accepted != species_name:
                # 查找 accepted 学名的门级
                if accepted in accepted_to_phylum:
                    phylum = accepted_to_phylum[accepted]
                    if phylum in APSMA_PHYLA:
                        return phylum, "via_accepted"
                # 也可能在 name_to_info 中(非 accepted 状态但有门级)
                if accepted in name_to_info:
                    for acc_info in name_to_info[accepted]:
                        if acc_info["phylum"] in APSMA_PHYLA:
                            return acc_info["phylum"], "via_accepted"

    # 3. 尝试属名匹配 (第一个单词)
    if " " in species_name:
        genus = species_name.split()[0]
        if genus in name_to_info:
            # 找到属名对应的门级
            for info in name_to_info[genus]:
                if info["phylum"] in APSMA_PHYLA:
                    return info["phylum"], "genus"

    return "", ""


def move_species_dir(src_dir: Path, dst_dir: Path):
    """移动物种目录(含符号链接图片)到新门级目录。"""
    if not src_dir.exists():
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = 0

    for item in src_dir.iterdir():
        dst_item = dst_dir / item.name
        if dst_item.exists():
            continue
        # 移动符号链接或文件
        shutil.move(str(item), str(dst_item))
        moved += 1

    return moved


def move_desc_file(src_file: Path, dst_file: Path):
    """移动描述 MD 文件到新门级目录。"""
    if not src_file.exists():
        return False
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if dst_file.exists():
        return False
    shutil.move(str(src_file), str(dst_file))
    return True


def main():
    print("=" * 70)
    print("Unknown 物种门级补全")
    print("=" * 70)

    # 1. 加载 Unknown 物种
    img_species, desc_species = load_unknown_species()
    print(f"\nUnknown 图片物种: {len(img_species)}")
    print(f"Unknown 描述物种: {len(desc_species)}")

    # 合并所有 Unknown 物种名
    all_unknown_names = set()
    for _, species in img_species:
        all_unknown_names.add(species)
    for _, species in desc_species:
        all_unknown_names.add(species)
    print(f"合并去重后 Unknown 物种: {len(all_unknown_names)}")

    # 2. 加载 WoRMS 完整数据
    name_to_info, accepted_to_phylum = build_worms_lookup()

    # 3. 解析每个 Unknown 物种的真实门级
    print(f"\n解析 {len(all_unknown_names)} 个 Unknown 物种...")
    resolved = {}  # species_name → (phylum, method)
    stats = defaultdict(int)
    stats_by_phylum = defaultdict(int)
    stats_by_method = defaultdict(int)

    for species in all_unknown_names:
        phylum, method = resolve_species_phylum(species, name_to_info, accepted_to_phylum)
        resolved[species] = (phylum, method)
        stats_by_method[method or "unresolved"] += 1
        if phylum:
            stats_by_phylum[phylum] += 1
        else:
            stats["unresolved"] += 1

    print(f"\n解析结果 (按方法):")
    for method, count in sorted(stats_by_method.items(), key=lambda x: -x[1]):
        print(f"  {method:20s}: {count}")

    print(f"\n解析结果 (按门级,可补全的):")
    for phylum, count in sorted(stats_by_phylum.items(), key=lambda x: -x[1]):
        print(f"  {phylum:25s}: {count}")

    unresolved = stats["unresolved"]
    print(f"\n无法补全 (硅藻/植物/未收录): {unresolved}")

    # 4. 移动图片符号链接
    print(f"\n移动图片库 Unknown → 正确门级...")
    moved_img = 0
    moved_img_files = 0
    unknown_img_dir = IMAGE_LIBRARY_DIR / "Unknown"

    for dirname, species in img_species:
        if species not in resolved:
            continue
        phylum, method = resolved[species]
        if not phylum:
            continue  # 保留为 Unknown

        src_dir = unknown_img_dir / dirname
        dst_dir = IMAGE_LIBRARY_DIR / phylum / dirname
        if not src_dir.exists():
            continue
        n = move_species_dir(src_dir, dst_dir)
        if n > 0:
            moved_img += 1
            moved_img_files += n
            # 删除空的源目录
            try:
                src_dir.rmdir()
            except OSError:
                pass

    print(f"  移动 {moved_img} 个物种目录, {moved_img_files} 个图片文件")

    # 5. 移动描述 MD 文件
    print(f"\n移动知识库 Unknown → 正确门级...")
    moved_desc = 0
    unknown_desc_dir = KNOWLEDGE_BASE_DIR / "descriptions" / "Unknown"

    for filename, species in desc_species:
        if species not in resolved:
            continue
        phylum, method = resolved[species]
        if not phylum:
            continue

        src_file = unknown_desc_dir / filename
        dst_file = KNOWLEDGE_BASE_DIR / "descriptions" / phylum / filename
        if move_desc_file(src_file, dst_file):
            moved_desc += 1

    print(f"  移动 {moved_desc} 个描述文件")

    # 6. 输出解析报告
    report = {
        "total_unknown": len(all_unknown_names),
        "resolved": sum(1 for p, _ in resolved.values() if p),
        "unresolved": sum(1 for p, _ in resolved.values() if not p),
        "by_method": dict(stats_by_method),
        "by_phylum": dict(stats_by_phylum),
        "moved_image_species": moved_img,
        "moved_image_files": moved_img_files,
        "moved_description_files": moved_desc,
        "resolved_species": {
            species: {"phylum": p, "method": m}
            for species, (p, m) in resolved.items()
            if p
        },
        "unresolved_species_sample": [
            species for species, (p, _) in resolved.items()
            if not p
        ][:50],
    }

    report_path = PROJECT_ROOT / "data" / "processed" / "species_image_library" / "unknown_resolution_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n解析报告: {report_path}")

    print(f"\n✅ Unknown 物种门级补全完成!")
    print(f"  补全: {report['resolved']} / {report['total_unknown']}")
    print(f"  未补全: {report['unresolved']} (硅藻/植物/未收录)")
    print(f"  移动图片: {moved_img} 物种 / {moved_img_files} 文件")
    print(f"  移动描述: {moved_desc} 文件")


if __name__ == "__main__":
    main()
