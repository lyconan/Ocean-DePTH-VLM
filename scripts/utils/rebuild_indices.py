"""从已存在的目录结构重建图片检索索引和知识库索引。

用途:
    当 integrate_oceanus_all.py 的 E3/E4 任务已运行过，但索引文件为空或不完整时，
    使用本脚本从已存在的目录结构扫描重建索引，无需重新运行耗时的 E3/E4 任务。

输出:
    data/processed/species_image_library/index.json
    data/processed/species_knowledge_base/index.json
"""
import json
from pathlib import Path
from collections import defaultdict
import sys

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data" / "processed" / "species_image_library"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "species_knowledge_base"

# APSMA 扩展21门 + Other
APSMA_PHYLA = {
    "Porifera", "Echinodermata", "Cnidaria", "Annelida",
    "Arthropoda", "Xenophyophorea", "Mollusca", "Brachiopoda",
    "Hemichordata", "Chordata", "Nematoda", "Foraminifera",
    "Bryozoa", "Ctenophora", "Nemertea",
    "Platyhelminthes", "Sipuncula", "Echiura",
    "Priapulida", "Kinorhyncha", "Loricifera", "Chaetognatha",
}


def rebuild_image_index():
    """从 species_image_library 目录扫描重建图片索引。"""
    print(f"\n[重建图片索引] 扫描: {IMAGE_LIBRARY_DIR}")

    species_images = defaultdict(list)
    fauna_images = {}
    fauna_species_count = 0
    fauna_total = 0

    # 1. 扫描 WoRMS 物种图片 (按门级/物种 组织)
    worms_total = 0
    worms_species_count = 0
    for phylum_dir in sorted(IMAGE_LIBRARY_DIR.iterdir()):
        if not phylum_dir.is_dir():
            continue
        if phylum_dir.name == "中国动物志":
            # 中国动物志 - 新结构: 中国动物志/{Phylum}/{Species}/img_xxx.jpg
            for sub_dir in phylum_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                # 检查 sub_dir 是 Phylum 目录(包含 Species 子目录)
                sub_dirs = [d for d in sub_dir.iterdir() if d.is_dir()]
                if sub_dirs:
                    # 新结构: sub_dir 是 Phylum, sub_dirs 是 Species
                    phylum_name = sub_dir.name
                    for species_dir in sub_dirs:
                        img_files = [f for f in species_dir.iterdir()
                                    if f.is_symlink() or f.is_file()]
                        if img_files:
                            fauna_species_count += 1
                            for img in img_files:
                                species_images[f"中国动物志/{phylum_name}/{species_dir.name}"].append(
                                    str(img.relative_to(PROJECT_ROOT))
                                )
                                fauna_total += 1
                else:
                    # 旧结构: sub_dir 是卷目录,直接包含图片
                    img_files = [f for f in sub_dir.iterdir()
                                if f.is_symlink() or f.is_file()]
                    if img_files:
                        fauna_images[sub_dir.name] = {
                            "phylum": "Unknown",
                            "image_count": len(img_files),
                            "images_dir": str(sub_dir.relative_to(PROJECT_ROOT)),
                        }
            continue

        # WoRMS 物种图片 (phylum/species 结构)
        phylum_name = phylum_dir.name
        for species_dir in phylum_dir.iterdir():
            if not species_dir.is_dir():
                continue
            worms_species_count += 1
            img_files = list(species_dir.iterdir())
            img_files = [f for f in img_files if f.is_symlink() or f.is_file()]
            for img in img_files:
                species_images[f"{phylum_name}/{species_dir.name}"].append(
                    str(img.relative_to(PROJECT_ROOT))
                )
                worms_total += 1

    print(f"  WoRMS 物种: {worms_species_count} 物种, {worms_total} 图片")
    print(f"  中国动物志(按物种): {fauna_species_count} 物种, {fauna_total} 图片")
    print(f"  中国动物志(按卷): {len(fauna_images)} 卷")

    # 2. 补充中国动物志卷的门级映射
    # 从 integrate_oceanus_all.py 导入映射
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
    try:
        from integrate_oceanus_all import FAUNA_SINICA_DEEP_SEA_VOLUMES
        for vol_name, phylum in FAUNA_SINICA_DEEP_SEA_VOLUMES.items():
            if vol_name in fauna_images:
                fauna_images[vol_name]["phylum"] = phylum
    except ImportError:
        print("  ⚠️ 无法导入 FAUNA_SINICA_DEEP_SEA_VOLUMES, 门级保持 Unknown")

    # 3. 构建索引
    image_index = {
        "source": "Oceanus-all WoRMS_papers + 中国动物志",
        "apsma_phyla_count": len(APSMA_PHYLA),
        "worms_species_count": worms_species_count,
        "worms_images_count": worms_total,
        "fauna_volumes_count": len(fauna_images),
        "fauna_images_count": sum(v["image_count"] for v in fauna_images.values()),
        "total_images": worms_total + sum(v["image_count"] for v in fauna_images.values()),
        "worms_species": dict(species_images),
        "fauna_sinica": fauna_images,
    }

    # 按门级统计
    phyla_stats = defaultdict(lambda: {"species": 0, "images": 0})
    for key, imgs in species_images.items():
        phylum = key.split("/")[0]
        phyla_stats[phylum]["species"] += 1
        phyla_stats[phylum]["images"] += len(imgs)
    for vol_name, info in fauna_images.items():
        phylum = info["phylum"]
        phyla_stats[phylum]["species"] += 0  # 卷级,不增加物种数
        phyla_stats[phylum]["images"] += info["image_count"]

    image_index["phyla_distribution"] = {
        p: {"species": s["species"], "images": s["images"]}
        for p, s in sorted(phyla_stats.items(), key=lambda x: -x[1]["images"])
    }

    # 4. 写入索引文件
    index_path = IMAGE_LIBRARY_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(image_index, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 图片索引已写入: {index_path}")
    print(f"     总图片数: {image_index['total_images']}")
    print(f"     门级分布:")
    for p, s in image_index["phyla_distribution"].items():
        print(f"       {p:25s}: {s['species']:5d} 物种, {s['images']:6d} 图片")

    return image_index


def rebuild_knowledge_index():
    """从 species_knowledge_base 目录扫描重建知识库索引。"""
    print(f"\n[重建知识库索引] 扫描: {KNOWLEDGE_BASE_DIR}")

    descriptions = {}
    fauna_descriptions = {}
    structured_desc = {}

    # 1. 扫描 WoRMS 描述 (descriptions/{phylum}/{species}.md)
    desc_dir = KNOWLEDGE_BASE_DIR / "descriptions"
    if desc_dir.exists():
        for phylum_dir in sorted(desc_dir.iterdir()):
            if not phylum_dir.is_dir():
                continue
            phylum_name = phylum_dir.name
            for md_file in phylum_dir.iterdir():
                if not (md_file.is_symlink() or md_file.is_file()):
                    continue
                species_name = md_file.stem.replace("_", " ")
                # 读取前500字符作为摘要
                snippet = ""
                try:
                    target = md_file.resolve() if md_file.is_symlink() else md_file
                    with open(target, encoding="utf-8") as f:
                        content = f.read()
                    snippet = content[:500].replace("\n", " ").strip()
                except Exception:
                    pass
                descriptions[f"{phylum_name}/{species_name}"] = {
                    "path": str(md_file.relative_to(PROJECT_ROOT)),
                    "phylum": phylum_name,
                    "snippet": snippet,
                }

    # 2. 扫描中国动物志描述 (fauna_sinica/{phylum}/{volume}.md)
    fs_dir = KNOWLEDGE_BASE_DIR / "fauna_sinica"
    if fs_dir.exists():
        for phylum_dir in sorted(fs_dir.iterdir()):
            if not phylum_dir.is_dir():
                continue
            phylum_name = phylum_dir.name
            for md_file in phylum_dir.iterdir():
                if not (md_file.is_symlink() or md_file.is_file()):
                    continue
                vol_name = md_file.stem
                snippet = ""
                try:
                    target = md_file.resolve() if md_file.is_symlink() else md_file
                    with open(target, encoding="utf-8") as f:
                        content = f.read()
                    snippet = content[:500].replace("\n", " ").strip()
                except Exception:
                    pass
                fauna_descriptions[vol_name] = {
                    "path": str(md_file.relative_to(PROJECT_ROOT)),
                    "phylum": phylum_name,
                    "snippet": snippet,
                }

    # 3. 扫描结构化描述 (structured/{phylum}/{species}.json)
    structured_dir = KNOWLEDGE_BASE_DIR / "structured"
    if structured_dir.exists():
        for phylum_dir in sorted(structured_dir.iterdir()):
            if not phylum_dir.is_dir():
                continue
            phylum_name = phylum_dir.name
            for json_file in phylum_dir.iterdir():
                if not json_file.is_file():
                    continue
                species_name = json_file.stem.replace("_", " ")
                structured_desc[f"{phylum_name}/{species_name}"] = {
                    "path": str(json_file.relative_to(PROJECT_ROOT)),
                    "phylum": phylum_name,
                }

    print(f"  WoRMS 描述: {len(descriptions)} 物种")
    print(f"  中国动物志描述: {len(fauna_descriptions)} 卷")
    print(f"  结构化描述: {len(structured_desc)} 物种")

    # 3. 按门级统计描述
    desc_phyla = defaultdict(int)
    for key, info in descriptions.items():
        desc_phyla[info["phylum"]] += 1

    # 4. 构建索引
    knowledge_index = {
        "source": "Oceanus-all WoRMS_markdown + 中国动物志 + 结构化描述",
        "apsma_phyla_count": len(APSMA_PHYLA),
        "worms_species_count": len(descriptions),
        "fauna_volumes_count": len(fauna_descriptions),
        "structured_species_count": len(structured_desc),
        "total_descriptions": len(descriptions) + len(fauna_descriptions) + len(structured_desc),
        "descriptions": descriptions,
        "fauna_sinica": fauna_descriptions,
        "structured": structured_desc,
        "descriptions_phyla_distribution": dict(
            sorted(desc_phyla.items(), key=lambda x: -x[1])
        ),
    }

    # 5. 写入索引文件
    index_path = KNOWLEDGE_BASE_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_index, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 知识库索引已写入: {index_path}")
    print(f"     总描述数: {knowledge_index['total_descriptions']}")
    print(f"     门级分布:")
    for p, c in knowledge_index["descriptions_phyla_distribution"].items():
        print(f"       {p:25s}: {c:5d} 描述")

    return knowledge_index


if __name__ == "__main__":
    image_idx = rebuild_image_index()
    knowledge_idx = rebuild_knowledge_index()
    print("\n✅ 索引重建完成!")
    print(f"  图片: {image_idx['total_images']} 张")
    print(f"  描述: {knowledge_idx['total_descriptions']} 个")
