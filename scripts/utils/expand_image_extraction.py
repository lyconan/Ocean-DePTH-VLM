"""扩展图片提取: 为每个物种添加更多图片(从 img_006 到 img_020)。

不修改已有数据,只追加缺失的图片。
对每个 WoRMS_papers 目录,如果物种已在 species_image_library 中,
则添加 img_006~img_020 的符号链接。

输入:
    Oceanus-all WoRMS_papers/  (7,556个论文目录)
    data/processed/species_image_library/{Phylum}/{Species}/  (已有 7,514 物种)
    data/raw/worms_inherited/apsma_worms_taxon_subset.csv  (门级映射)

输出:
    data/processed/species_image_library/{Phylum}/{Species}/img_006~img_020.jpg
"""
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
OCEANUS_ROOT = Path("/mnt/oneocean_200TBnas_feitian/Oceanus-all/oceanus_dataset")
WORMS_PAPERS_DIR = OCEANUS_ROOT / "生物分类书籍格式转化路径" / "pdf-md" / "WoRMS_papers"
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data" / "processed" / "species_image_library"
TAXON_CSV = PROJECT_ROOT / "data/raw/worms_inherited/apsma_worms_taxon_subset.csv"

# 扩展参数
NEW_MAX = 20  # 新上限
OLD_MAX = 5   # 旧上限


def parse_species_from_dirname(dirname):
    """从 WoRMS_papers 目录名解析物种名。

    目录名格式: {物种名}_{DOI}
    DOI 通常以 10. 开头
    """
    parts = dirname.split("_10.")
    if len(parts) >= 2:
        species = parts[0]
    else:
        # 备用: 找最后一个 _ 后的部分判断是否是 DOI
        idx = dirname.rfind("_")
        if idx > 0:
            species = dirname[:idx]
        else:
            species = dirname
    return species.replace("_", " ").strip()


def load_taxon_mapping():
    """加载 WoRMS 子集的物种名→门级映射。"""
    print("加载 WoRMS 子集门级映射...")
    mapping = {}
    with open(TAXON_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("taxonomicStatus", "").strip() == "accepted":
                sci_name = row.get("scientificName", "").strip()
                phylum = row.get("phylum", "").strip()
                if sci_name and phylum:
                    mapping[sci_name] = phylum
                    # 也记录属名
                    if " " in sci_name:
                        genus = sci_name.split()[0]
                        if genus not in mapping:
                            mapping[genus] = phylum
    print(f"  加载 {len(mapping):,} 条映射")
    return mapping


def build_species_to_papers_index():
    """构建 species_name → WoRMS_papers 目录路径 的映射。"""
    print("扫描 WoRMS_papers 目录...")
    species_to_papers = {}
    for paper_dir in WORMS_PAPERS_DIR.iterdir():
        if not paper_dir.is_dir():
            continue
        species = parse_species_from_dirname(paper_dir.name)
        if species:
            species_to_papers[species] = paper_dir
    print(f"  扫描到 {len(species_to_papers):,} 个物种论文")
    return species_to_papers


def expand_images(taxon_mapping, species_to_papers):
    """为每个物种扩展图片到 NEW_MAX 张。"""
    print(f"\n扩展图片提取 (从 {OLD_MAX} → {NEW_MAX} 张/物种)...")
    expanded_species = 0
    added_images = 0
    skipped_no_paper = 0
    skipped_already_full = 0

    # 遍历 species_image_library 中的所有物种目录
    for phylum_dir in IMAGE_LIBRARY_DIR.iterdir():
        if not phylum_dir.is_dir() or phylum_dir.name == "中国动物志":
            continue
        phylum = phylum_dir.name

        for species_dir in phylum_dir.iterdir():
            if not species_dir.is_dir():
                continue
            species_name = species_dir.name.replace("_", " ")

            # 检查已有图片数
            existing_imgs = sorted([
                f for f in species_dir.iterdir()
                if f.is_symlink() or f.is_file()
            ])
            n_existing = len(existing_imgs)
            if n_existing >= NEW_MAX:
                skipped_already_full += 1
                continue

            # 找到对应的 WoRMS_papers 目录
            paper_dir = species_to_papers.get(species_name)
            if not paper_dir:
                # 尝试用目录名查找
                paper_dir = species_to_papers.get(species_dir.name)
            if not paper_dir:
                skipped_no_paper += 1
                continue

            # 获取原始图片列表
            auto_dir = paper_dir / "auto"
            if not auto_dir.exists():
                skipped_no_paper += 1
                continue
            images_dir = auto_dir / "images"
            if not images_dir.exists():
                skipped_no_paper += 1
                continue

            all_images = sorted(list(images_dir.glob("*.jpg")) +
                                list(images_dir.glob("*.png")))
            if len(all_images) <= OLD_MAX:
                continue  # 原始图片数不超过 OLD_MAX,无需扩展

            # 添加 img_{OLD_MAX+1} 到 img_{NEW_MAX} 的图片
            # 保留前 OLD_MAX 张不变,从第 OLD_MAX+1 张开始追加
            existing_names = {f.name for f in existing_imgs}
            added = 0
            for j, img in enumerate(all_images[:NEW_MAX]):
                dst_name = f"img_{j+1:03d}{img.suffix}"
                dst = species_dir / dst_name
                if dst_name in existing_names:
                    continue
                try:
                    os.symlink(img, dst)
                    added += 1
                except OSError:
                    pass

            if added > 0:
                expanded_species += 1
                added_images += added

    print(f"  扩展物种数: {expanded_species:,}")
    print(f"  新增图片数: {added_images:,}")
    print(f"  跳过(无论文): {skipped_no_paper:,}")
    print(f"  跳过(已达上限): {skipped_already_full:,}")
    return expanded_species, added_images


def main():
    print("=" * 70)
    print(f"扩展图片提取: {OLD_MAX} → {NEW_MAX} 张/物种")
    print("=" * 70)

    taxon_mapping = load_taxon_mapping()
    species_to_papers = build_species_to_papers_index()

    expanded, added = expand_images(taxon_mapping, species_to_papers)

    print(f"\n✅ 扩展完成!")
    print(f"  扩展物种: {expanded:,}")
    print(f"  新增图片: {added:,}")
    print(f"  总图片数(扩展后): {37_551 + added:,}")


if __name__ == "__main__":
    main()
