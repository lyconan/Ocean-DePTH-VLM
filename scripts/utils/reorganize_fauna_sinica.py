"""将中国动物志图片从按卷组织改为按物种组织。

解析每个中国动物志 MD 文件,提取物种描述块,找到对应的图片,
将图片符号链接到按物种组织的目录。

输入:
    data/processed/species_image_library/中国动物志/{Volume}/  (按卷)
    Oceanus-all 生物分类书籍/{Volume}/auto/*.md  (物种描述)

输出:
    data/processed/species_image_library/中国动物志/{Phylum}/{Species}/  (按物种)
"""
import os
import re
import sys
import shutil
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
OCEANUS_ROOT = Path("/mnt/oneocean_200TBnas_feitian/Oceanus-all/oceanus_dataset")
FAUNA_SINICA_DIR = OCEANUS_ROOT / "生物分类书籍格式转化路径" / "pdf-md" / "生物分类书籍"
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data/processed/species_image_library"
FS_IMG_DIR = IMAGE_LIBRARY_DIR / "中国动物志"

# 导入中国动物志卷→门级映射
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
from integrate_oceanus_all import FAUNA_SINICA_DEEP_SEA_VOLUMES


def parse_species_from_md(md_path):
    """从中国动物志 MD 文件中解析物种描述块。

    物种描述格式:
        # (N）中文名 学名 Author, Year（图 N）
        ...描述...
        ![](images/xxx.jpg)
        图N 描述
        模式标本产地 ...
        形态 ...
        生物学资料 ...
        地理分布 ...

    Returns:
        list of (species_latin_name, species_chinese_name, [image_filenames])
    """
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    # 匹配物种标题: # (N）中文名 学名 Author, Year（图 N）
    # 或: # N. 中文名 学名 Author, Year
    # 学名格式: Genus species Author, Year
    species_pattern = re.compile(
        r'^#\s*[\(（]?(\d+)[\)）\.．]?\s*'  # 编号
        r'([^\n]+?)\s*$'  # 标题内容
        , re.MULTILINE
    )

    # 学名匹配: Genus species (Author, Year) 或 Genus species Author, Year
    latin_pattern = re.compile(
        r'([A-Z][a-z]+(?:\s+(?:sp\.|subsp\.|var\.)?\s*[a-z]+)?)'  # 学名
        r'\s+(?:[A-Z][a-z]+\s+)?'  # 作者姓
        r'(?:[A-Z][a-z]+(?:\s+et\s+[A-Z][a-z]+)?,)?\s*'  # 作者
        r'(\d{4})'  # 年份
    )

    # 图片引用: ![](images/xxx.jpg)
    image_pattern = re.compile(r'!\[\]\(images/([a-f0-9]+\.jpg)\)')

    species_list = []

    # 按行分割,找到物种标题
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        # 匹配物种标题
        match = species_pattern.match(line)
        if match:
            title = match.group(2)
            # 尝试从标题中提取学名
            latin_match = latin_pattern.search(title)
            if latin_match:
                latin_name = latin_match.group(1).strip()
                # 提取中文名(标题中非拉丁文部分)
                chinese_name = title.replace(latin_match.group(0), '').strip()

                # 收集该物种描述块中的图片
                # 向后扫描,直到下一个物种标题
                block_images = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    if species_pattern.match(next_line):
                        break
                    # 也检查是否是属级标题(# N. 属名)
                    if next_line.startswith('#') and re.search(r'Genus|属', next_line):
                        break
                    for img_match in image_pattern.finditer(next_line):
                        block_images.append(img_match.group(1))
                    j += 1

                if block_images:
                    species_list.append({
                        'latin_name': latin_name,
                        'chinese_name': chinese_name,
                        'images': block_images,
                        'line': i + 1,
                    })
        i += 1

    return species_list


def reorganize_by_species():
    """将中国动物志图片按物种重新组织。"""
    print("=" * 70)
    print("中国动物志按物种组织")
    print("=" * 70)

    total_species = 0
    total_images = 0
    species_by_phylum = defaultdict(int)
    images_by_phylum = defaultdict(int)

    for vol_name, phylum in FAUNA_SINICA_DEEP_SEA_VOLUMES.items():
        vol_dir = FAUNA_SINICA_DIR / vol_name / "auto"
        if not vol_dir.exists():
            print(f"  ⚠️ 跳过(目录不存在): {vol_name[:50]}")
            continue

        # 找到 MD 文件
        md_files = list(vol_dir.glob("*.md"))
        if not md_files:
            print(f"  ⚠️ 跳过(无MD): {vol_name[:50]}")
            continue

        md_path = md_files[0]
        print(f"\n处理: {vol_name[:50]}")
        print(f"  门级: {phylum}")
        print(f"  MD: {md_path.name}")

        # 解析物种
        species_list = parse_species_from_md(md_path)
        print(f"  解析到物种数: {len(species_list)}")

        # 图片目录
        images_dir = vol_dir / "images"
        if not images_dir.exists():
            print(f"  ⚠️ 无图片目录")
            continue

        # 为每个物种创建目录并链接图片
        for sp in species_list:
            latin = sp['latin_name']
            # 清理物种名(移除特殊字符)
            clean_name = re.sub(r'[^\w\s]', '', latin).strip().replace(' ', '_')
            if not clean_name:
                continue

            # 输出目录: 中国动物志/{Phylum}/{Species}/
            # 对于 Other 门级,使用原始卷名
            output_phylum = phylum if phylum != "Other" else "Other"
            species_out_dir = FS_IMG_DIR / output_phylum / clean_name
            species_out_dir.mkdir(parents=True, exist_ok=True)

            # 链接图片
            for img_filename in sp['images']:
                src = images_dir / img_filename
                if not src.exists():
                    continue
                dst = species_out_dir / img_filename
                if dst.exists():
                    continue
                try:
                    os.symlink(src, dst)
                    total_images += 1
                    images_by_phylum[output_phylum] += 1
                except OSError:
                    pass

            if sp['images']:
                total_species += 1
                species_by_phylum[output_phylum] += 1

    print(f"\n{'=' * 70}")
    print(f"✅ 中国动物志按物种组织完成!")
    print(f"  总物种数: {total_species:,}")
    print(f"  总图片数: {total_images:,}")
    print(f"\n按门级分布:")
    for phylum in sorted(species_by_phylum.keys()):
        print(f"  {phylum:25s}: {species_by_phylum[phylum]:4d} 物种, {images_by_phylum[phylum]:5d} 图片")

    return total_species, total_images


def cleanup_old_volume_dirs():
    """清理旧的按卷组织目录(保留新的按物种组织)。"""
    print(f"\n清理旧的按卷目录...")
    removed = 0
    for item in FS_IMG_DIR.iterdir():
        # 跳过门级目录
        if item.is_dir() and item.name in {"Arthropoda", "Mollusca", "Echinodermata",
                                            "Annelida", "Chordata", "Sipuncula",
                                            "Chaetognatha", "Other"}:
            continue
        if item.is_dir():
            shutil.rmtree(item)
            removed += 1
    print(f"  删除 {removed} 个旧卷目录")


if __name__ == "__main__":
    total_species, total_images = reorganize_by_species()
    # 不自动删除旧目录,让用户确认后再删除
    print(f"\n注: 旧的按卷目录仍保留,如需清理请运行 cleanup_old_volume_dirs()")
