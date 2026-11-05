"""整合 Oceanus-all 和中国动物志的物种图片和描述到本项目。

任务:
    E3: 从 Oceanus-all WoRMS_papers 提取 APSMA 物种图片(PDF提取的jpg)
    E4: 从中国动物志提取深海相关卷的图片
    F1: 从 WoRMS_markdown 整合 APSMA 物种描述到知识库
    F2: 从中国动物志MD整合描述

策略:
    使用符号链接避免复制大文件，按物种名/卷名组织目录结构，
    最后合并生成图片检索索引和知识库索引。

源数据:
    /mnt/oneocean_200TBnas_feitian/Oceanus-all/oceanus_dataset/
"""
import os
import json
import shutil
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")

# Oceanus-all 源路径
OCEANUS_ROOT = Path("/mnt/oneocean_200TBnas_feitian/Oceanus-all/oceanus_dataset")
WORMS_PAPERS_DIR = OCEANUS_ROOT / "生物分类书籍格式转化路径" / "pdf-md" / "WoRMS_papers"
WORMS_MD_DIR = OCEANUS_ROOT / "生物分类书籍格式转化路径" / "pdf-md" / "WoRMS_markdown提取"
FAUNA_SINICA_DIR = OCEANUS_ROOT / "生物分类书籍格式转化路径" / "pdf-md" / "生物分类书籍"

# 输出路径
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data" / "processed" / "species_image_library"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "species_knowledge_base"

# APSMA 扩展21门
APSMA_PHYLA = {
    "Porifera", "Echinodermata", "Cnidaria", "Annelida",
    "Arthropoda", "Xenophyophorea", "Mollusca", "Brachiopoda",
    "Hemichordata", "Chordata", "Nematoda", "Foraminifera",
    "Bryozoa", "Ctenophora", "Nemertea",
    "Platyhelminthes", "Sipuncula", "Echiura",
    "Priapulida", "Kinorhyncha", "Loricifera", "Chaetognatha",
}

# 门级到中文动物志卷名的映射
# 用于 E4: 筛选深海相关卷
FAUNA_SINICA_DEEP_SEA_VOLUMES = {
    # Echinodermata
    "中国动物志 无脊椎动物 第40卷 棘皮动物门 蛇尾纲": "Echinodermata",
    # Annelida
    "中国动物志 环节动物门 多毛纲 Ⅰ叶须虫目【无封面、版权页】": "Annelida",
    "中国动物志 环节动物门 蛭纲": "Annelida",
    # Arthropoda
    "中国动物志 节肢动物门 甲壳动物亚门 糠虾目": "Arthropoda",
    "中国动物志 节肢动物门 甲壳动物亚门 软甲纲 十足目 束腹蟹科 溪蟹科": "Arthropoda",
    "中国动物志 节肢动物门 甲壳纲 淡水桡足类": "Arthropoda",
    "中国动物志 节肢动物门 甲壳纲 淡水枝角类": "Arthropoda",
    "中国动物志 无脊椎动物 第30卷 节肢动物门 甲壳动物亚门 短尾次目 海洋低等蟹类": "Arthropoda",
    "中国动物志 无脊椎动物 第36卷 甲壳动物亚门 十足目 匙指虾科": "Arthropoda",
    "中国动物志 无脊椎动物 第四十一卷 甲壳动物亚门 端足目 钩虾亚目（一）FAUNA SINICA - INVERTEBRATA Vol. 41 Crustacea Amphipoda Gammaridea I": "Arthropoda",
    "中国动物志 无脊椎动物 第四十二卷 甲壳动物亚门 蔓足下纲 围胸总目FAUNA SINICA - INVERTEBRATA Vol. 42 Crustacea Amphipoda Thoracica": "Arthropoda",
    "中国动物志 无脊椎动物 第四十三卷 甲壳动物亚门 端足目 钩虾亚目（二）FAUNA SINICA - INVERTEBRATA Vol. 41 Crustacea Amphipoda Gammaridea II": "Arthropoda",
    # Mollusca
    "中国动物志 第56卷 Vol.56 无脊椎动物 软体动物门 腹足纲 凤螺总科 玉螺总科": "Mollusca",
    "中国动物志 软体动物门 腹足纲 肺螺亚纲 柄眼目 烟管螺科": "Mollusca",
    "中国动物志 软体动物门 双壳纲 帘蛤科": "Mollusca",
    "中国动物志 软体动物门 头足纲": "Mollusca",
    "中国动物志 无脊椎动物 第31卷 软体动物门 双壳纲 珍珠贝亚目": "Mollusca",
    "中国动物志 无脊椎动物 第34卷 软体动物门 腹足纲 鹑螺总科": "Mollusca",
    "中国动物志 无脊椎动物 第37卷 软体动物门 腹足纲 巴蜗牛科": "Mollusca",
    "中国动物志 无脊椎动物 第48卷 软体动物门 满月蛤总科": "Mollusca",
    "中国动物志 无脊椎动物 第55卷 软体动物门 腹足纲，芋螺科": "Mollusca",
    # Chordata (硬骨鱼纲)
    "中国动物志 硬骨鱼纲 鲟形目 海鲢目 鲱形目 鼠鱚目": "Chordata",
    "中国动物志 硬骨鱼纲、银汉鱼目、鳉形目、颌针鱼目、蛇鳚目、鳕形目": "Chordata",
    # Sipuncula + Echiura
    "中国动物志 无脊椎动物 第46卷 星虫动物门 螠虫动物门 Invertebrata Vol. 46 Sipuncula Echiura": "Sipuncula",
    # Chaetognatha
    "中国动物志 无脊椎动物 第三十八卷 毛颚动物门 箭虫纲": "Chaetognatha",
    # 其他无脊椎动物 (Myxozoa 现归 Cnidaria)
    "中国动物志 无脊椎动物 第十五卷 粘体动物门 粘孢子纲FAUNA SINICA - INVERTEBRATA Vol. 15 Myxosporea": "Other",
}


# ============ WoRMS 文件名解析辅助 ============

def parse_species_from_dirname(dirname: str) -> Dict:
    """从 WoRMS_papers 目录名解析物种名和 DOI。

    目录名格式: "{物种名}_{DOI}"
    例如: "Abatus shackletoni_10.5962_bhl.title.12579"
    """
    # 分割物种名和 DOI (最后一个 _10.xxx 之前都是物种名)
    match = re.match(r'^(.+?)_(10\..+)$', dirname)
    if match:
        species_name = match.group(1)
        doi = match.group(2)
    else:
        species_name = dirname
        doi = ""

    return {
        "species_name": species_name,
        "doi": doi,
    }


def is_apsma_species(species_name: str, phyla_mapping: Dict[str, str] = None) -> Tuple[bool, str]:
    """检查物种名是否属于 APSMA 门级。

    Args:
        species_name: 物种名(如 "Holothuria scabra")
        phyla_mapping: 可选的 物种名→门级 映射字典

    Returns:
        (is_match, matched_phylum)
    """
    # 先查精确映射
    if phyla_mapping and species_name in phyla_mapping:
        phylum = phyla_mapping[species_name]
        return (phylum in APSMA_PHYLA, phylum)

    # 通过属名前缀粗略判断
    name_lower = species_name.lower()
    # 常见深海属名(粗略匹配)
    # 注: 这是近似匹配, 精确的映射需要通过 WoRMS API
    return (False, "")


# ============ 任务E3: 从 WoRMS_papers 提取图片 ============

def extract_worms_paper_images(
    papers_dir: Path,
    output_dir: Path,
    apsma_taxon: Dict[str, str] = None,
    use_symlink: bool = True,
    max_per_species: int = 5,
) -> Dict:
    """E3: 从 WoRMS_papers 提取 APSMA 物种图片。

    Args:
        papers_dir: WoRMS_papers 目录
        output_dir: 输出目录
        apsma_taxon: 物种名→门级映射 (来自 WoRMS 子集筛选结果)
        use_symlink: 是否用符号链接(避免复制)
        max_per_species: 每物种最多保留几张图

    Returns:
        species_images: {物种名: [图片路径列表]}
    """
    print(f"\n[E3] 从 WoRMS_papers 提取图片...")
    print(f"  源: {papers_dir}")
    print(f"  输出: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    species_images: Dict[str, List[str]] = defaultdict(list)
    total_extracted = 0
    total_skipped = 0

    # 遍历所有物种目录
    all_dirs = [d for d in papers_dir.iterdir() if d.is_dir()]
    print(f"  共 {len(all_dirs)} 个物种目录")

    for i, species_dir in enumerate(all_dirs):
        if i % 500 == 0:
            print(f"  进度: {i}/{len(all_dirs)} (已提取 {total_extracted}, 跳过 {total_skipped})")

        # 解析物种名
        info = parse_species_from_dirname(species_dir.name)
        species_name = info["species_name"]

        # 查找图片目录
        auto_dir = species_dir / "auto"
        images_dir = auto_dir / "images"
        if not images_dir.exists():
            total_skipped += 1
            continue

        # 获取所有图片
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            total_skipped += 1
            continue

        # 确定门级
        phylum = "Unknown"
        if apsma_taxon:
            # 精确匹配物种名
            if species_name in apsma_taxon:
                phylum = apsma_taxon[species_name]
            else:
                # 尝试匹配属名(取第一个单词)
                genus = species_name.split()[0] if " " in species_name else species_name
                if genus in apsma_taxon:
                    phylum = apsma_taxon[genus]

        # 限制每物种图片数
        image_files = image_files[:max_per_species]

        # 输出目录: {phylum}/{species_name}/
        # 用符号链接或复制
        species_out_dir = output_dir / phylum / species_name.replace(" ", "_")
        species_out_dir.mkdir(parents=True, exist_ok=True)

        for j, img in enumerate(image_files):
            dst = species_out_dir / f"img_{j+1:03d}{img.suffix}"
            if dst.exists():
                continue
            if use_symlink:
                try:
                    os.symlink(img, dst)
                except OSError:
                    # 符号链接失败则复制
                    shutil.copy2(img, dst)
            else:
                shutil.copy2(img, dst)
            species_images[species_name].append(str(dst.relative_to(PROJECT_ROOT)))
            total_extracted += 1

    print(f"\n[E3] 完成: 提取 {total_extracted} 张图片, 覆盖 {len(species_images)} 个物种")
    return dict(species_images)


# ============ 任务F1: 从 WoRMS_markdown 整合描述 ============

def integrate_worms_descriptions(
    md_dir: Path,
    output_dir: Path,
    apsma_taxon: Dict[str, str] = None,
) -> Dict:
    """F1: 从 WoRMS_markdown 整合 APSMA 物种描述。

    Args:
        md_dir: WoRMS_markdown提取 目录
        output_dir: 输出目录 (descriptions/)
        apsma_taxon: 物种名→门级映射

    Returns:
        species_descriptions: {物种名: {path, phylum, snippet}}
    """
    print(f"\n[F1] 整合 WoRMS 物种描述 MD...")
    print(f"  源: {md_dir}")
    print(f"  输出: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    species_descriptions: Dict[str, Dict] = {}
    total = 0

    all_mds = list(md_dir.glob("*.md"))
    print(f"  共 {len(all_mds)} 个 MD 文件")

    for i, md_file in enumerate(all_mds):
        if i % 1000 == 0:
            print(f"  进度: {i}/{len(all_mds)}")

        # 解析物种名
        info = parse_species_from_dirname(md_file.stem)
        species_name = info["species_name"]

        # 确定门级
        phylum = "Unknown"
        if apsma_taxon:
            if species_name in apsma_taxon:
                phylum = apsma_taxon[species_name]
            else:
                genus = species_name.split()[0] if " " in species_name else species_name
                if genus in apsma_taxon:
                    phylum = apsma_taxon[genus]

        # 创建门级子目录
        phylum_dir = output_dir / phylum
        phylum_dir.mkdir(parents=True, exist_ok=True)

        # 复制 MD 文件
        out_md = phylum_dir / f"{species_name.replace(' ', '_')}.md"

        # 用符号链接
        if not out_md.exists():
            try:
                os.symlink(md_file, out_md)
            except OSError:
                shutil.copy2(md_file, out_md)

        # 读取前500字符作为摘要
        try:
            with open(md_file, encoding="utf-8") as f:
                content = f.read()
            snippet = content[:500].replace("\n", " ").strip()
        except Exception:
            snippet = ""

        species_descriptions[species_name] = {
            "path": str(out_md.relative_to(PROJECT_ROOT)),
            "phylum": phylum,
            "snippet": snippet,
        }
        total += 1

    print(f"\n[F1] 完成: 整合 {total} 个物种描述")
    return species_descriptions


# ============ 任务E4: 从中国动物志提取深海相关图片 ============

def extract_fauna_sinica_images(
    fauna_dir: Path,
    output_dir: Path,
    use_symlink: bool = True,
) -> Dict:
    """E4: 从中国动物志提取深海相关卷的图片。

    Args:
        fauna_dir: 生物分类书籍 PDF→MD 输出目录
        output_dir: 输出目录
        use_symlink: 是否用符号链接

    Returns:
        volume_images: {卷名: {phylum, image_count, images: [路径]}}
    """
    print(f"\n[E4] 从中国动物志提取深海相关图片...")
    print(f"  源: {fauna_dir}")
    print(f"  输出: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    volume_images: Dict[str, Dict] = {}
    total_images = 0

    for volume_name, phylum in FAUNA_SINICA_DEEP_SEA_VOLUMES.items():
        # 查找该卷的目录
        volume_dir = fauna_dir / volume_name
        if not volume_dir.exists():
            print(f"  ⚠️ 未找到: {volume_name}")
            continue

        # 查找图片目录
        images_dir = volume_dir / "auto" / "images"
        if not images_dir.exists():
            print(f"  ⚠️ 无图片目录: {volume_name}")
            continue

        # 获取所有图片
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            continue

        # 输出目录: 中国动物志/{卷名}/
        # 用符号链接
        volume_out = output_dir / "中国动物志" / volume_name
        volume_out.mkdir(parents=True, exist_ok=True)

        linked_count = 0
        for img in image_files:
            dst = volume_out / img.name
            if dst.exists():
                continue
            if use_symlink:
                try:
                    os.symlink(img, dst)
                    linked_count += 1
                except OSError:
                    shutil.copy2(img, dst)
                    linked_count += 1
            else:
                shutil.copy2(img, dst)
                linked_count += 1

        total_images += linked_count
        volume_images[volume_name] = {
            "phylum": phylum,
            "image_count": linked_count,
            "images_dir": str(volume_out.relative_to(PROJECT_ROOT)),
        }
        print(f"  ✅ {volume_name}: {linked_count} 张图 ({phylum})")

    print(f"\n[E4] 完成: 共 {len(volume_images)} 卷, {total_images} 张图片")
    return volume_images


# ============ 任务F2: 从中国动物志MD整合描述 ============

def integrate_fauna_sinica_descriptions(
    fauna_dir: Path,
    output_dir: Path,
    use_symlink: bool = True,
) -> Dict:
    """F2: 从中国动物志 MD 整合描述。

    Args:
        fauna_dir: 生物分类书籍 PDF→MD 输出目录
        output_dir: 输出目录 (fauna_sinica/)
        use_symlink: 是否用符号链接

    Returns:
        volume_descriptions: {卷名: {path, phylum, snippet}}
    """
    print(f"\n[F2] 整合中国动物志 MD 描述...")
    print(f"  源: {fauna_dir}")
    print(f"  输出: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    volume_descriptions: Dict[str, Dict] = {}

    for volume_name, phylum in FAUNA_SINICA_DEEP_SEA_VOLUMES.items():
        volume_dir = fauna_dir / volume_name
        if not volume_dir.exists():
            continue

        # 查找 MD 文件
        auto_dir = volume_dir / "auto"
        md_files = list(auto_dir.glob("*.md")) + list(volume_dir.glob("*.md"))

        if not md_files:
            continue

        # 创建门级子目录
        phylum_dir = output_dir / phylum
        phylum_dir.mkdir(parents=True, exist_ok=True)

        for md_file in md_files:
            out_md = phylum_dir / f"{volume_name}.md"
            if out_md.exists():
                continue
            if use_symlink:
                try:
                    os.symlink(md_file, out_md)
                except OSError:
                    shutil.copy2(md_file, out_md)
            else:
                shutil.copy2(md_file, out_md)

            # 读取前500字符作为摘要
            try:
                with open(md_file, encoding="utf-8") as f:
                    content = f.read()
                snippet = content[:500].replace("\n", " ").strip()
            except Exception:
                snippet = ""

            volume_descriptions[volume_name] = {
                "path": str(out_md.relative_to(PROJECT_ROOT)),
                "phylum": phylum,
                "snippet": snippet,
            }
            break  # 每卷只取第一个 MD 文件

    print(f"\n[F2] 完成: {len(volume_descriptions)} 卷描述")
    return volume_descriptions


# ============ 任务E5+F4: 合并索引 ============

def build_combined_index(
    species_images: Dict[str, List[str]],
    species_descriptions: Dict[str, Dict],
    fauna_images: Dict[str, Dict],
    fauna_descriptions: Dict[str, Dict],
    image_library_dir: Path,
    knowledge_base_dir: Path,
):
    """E5+F4: 合并构建图片检索索引 + 知识库索引。"""
    print(f"\n[E5+F4] 构建合并索引...")

    # 1. 图片检索索引
    image_index = {
        "source": "Oceanus-all WoRMS_papers + 中国动物志",
        "apsma_phyla_count": len(APSMA_PHYLA),
        "species_count": len(species_images),
        "fauna_volumes_count": len(fauna_images),
        "total_images": sum(len(v) for v in species_images.values()) +
                       sum(v["image_count"] for v in fauna_images.values()),
        "species": species_images,
        "fauna_sinica": fauna_images,
    }

    image_index_path = image_library_dir / "index.json"
    with open(image_index_path, "w", encoding="utf-8") as f:
        json.dump(image_index, f, ensure_ascii=False, indent=2)
    print(f"  图片索引: {image_index_path} ({image_index['total_images']} 张)")

    # 2. 知识库索引
    knowledge_index = {
        "source": "Oceanus-all WoRMS_markdown + 中国动物志",
        "apsma_phyla_count": len(APSMA_PHYLA),
        "worms_species_count": len(species_descriptions),
        "fauna_volumes_count": len(fauna_descriptions),
        "descriptions": species_descriptions,
        "fauna_sinica": fauna_descriptions,
    }

    knowledge_index_path = knowledge_base_dir / "index.json"
    with open(knowledge_index_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_index, f, ensure_ascii=False, indent=2)
    print(f"  知识库索引: {knowledge_index_path}")

    return image_index, knowledge_index


# ============ 主函数 ============

def load_apsma_taxon_mapping() -> Dict[str, str]:
    """从 WoRMS 子集 CSV 加载 物种名→门级 映射。"""
    taxon_csv = PROJECT_ROOT / "data" / "raw" / "worms_inherited" / "apsma_worms_taxon_subset.csv"
    if not taxon_csv.exists():
        print(f"⚠️ WoRMS 子集不存在: {taxon_csv}")
        return {}

    import csv
    mapping = {}
    with open(taxon_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("scientificName", "").strip()
            phylum = row.get("phylum", "").strip()
            if name and phylum:
                mapping[name] = phylum
                # 也添加属名(第一个单词)映射
                if " " in name:
                    genus = name.split()[0]
                    if genus not in mapping:
                        mapping[genus] = phylum

    print(f"加载 WoRMS 子集映射: {len(mapping)} 个名称")
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="整合 Oceanus-all 和中国动物志的物种图片和描述"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["E3", "F1", "E4", "F2", "E5F4"],
        help="执行的任务列表",
    )
    args = parser.parse_args()

    # 加载 WoRMS 子集映射
    apsma_taxon = load_apsma_taxon_mapping()

    # 任务E3: WoRMS_papers 图片提取
    species_images = {}
    if "E3" in args.tasks:
        species_images = extract_worms_paper_images(
            papers_dir=WORMS_PAPERS_DIR,
            output_dir=IMAGE_LIBRARY_DIR,
            apsma_taxon=apsma_taxon,
            use_symlink=True,
            max_per_species=5,
        )

    # 任务F1: WoRMS_markdown 描述整合
    species_descriptions = {}
    if "F1" in args.tasks:
        species_descriptions = integrate_worms_descriptions(
            md_dir=WORMS_MD_DIR,
            output_dir=KNOWLEDGE_BASE_DIR / "descriptions",
            apsma_taxon=apsma_taxon,
        )

    # 任务E4: 中国动物志图片
    fauna_images = {}
    if "E4" in args.tasks:
        fauna_images = extract_fauna_sinica_images(
            fauna_dir=FAUNA_SINICA_DIR,
            output_dir=IMAGE_LIBRARY_DIR,
            use_symlink=True,
        )

    # 任务F2: 中国动物志描述
    fauna_descriptions = {}
    if "F2" in args.tasks:
        fauna_descriptions = integrate_fauna_sinica_descriptions(
            fauna_dir=FAUNA_SINICA_DIR,
            output_dir=KNOWLEDGE_BASE_DIR / "fauna_sinica",
            use_symlink=True,
        )

    # 任务E5+F4: 合并索引
    if "E5F4" in args.tasks:
        build_combined_index(
            species_images=species_images,
            species_descriptions=species_descriptions,
            fauna_images=fauna_images,
            fauna_descriptions=fauna_descriptions,
            image_library_dir=IMAGE_LIBRARY_DIR,
            knowledge_base_dir=KNOWLEDGE_BASE_DIR,
        )

    print("\n✅ 所有整合任务完成!")
