"""
生物图集 PDF 解析器 — 提取图像并按分类组织为检索库

功能:
  1. 解析 APSMA 图集 PDF (7卷) — 按形态型ID提取图像
  2. 解析中文图集 PDF (西太平洋深渊底栖生物) — 按章节/物种提取图像
  3. 解析北太平洋图集 PDF (APSMA完整版) — 按形态型ID提取图像
  4. 按分类层级组织输出目录结构
  5. 生成 JSON 检索索引文件

输出目录结构:
  data/processed/atlas_lookup/
  ├── index.json                    # 完整检索索引
  ├── APSMA/
  │   ├── Porifera/
  │   │   ├── DES_002/
  │   │   │   ├── DES_002_001.png
  │   │   │   ├── DES_002_002.png
  │   │   │   └── ...
  │   │   ├── HEX_001/
  │   │   └── ...
  │   ├── Echinodermata/
  │   └── ...
  ├── 西太平洋图集/
  │   ├── Porifera_多孔动物门/
  │   └── ...
  └── 北太平洋图集/
      └── ...

用法:
  # 解析所有图集
  .venv/bin/python scripts/utils/parse_atlas_pdfs.py --all

  # 只解析 APSMA
  .venv/bin/python scripts/utils/parse_atlas_pdfs.py --apsma

  # 只解析中文图集
  .venv/bin/python scripts/utils/parse_atlas_pdfs.py --chinese

  # 指定输出目录
  .venv/bin/python scripts/utils/parse_atlas_pdfs.py --all --output data/processed/atlas_lookup

  # 查询检索库
  .venv/bin/python scripts/utils/parse_atlas_pdfs.py --query "Porifera"
  .venv/bin/python scripts/utils/parse_atlas_pdfs.py --query "DES_002"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- APSMA PDF 解析 ----

# APSMA 卷名到门类别的映射
APSMA_VOLUME_MAP = {
    "APSMA_1_Porifera_DES_HEX_v1.pdf": {
        "phylum": "Porifera",
        "phylum_cn": "多孔动物门",
        "classes": ["Demospongiae", "Hexactinellida"],
        "class_codes": ["DES", "HEX"],
    },
    "APSMA_2a_Cnidaria_ACT_v1.pdf": {
        "phylum": "Cnidaria",
        "phylum_cn": "刺胞动物门",
        "classes": ["Actiniaria"],
        "class_codes": ["ACT"],
    },
    "APSMA_2b_Cnidaria_ALC_ANT_COR_HYD_SCY_v1.pdf": {
        "phylum": "Cnidaria",
        "phylum_cn": "刺胞动物门",
        "classes": ["Alcyonacea", "Antipatharia", "Scleractinia", "Hydrozoa", "Scyphozoa"],
        "class_codes": ["ALC", "ANT", "COR", "HYD", "SCY"],
    },
    "APSMA_3a_Echinodermata_AST_CRI_URC_OPH_v1.pdf": {
        "phylum": "Echinodermata",
        "phylum_cn": "棘皮动物门",
        "classes": ["Asteroidea", "Crinoidea", "Echinoidea", "Ophiuroidea"],
        "class_codes": ["AST", "CRI", "URC", "OPH"],
    },
    "APSMA_3b_Echinodermata_HOL_v1.pdf": {
        "phylum": "Echinodermata",
        "phylum_cn": "棘皮动物门",
        "classes": ["Holothuroidea"],
        "class_codes": ["HOL"],
    },
    "APSMA_4_Arthropoda_ART_DEC_v1.pdf": {
        "phylum": "Arthropoda",
        "phylum_cn": "节肢动物门",
        "classes": ["Arthropoda_other", "Decapoda"],
        "class_codes": ["ART", "DEC"],
    },
    "APSMA_5_other_v1.pdf": {
        "phylum": "Other",
        "phylum_cn": "其他门类",
        "classes": ["Xenophyophorea", "Mollusca", "Annelida", "Brachiopoda", "Foraminifera"],
        "class_codes": ["XEN", "MOL", "ANN", "BRA", "FOR"],
    },
}

# 形态型ID前缀到纲的映射
def _morphotype_to_class(morphotype_id: str, volume_info: dict) -> str:
    """根据形态型ID前缀推断所属纲。"""
    prefix = morphotype_id.split("_")[0] if "_" in morphotype_id else ""
    codes = volume_info.get("class_codes", [])
    classes = volume_info.get("classes", [])
    for code, cls in zip(codes, classes):
        if prefix == code:
            return cls
    # 对于 POR_ 前缀 (Porifera 但不属于 DES/HEX)
    if prefix == "POR":
        return "Porifera_unclassified"
    return "unclassified"


def parse_apsma_pdf(
    pdf_path: str,
    output_dir: str,
    min_image_size: int = 5000,
) -> List[Dict]:
    """
    解析单个 APSMA PDF,提取图像并按形态型组织。

    APSMA PDF 结构:
      - 每个形态型占1-2页
      - 页面第一行通常是形态型ID (如 DES_002)
      - 每页包含 2-7 张原位图像
      - 图像下方有文字描述 (Morphology / Remarks / Distribution)
    """
    import fitz

    pdf_name = os.path.basename(pdf_path)
    volume_info = APSMA_VOLUME_MAP.get(pdf_name, {})
    phylum = volume_info.get("phylum", "Unknown")
    phylum_cn = volume_info.get("phylum_cn", "未知")

    doc = fitz.open(pdf_path)
    entries = []
    current_morphotype = None
    current_class = None
    current_description = {}
    current_images = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text().strip()
        images = page.get_images(full=True)

        # 检测分类标题页 (如 "Phylum Porifera / Class Demospongiae")
        phylum_match = re.search(r"Phylum\s+(\w+)", text)
        class_match = re.search(r"Class\s+(\w+)", text)

        if phylum_match and len(images) <= 1:
            # 这是分类标题页,不是形态型页
            if current_morphotype and current_images:
                # 保存前一个形态型
                entry = _save_morphotype_entry(
                    current_morphotype, current_class, phylum, phylum_cn,
                    current_description, current_images, output_dir, doc, "APSMA"
                )
                entries.append(entry)
                current_images = []
                current_description = {}

            if class_match:
                current_class = class_match.group(1)
            continue

        # 检测形态型ID (如 DES_002, HEX_030, POR_014)
        morph_match = re.match(r"^([A-Z]{3}_\d{3})", text)

        if morph_match:
            # 新形态型开始 — 保存前一个
            if current_morphotype and current_images:
                entry = _save_morphotype_entry(
                    current_morphotype, current_class, phylum, phylum_cn,
                    current_description, current_images, output_dir, doc, "APSMA"
                )
                entries.append(entry)

            current_morphotype = morph_match.group(1)
            current_class = _morphotype_to_class(current_morphotype, volume_info)

            # 提取描述信息
            desc = _extract_apsma_description(text)
            current_description = desc
            current_images = []

        # 收集当前页面的图像
        for img_info in images:
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                img_ext = base["ext"]

                # 跳过过小的图像 (logo/icon等)
                if len(img_bytes) < min_image_size:
                    continue

                current_images.append({
                    "xref": xref,
                    "ext": img_ext,
                    "bytes": img_bytes,
                    "width": base["width"],
                    "height": base["height"],
                    "page": page_idx + 1,
                })
            except Exception:
                continue

    # 保存最后一个形态型
    if current_morphotype and current_images:
        entry = _save_morphotype_entry(
            current_morphotype, current_class, phylum, phylum_cn,
            current_description, current_images, output_dir, doc, "APSMA"
        )
        entries.append(entry)

    doc.close()
    print(f"[OK] {pdf_name}: 提取 {len(entries)} 个形态型")
    return entries


def _extract_apsma_description(text: str) -> Dict:
    """从 APSMA 形态型页面文本中提取描述信息。"""
    desc = {}
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Morphology:"):
            desc["morphology"] = line.replace("Morphology:", "").strip()
        elif line.startswith("Remarks:"):
            desc["remarks"] = line.replace("Remarks:", "").strip()
        elif line.startswith("Distribution:"):
            desc["distribution"] = line.replace("Distribution:", "").strip()
        elif line.startswith("Depth:"):
            desc["depth"] = line.replace("Depth:", "").strip()
        elif line.startswith("Size:"):
            desc["size"] = line.replace("Size:", "").strip()
        elif line.startswith("Similar:"):
            desc["similar"] = line.replace("Similar:", "").strip()
    # 整体文本也保留
    desc["full_text"] = text
    return desc


def _save_morphotype_entry(
    morphotype_id: str,
    class_name: str,
    phylum: str,
    phylum_cn: str,
    description: Dict,
    images: List[Dict],
    output_dir: str,
    doc: Any,
    source: str,
) -> Dict:
    """保存形态型图像到目录,返回索引条目。"""
    # 构建输出路径: APSMA/Phylum/MORPH_ID/
    safe_phylum = phylum.replace(" ", "_")
    safe_class = class_name.replace(" ", "_")
    morph_dir = os.path.join(output_dir, source, safe_phylum, morphotype_id)
    os.makedirs(morph_dir, exist_ok=True)

    saved_images = []
    for i, img in enumerate(images):
        filename = f"{morphotype_id}_{i+1:03d}.{img['ext']}"
        filepath = os.path.join(morph_dir, filename)
        with open(filepath, "wb") as f:
            f.write(img["bytes"])
        saved_images.append({
            "filename": filename,
            "path": filepath,
            "width": img["width"],
            "height": img["height"],
            "page": img["page"],
        })

    return {
        "morphotype_id": morphotype_id,
        "phylum": phylum,
        "phylum_cn": phylum_cn,
        "class": class_name,
        "source": source,
        "description": {k: v for k, v in description.items() if k != "full_text"},
        "full_text": description.get("full_text", ""),
        "image_count": len(saved_images),
        "images": saved_images,
        "directory": morph_dir,
    }


# ---- 中文图集 PDF 解析 ----

# 中文图集的章节 → 门类映射
CN_CHAPTER_MAP = {
    "多孔动物": "Porifera",
    "刺胞动物": "Cnidaria",
    "环节动物": "Annelida",
    "节肢动物": "Arthropoda",
    "棘皮动物": "Echinodermata",
    "软体动物": "Mollusca",
    "腕足动物": "Brachiopoda",
    "脊索动物": "Chordata",
    "有孔虫": "Foraminifera",
    "异胞藻": "Xenophyophorea",
}


def parse_chinese_pdf(
    pdf_path: str,
    output_dir: str,
    min_image_size: int = 5000,
) -> List[Dict]:
    """
    解析中文图集 PDF。

    中文图集是扫描版,文本质量较差,使用以下策略:
      1. 提取所有图像
      2. 尝试从 OCR 文本中识别门/纲/物种名
      3. 如果文本无法解析,按页码范围和图像相似度聚类
    """
    import fitz

    doc = fitz.open(pdf_path)
    entries = []
    current_phylum = "Unknown"
    current_phylum_cn = "未知"
    current_section = ""
    current_images = []
    current_page_start = 0

    # 先扫描全文,找到章节分界点
    section_pages = {}
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text().strip()
        for cn_name, en_name in CN_CHAPTER_MAP.items():
            if cn_name in text and ("门" in text or "动物" in text):
                # 这是章节标题页
                section_pages[page_idx] = (cn_name, en_name)
                break

    # 如果找不到章节,使用全文作为一节
    if not section_pages:
        section_pages[0] = ("全部", "All")

    # 按章节提取图像
    sorted_sections = sorted(section_pages.items())
    for sec_idx, (sec_page, (cn_name, en_name)) in enumerate(sorted_sections):
        # 章节结束页
        if sec_idx + 1 < len(sorted_sections):
            end_page = sorted_sections[sec_idx + 1][0]
        else:
            end_page = len(doc)

        phylum_dir = os.path.join(output_dir, "西太平洋图集", en_name)
        os.makedirs(phylum_dir, exist_ok=True)

        sec_images = []
        for page_idx in range(sec_page, end_page):
            page = doc[page_idx]
            images = page.get_images(full=True)
            text = page.get_text().strip()

            # 尝试识别物种名
            species_name = ""
            for line in text.split("\n"):
                line = line.strip()
                # 中文图集中的物种名通常在图像旁边
                if len(line) > 2 and len(line) < 50 and not line.startswith("西太平洋"):
                    if any(c in line for c in "海绵珊瑚海参海葵虾蟹螺贝虫"):
                        species_name = line
                        break

            for img_info in images:
                xref = img_info[0]
                try:
                    base = doc.extract_image(xref)
                    img_bytes = base["image"]
                    if len(img_bytes) < min_image_size:
                        continue

                    page_label = f"p{page_idx+1:03d}"
                    img_filename = f"{en_name}_{page_label}_{xref}.{base['ext']}"
                    filepath = os.path.join(phylum_dir, img_filename)

                    with open(filepath, "wb") as f:
                        f.write(img_bytes)

                    sec_images.append({
                        "filename": img_filename,
                        "path": filepath,
                        "width": base["width"],
                        "height": base["height"],
                        "page": page_idx + 1,
                        "species_hint": species_name,
                    })
                except Exception:
                    continue

        if sec_images:
            entry = {
                "phylum": en_name,
                "phylum_cn": cn_name,
                "source": "西太平洋图集",
                "image_count": len(sec_images),
                "images": sec_images,
                "directory": phylum_dir,
                "page_range": f"{sec_page+1}-{end_page}",
            }
            entries.append(entry)
            print(f"[OK] {cn_name} ({en_name}): {len(sec_images)} 张图像")

    doc.close()
    return entries


# ---- 北太平洋图集 PDF 解析 (APSMA 完整版) ----

def parse_north_pacific_pdf(
    pdf_path: str,
    output_dir: str,
    min_image_size: int = 5000,
) -> List[Dict]:
    """
    解析北太平洋图集 PDF (APSMA完整版253页)。
    结构与 APSMA 分卷相同,但包含所有门类。
    """
    import fitz

    doc = fitz.open(pdf_path)
    entries = []
    current_morphotype = None
    current_phylum = "Unknown"
    current_phylum_cn = "未知"
    current_class = "Unknown"
    current_description = {}
    current_images = []

    # 从所有 volume info 中收集 class_code 映射
    all_class_codes = {}
    for vol_info in APSMA_VOLUME_MAP.values():
        for code, cls in zip(vol_info.get("class_codes", []),
                             vol_info.get("classes", [])):
            all_class_codes[code] = (cls, vol_info["phylum"], vol_info["phylum_cn"])

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text().strip()
        images = page.get_images(full=True)

        # 检测门标题页
        phylum_match = re.search(r"Phylum\s+(\w+)", text)
        class_match = re.search(r"Class\s+(\w+)", text)

        if phylum_match and len(images) <= 1:
            if current_morphotype and current_images:
                entry = _save_morphotype_entry(
                    current_morphotype, current_class, current_phylum,
                    current_phylum_cn, current_description, current_images,
                    output_dir, doc, "北太平洋图集"
                )
                entries.append(entry)
                current_images = []
                current_description = {}

            detected_phylum = phylum_match.group(1)
            # 在映射中查找中文名
            for vol_info in APSMA_VOLUME_MAP.values():
                if vol_info["phylum"] == detected_phylum:
                    current_phylum_cn = vol_info["phylum_cn"]
                    break
            else:
                current_phylum_cn = detected_phylum

            current_phylum = detected_phylum
            if class_match:
                current_class = class_match.group(1)
            continue

        # 检测形态型ID
        morph_match = re.match(r"^([A-Z]{3}_\d{3})", text)

        if morph_match:
            if current_morphotype and current_images:
                entry = _save_morphotype_entry(
                    current_morphotype, current_class, current_phylum,
                    current_phylum_cn, current_description, current_images,
                    output_dir, doc, "北太平洋图集"
                )
                entries.append(entry)

            current_morphotype = morph_match.group(1)
            prefix = current_morphotype.split("_")[0]
            if prefix in all_class_codes:
                current_class, _, _ = all_class_codes[prefix]
            else:
                current_class = "unclassified"

            desc = _extract_apsma_description(text)
            current_description = desc
            current_images = []

        # 收集图像
        for img_info in images:
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                if len(img_bytes) < min_image_size:
                    continue
                current_images.append({
                    "xref": xref,
                    "ext": base["ext"],
                    "bytes": img_bytes,
                    "width": base["width"],
                    "height": base["height"],
                    "page": page_idx + 1,
                })
            except Exception:
                continue

    # 保存最后一个
    if current_morphotype and current_images:
        entry = _save_morphotype_entry(
            current_morphotype, current_class, current_phylum,
            current_phylum_cn, current_description, current_images,
            output_dir, doc, "北太平洋图集"
        )
        entries.append(entry)

    doc.close()
    print(f"[OK] 北太平洋图集: 提取 {len(entries)} 个形态型")
    return entries


# ---- 索引构建 ----

def build_index(entries: List[Dict], output_dir: str) -> Dict:
    """
    构建完整的检索索引。

    索引结构:
    {
      "version": "1.0",
      "created": "2026-07-16",
      "statistics": {...},
      "by_phylum": {
        "Porifera": { "morphotype_count": N, "image_count": M, "morphotypes": [...] }
      },
      "by_morphotype": {
        "DES_002": { "phylum": "Porifera", "class": "Demospongiae", ... }
      },
      "by_class": {
        "Demospongiae": { "morphotypes": [...] }
      },
      "entries": [...]  # 完整条目列表
    }
    """
    from datetime import datetime

    index = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "statistics": {
            "total_entries": len(entries),
            "total_images": sum(e.get("image_count", 0) for e in entries),
            "total_morphotypes": len(set(
                e.get("morphotype_id", e.get("phylum", "")) for e in entries
            )),
            "phyla": sorted(set(e.get("phylum", "Unknown") for e in entries)),
            "sources": sorted(set(e.get("source", "Unknown") for e in entries)),
        },
        "by_phylum": {},
        "by_morphotype": {},
        "by_class": {},
        "entries": entries,
    }

    for entry in entries:
        phylum = entry.get("phylum", "Unknown")
        morph_id = entry.get("morphotype_id", "")
        class_name = entry.get("class", "Unknown")

        # 按门分组
        if phylum not in index["by_phylum"]:
            index["by_phylum"][phylum] = {
                "morphotype_count": 0,
                "image_count": 0,
                "morphotypes": [],
                "phylum_cn": entry.get("phylum_cn", ""),
            }
        index["by_phylum"][phylum]["morphotype_count"] += 1
        index["by_phylum"][phylum]["image_count"] += entry.get("image_count", 0)
        if morph_id:
            index["by_phylum"][phylum]["morphotypes"].append(morph_id)

        # 按形态型分组
        if morph_id:
            index["by_morphotype"][morph_id] = {
                "phylum": phylum,
                "phylum_cn": entry.get("phylum_cn", ""),
                "class": class_name,
                "source": entry.get("source", ""),
                "description": entry.get("description", {}),
                "image_count": entry.get("image_count", 0),
                "directory": entry.get("directory", ""),
                "images": [
                    {"filename": img["filename"], "path": img["path"]}
                    for img in entry.get("images", [])
                ],
            }

        # 按纲分组
        if class_name and class_name != "Unknown":
            if class_name not in index["by_class"]:
                index["by_class"][class_name] = {
                    "phylum": phylum,
                    "morphotypes": [],
                }
            if morph_id:
                index["by_class"][class_name]["morphotypes"].append(morph_id)

    # 保存索引
    index_path = os.path.join(output_dir, "index.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 索引已保存: {index_path}")
    print(f"  总条目: {index['statistics']['total_entries']}")
    print(f"  总图像: {index['statistics']['total_images']}")
    print(f"  形态型: {index['statistics']['total_morphotypes']}")
    print(f"  门类: {', '.join(index['statistics']['phyla'])}")
    print(f"  来源: {', '.join(index['statistics']['sources'])}")

    return index


# ---- 查询功能 ----

def query_index(index_path: str, keyword: str) -> List[Dict]:
    """从索引中查询匹配的条目。"""
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    keyword_lower = keyword.lower()
    results = []

    # 在形态型中搜索
    for morph_id, info in index.get("by_morphotype", {}).items():
        if (keyword_lower in morph_id.lower() or
            keyword_lower in info.get("phylum", "").lower() or
            keyword_lower in info.get("class", "").lower() or
            keyword_lower in info.get("phylum_cn", "").lower() or
            any(keyword_lower in str(v).lower()
                for v in info.get("description", {}).values())):
            results.append({"morphotype_id": morph_id, **info})

    # 在门中搜索
    for phylum, info in index.get("by_phylum", {}).items():
        if keyword_lower in phylum.lower() or keyword_lower in info.get("phylum_cn", "").lower():
            results.append({"phylum": phylum, **info})

    return results


# ---- 主入口 ----

def main():
    parser = argparse.ArgumentParser(description="生物图集 PDF 解析器")
    parser.add_argument("--all", action="store_true", help="解析所有图集")
    parser.add_argument("--apsma", action="store_true", help="只解析 APSMA 7卷")
    parser.add_argument("--chinese", action="store_true", help="只解析中文图集")
    parser.add_argument("--north-pacific", action="store_true", help="只解析北太平洋图集")
    parser.add_argument("--output", default="data/processed/atlas_lookup",
                       help="输出目录")
    parser.add_argument("--min-image-size", type=int, default=5000,
                       help="最小图像字节数,过滤小图标")
    parser.add_argument("--query", default="", help="查询检索库")
    parser.add_argument("--index-path", default="data/processed/atlas_lookup/index.json",
                       help="索引文件路径 (用于查询)")

    args = parser.parse_args()

    if args.query:
        results = query_index(args.index_path, args.query)
        if results:
            for r in results:
                mid = r.get("morphotype_id", r.get("phylum", ""))
                print(f"  {mid}: {r.get('phylum', '')} / {r.get('class', '')} "
                      f"({r.get('image_count', 0)} images)")
                if "directory" in r:
                    print(f"    目录: {r['directory']}")
        else:
            print(f"未找到匹配 '{args.query}' 的条目")
        return

    if not any([args.all, args.apsma, args.chinese, args.north_pacific]):
        parser.print_help()
        return

    all_entries = []

    if args.apsma or args.all:
        apsma_dir = "data/raw/apsma_pdfs"
        for pdf_name in sorted(APSMA_VOLUME_MAP.keys()):
            pdf_path = os.path.join(apsma_dir, pdf_name)
            if os.path.exists(pdf_path):
                entries = parse_apsma_pdf(pdf_path, args.output, args.min_image_size)
                all_entries.extend(entries)
            else:
                print(f"[SKIP] 未找到: {pdf_path}")

    if args.chinese or args.all:
        cn_pdf = "data/raw/生物图集/西太平洋深渊及临近区底栖生物合集.pdf"
        if os.path.exists(cn_pdf):
            entries = parse_chinese_pdf(cn_pdf, args.output, args.min_image_size)
            all_entries.extend(entries)

    if args.north_pacific or args.all:
        np_pdf = "data/raw/生物图集/Benthic Megafauna from the North Pacific Abyss 2023-.pdf"
        if os.path.exists(np_pdf):
            entries = parse_north_pacific_pdf(np_pdf, args.output, args.min_image_size)
            all_entries.extend(entries)

    # 构建索引
    build_index(all_entries, args.output)


if __name__ == "__main__":
    main()
