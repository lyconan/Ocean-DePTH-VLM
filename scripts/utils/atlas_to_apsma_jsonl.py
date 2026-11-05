"""将 atlas_lookup/index.json 转换为 Phase 2 期望的 apsma_knowledge.jsonl 格式。

应用场景:
    scripts/utils/parse_atlas_pdfs.py 已经从 APSMA 完整版 PDF 提取了 186 个形态型
    及对应图像,生成 atlas_lookup/index.json (按 phylum/morphotype/class 三维索引)。
    但 Phase 2 下游脚本 (09_build_knowledge_graph.py / 07_extract_clip_embeddings.py)
    期望读取 apsma_knowledge.jsonl,每行一条记录,字段:
      morphotype_id, taxonomy{phylum,class,order,morphotype},
      features{description,size_range_cm,substrate,depth_range_m},
      image{path,dimensions,type},
      occurrence{ccz_area,latitude_range,longitude_range},
      source{volume,page}

本脚本完成此格式转换,作为 06_parse_apsma.py 的替代实现(无需 MinerU/LLM)。
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List


def convert_atlas_to_jsonl(
    index_json_path: str,
    output_jsonl_path: str,
    primary_image_only: bool = True,
):
    """转换 atlas_lookup/index.json → apsma_knowledge.jsonl。

    Args:
        index_json_path: atlas_lookup/index.json 路径
        output_jsonl_path: 输出 apsma_knowledge.jsonl 路径
        primary_image_only: True 时只保留每个形态型的第一张图(代表图);
            False 时为每个形态型生成多条记录(每个图像一条)
    """
    with open(index_json_path, encoding="utf-8") as f:
        idx = json.load(f)

    by_morphotype: Dict = idx.get("by_morphotype", {})
    print(f"加载 {len(by_morphotype)} 个形态型")

    records: List[Dict] = []
    skipped_no_image = 0

    for morphotype_id, info in by_morphotype.items():
        phylum = info.get("phylum", "Unknown")
        class_name = info.get("class", "")
        source_name = info.get("source", "")
        description = info.get("description", {}) or {}
        morphology_text = description.get("morphology", "") if isinstance(description, dict) else str(description)
        images: List[Dict] = info.get("images", []) or []

        if not images:
            skipped_no_image += 1
            # 即使没有图,也生成一条记录(image.path 为空字符串)
            records.append(_build_record(
                morphotype_id=morphotype_id,
                phylum=phylum,
                class_name=class_name,
                morphology_text=morphology_text,
                img_path="",
                source_name=source_name,
            ))
            continue

        if primary_image_only:
            # 只取第一张图作为 ex_situ 代表图
            primary_img = images[0]
            img_path = primary_img.get("path", "")
            records.append(_build_record(
                morphotype_id=morphotype_id,
                phylum=phylum,
                class_name=class_name,
                morphology_text=morphology_text,
                img_path=img_path,
                source_name=source_name,
            ))
        else:
            # 为每张图生成一条记录(同 morphotype_id 但 image.path 不同)
            for img_info in images:
                img_path = img_info.get("path", "")
                records.append(_build_record(
                    morphotype_id=morphotype_id,
                    phylum=phylum,
                    class_name=class_name,
                    morphology_text=morphology_text,
                    img_path=img_path,
                    source_name=source_name,
                ))

    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"输出: {output_path}")
    print(f"总计: {len(records)} 条记录")
    print(f"  - 跳过(无图): {skipped_no_image}")
    print(f"  - 含图记录: {len(records) - skipped_no_image}")
    return records


def _build_record(
    morphotype_id: str,
    phylum: str,
    class_name: str,
    morphology_text: str,
    img_path: str,
    source_name: str,
) -> Dict:
    """构建 apsma_knowledge.jsonl 单条记录。

    生成符合 schemas.py MORPHOTYPE_ID_PATTERN 的 ID:
        原始 APSMA 编号(如 ANN_002) → APSMA_V1_Annelida_Annelida_Mo002
    同时保留 original_morphotype_id 字段以便 PDF 追溯。
    """
    # 从 morphotype_id 提取前缀作为 order(如 DES_002 → DES)
    prefix = morphotype_id.split("_")[0] if "_" in morphotype_id else ""

    # 从 source_name 推断 volume(APSMA完整版视为 volume=1)
    volume = 1 if "APSMA" in source_name or "北太平洋" in source_name else 2

    # 从原始 ID(如 ANN_002) 提取序号
    import re
    num_match = re.search(r"(\d+)$", morphotype_id)
    seq_num = int(num_match.group(1)) if num_match else 1

    # 生成符合 schemas.py MORPHOTYPE_ID_PATTERN 的 ID
    # pattern: ^APSMA_V\d+_[A-Z][a-z]+_[A-Z][a-z]+_Mo\d+$
    phylum_clean = phylum if phylum and phylum[0].isupper() else "Unknown"

    # 清洗 class_name: atlas_lookup 中有些 class 含下划线(如 "Arthropoda_other")
    # 表示"未识别到纲级",需要清洗为合法拉丁名格式
    if class_name and class_name[0].isupper():
        # 取下划线前的部分作为 class_name(如 "Arthropoda_other" → "Arthropoda")
        class_clean = class_name.split("_")[0]
    else:
        class_clean = phylum_clean
    # 二次校验: 必须是 [A-Z][a-z]+ 格式
    if not re.match(r"^[A-Z][a-z]+$", class_clean):
        class_clean = phylum_clean

    schema_morphotype_id = f"APSMA_V{volume}_{phylum_clean}_{class_clean}_Mo{seq_num:02d}"

    # 生成符合 schemas.py LATIN_NAME_PATTERN 的 morphotype name
    # pattern: ^[A-Z][a-z]+(_[a-z]+)*(_Mo\d+)?$
    schema_morphotype_name = f"{class_clean}_Mo{seq_num:02d}"

    return {
        "morphotype_id": schema_morphotype_id,
        "original_morphotype_id": morphotype_id,  # 保留原始 APSMA 编号便于 PDF 追溯
        "taxonomy": {
            "phylum": phylum,
            "class": class_name or phylum,
            "order": prefix,
            "morphotype": schema_morphotype_name,
        },
        "features": {
            "description": morphology_text,
            "size_range_cm": [None, None],  # PDF 解析未提取,留待 LLM 后处理
            "substrate": "",
            "depth_range_m": [None, None],
        },
        "image": {
            "path": img_path,
            "dimensions": {"width": 0, "height": 0},  # BioCLIP 提取时会自动读取
            "type": "ex_situ_specimen_photograph",
        },
        "occurrence": {
            "ccz_area": [],  # PDF 解析未提取
            "latitude_range": [None, None],
            "longitude_range": [None, None],
        },
        "source": {
            "volume": volume,
            "page": 0,
            "atlas_name": source_name,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 atlas_lookup/index.json 转换为 apsma_knowledge.jsonl"
    )
    parser.add_argument(
        "--index-json",
        default="data/processed/atlas_lookup/index.json",
        help="atlas_lookup/index.json 输入路径",
    )
    parser.add_argument(
        "--output",
        default="data/processed/apsma_parsed/apsma_knowledge.jsonl",
        help="apsma_knowledge.jsonl 输出路径",
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="为每个形态型的每张图都生成一条记录(默认只保留代表图)",
    )
    args = parser.parse_args()

    convert_atlas_to_jsonl(
        index_json_path=args.index_json,
        output_jsonl_path=args.output,
        primary_image_only=not args.all_images,
    )
