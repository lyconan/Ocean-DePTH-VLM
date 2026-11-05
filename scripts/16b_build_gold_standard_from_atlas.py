#!/usr/bin/env python3
"""从 APSMA 图集构建 Gold Standard 测试集 (v2.0)

修复 16_build_gold_standard.py 的问题:
- 原版从 FathomNet YOLO 测试集采样,但 FathomNet 标注只有门级,apsma_morphotype_id 全为 unknown
- 本版直接从 APSMA 图集(atlas_lookup)采样,每个形态型有真实的 morphotype_id 标注

数据源:
- atlas_lookup/index.json (186 形态型, 1287 张图像, 北太平洋图集)
- apsma_knowledge_validated.jsonl (建立 original_id → morphotype_id 映射)

输出 CSV 字段:
- image_path: 图集图像路径
- phylum: 门级 (从 atlas_lookup)
- apsma_morphotype_id: APSMA 形态型 ID (如 APSMA_V1_Annelida_Annelida_Mo02)
- class_id: YOLO class_id (门级映射, 便于检测评估)
- bbox: 整图 bbox (1.0 版本, 因图集图像是单目标裁剪)
- source: atlas_lookup
- original_morphotype_id: 原始形态型 ID (如 ANN_002)
- class_name: 纲名
- order_name: 目名

用法:
    python scripts/16b_build_gold_standard_from_atlas.py \\
        --num-samples 50 \\
        --output data/test/gold_standard/gold_standard_atlas.csv
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path


# YOLO class_id → phylum 映射 (与 config/yolo_training.yaml 一致)
CLASS_NAMES = {
    0: "Porifera",
    1: "Echinodermata",
    2: "Cnidaria",
    3: "Annelida",
    4: "Arthropoda",
    5: "Xenophyophorea",
    6: "Mollusca",
    7: "Brachiopoda",
    8: "Foraminifera",
    9: "Unknown",
}
# 反向映射: phylum → class_id
PHYLUM_TO_CLASS_ID = {v: k for k, v in CLASS_NAMES.items()}


def load_id_mapping(jsonl_path: str) -> dict:
    """从 apsma_knowledge_validated.jsonl 建立 original_id → morphotype_id 映射。"""
    mapping = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            orig = d.get("original_morphotype_id")
            mid = d.get("morphotype_id")
            if orig and mid:
                mapping[orig] = mid
    return mapping


def collect_atlas_samples(
    index_path: str, id_mapping: dict, num_samples: int = 50,
    per_class_max: int = 8, images_per_morphotype: int = 1,
):
    """从 atlas_lookup 按门级平衡采样。

    返回: list of dict, 每个 dict 包含 image_path, phylum, apsma_morphotype_id 等
    """
    with open(index_path) as f:
        idx = json.load(f)

    by_morphotype = idx["by_morphotype"]
    by_phylum = idx.get("by_phylum", {})

    # 按门级组织形态型
    phylum_to_morphotypes = defaultdict(list)
    for mt_key, mt_info in by_morphotype.items():
        phylum = mt_info.get("phylum", "Unknown")
        # 跳过未在 YOLO 10 类中的门 (如 Chelicerata 归到 Arthropoda, Bryozoa 等不在 YOLO 类)
        phylum_to_morphotypes[phylum].append((mt_key, mt_info))

    # 优先选择 YOLO 类别中的门 (Porifera/Echinodermata/Cnidaria/Annelida/Arthropoda/Mollusca)
    yolo_phyla = set(CLASS_NAMES.values())
    print(f"[Atlas] 总形态型: {len(by_morphotype)}")
    print(f"[Atlas] 按门级分布:")
    for phylum, mts in sorted(phylum_to_morphotypes.items(), key=lambda x: -len(x[1])):
        in_yolo = "✓" if phylum in yolo_phyla else " "
        print(f"  [{in_yolo}] {phylum:<20} {len(mts):>3} 形态型")

    # 类别平衡采样
    selected = []
    random.seed(42)  # 可复现

    # 1. 优先从 YOLO 类别中的门采样
    for phylum in CLASS_NAMES.values():
        if phylum not in phylum_to_morphotypes:
            continue
        mts = phylum_to_morphotypes[phylum]
        random.shuffle(mts)
        taken = 0
        for mt_key, mt_info in mts:
            if taken >= per_class_max:
                break
            # 获取该形态型的图像
            images = mt_info.get("images", [])
            if not images:
                continue
            random.shuffle(images)
            for img in images[:images_per_morphotype]:
                img_path = img.get("path", "")
                if not img_path or not os.path.exists(img_path):
                    continue
                # 映射到 Neo4j morphotype_id
                apsma_id = id_mapping.get(mt_key)
                if not apsma_id:
                    print(f"  [警告] {mt_key} 无 morphotype_id 映射, 跳过")
                    continue
                selected.append({
                    "image_path": img_path,
                    "phylum": phylum,
                    "apsma_morphotype_id": apsma_id,
                    "original_morphotype_id": mt_key,
                    "class_name": mt_info.get("class", ""),
                    "order_name": mt_info.get("order", ""),
                    "source": f"atlas_lookup/{mt_info.get('source', '')}",
                    "bbox": "0.0 0.0 1.0 1.0",  # 整图 bbox
                    "class_id": PHYLUM_TO_CLASS_ID.get(phylum, 9),
                })
                taken += 1
                if taken >= per_class_max:
                    break

    # 2. 如果不足 num_samples, 从其他门补充 (Bryozoa, Chelicerata, Ctenophora, Nemertea)
    if len(selected) < num_samples:
        other_phyla = [p for p in phylum_to_morphotypes if p not in yolo_phyla]
        for phylum in other_phyla:
            mts = phylum_to_morphotypes[phylum]
            random.shuffle(mts)
            for mt_key, mt_info in mts:
                if len(selected) >= num_samples:
                    break
                images = mt_info.get("images", [])
                if not images:
                    continue
                random.shuffle(images)
                for img in images[:images_per_morphotype]:
                    if len(selected) >= num_samples:
                        break
                    img_path = img.get("path", "")
                    if not img_path or not os.path.exists(img_path):
                        continue
                    apsma_id = id_mapping.get(mt_key)
                    if not apsma_id:
                        continue
                    selected.append({
                        "image_path": img_path,
                        "phylum": phylum,
                        "apsma_morphotype_id": apsma_id,
                        "original_morphotype_id": mt_key,
                        "class_name": mt_info.get("class", ""),
                        "order_name": mt_info.get("order", ""),
                        "source": f"atlas_lookup/{mt_info.get('source', '')}",
                        "bbox": "0.0 0.0 1.0 1.0",
                        "class_id": PHYLUM_TO_CLASS_ID.get(phylum, 9),
                    })

    return selected[:num_samples]


def build_gold_standard(
    num_samples: int = 50, output_path: str = None,
    per_class_max: int = 8,
):
    """构建从 APSMA 图集采样的 gold standard CSV。"""
    base = Path(__file__).parent.parent
    index_path = base / "data/processed/atlas_lookup/index.json"
    jsonl_path = base / "data/processed/apsma_parsed/apsma_knowledge_validated.jsonl"

    if not index_path.exists():
        print(f"[错误] atlas 索引不存在: {index_path}")
        return
    if not jsonl_path.exists():
        print(f"[错误] apsma_knowledge 不存在: {jsonl_path}")
        return

    print(f"[Gold Standard] 加载 ID 映射: {jsonl_path}")
    id_mapping = load_id_mapping(str(jsonl_path))
    print(f"[Gold Standard] ID 映射: {len(id_mapping)} 条")

    print(f"\n[Gold Standard] 从 atlas_lookup 采样 {num_samples} 样本 (每门最多 {per_class_max})...")
    samples = collect_atlas_samples(
        str(index_path), id_mapping, num_samples, per_class_max
    )

    if not samples:
        print("[错误] 未采样到任何样本!")
        return

    # 统计门级分布
    phylum_dist = defaultdict(int)
    for s in samples:
        phylum_dist[s["phylum"]] += 1

    print(f"\n[Gold Standard] 采样 {len(samples)} 个样本")
    print("[Gold Standard] 门级分布:")
    for phylum, cnt in sorted(phylum_dist.items(), key=lambda x: -x[1]):
        print(f"  {phylum:<20} {cnt:>3} 个")

    # 写入 CSV
    if output_path is None:
        output_path = "data/test/gold_standard/gold_standard_atlas.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "image_path", "phylum", "apsma_morphotype_id",
        "class_id", "bbox", "source",
        "original_morphotype_id", "class_name", "order_name",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(s)

    print(f"\n[Gold Standard] 保存: {output_path}")
    print(f"[Gold Standard] 总计: {len(samples)} 个样本 (全部有真实 apsma_morphotype_id)")

    # 同时生成 10 样本快速验证子集
    quick_path = output_path.replace(".csv", "_10.csv")
    with open(quick_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples[:10]:
            writer.writerow(s)
    print(f"[Gold Standard] 快速验证子集: {quick_path} (10 样本)")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 APSMA 图集构建 Gold Standard 测试集")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--per-class-max", type=int, default=8,
                        help="每个门最多采样数量")
    parser.add_argument(
        "--output", default="data/test/gold_standard/gold_standard_atlas.csv"
    )
    args = parser.parse_args()

    build_gold_standard(args.num_samples, args.output, args.per_class_max)
