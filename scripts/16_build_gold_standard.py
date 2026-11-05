#!/usr/bin/env python3
"""构建 gold standard 测试集 — 从 FathomNet YOLO 测试集标注生成 (v1.0)

生成消融实验所需的 gold_standard_morphotypes.csv:
- image_path: 图片路径 (解析符号链接)
- phylum: 门类 (从 YOLO class_id 映射)
- apsma_morphotype_id: APSMA 形态型 ID (通过 GraphRAG 检索获取或设为 unknown)
- class_id: YOLO 类别 ID
- bbox: YOLO 标注的 bbox (用于精确评估)

用法:
    python scripts/16_build_gold_standard.py \\
        --yolo-dir data/processed/fathomnet_yolo \\
        --config config/yolo_training.yaml \\
        --num-samples 50 \\
        --output data/test/gold_standard/gold_standard_morphotypes.csv
"""
import argparse
import csv
import os
import random
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


def collect_test_samples(
    yolo_dir: str, num_samples: int = 50, balance_classes: bool = True
):
    """从 FathomNet YOLO 测试集收集样本。

    返回: [(image_path, class_id, bbox_str), ...]
    """
    images_dir = Path(yolo_dir) / "images"
    labels_dir = Path(yolo_dir) / "labels"

    # 按类别收集样本 (用于类别平衡)
    class_samples = {i: [] for i in range(10)}

    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    random.shuffle(image_files)

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                bbox = " ".join(parts[1:5])
                class_samples[cls_id].append((str(img_path), cls_id, bbox))

        # 检查是否已收集足够样本
        total = sum(len(v) for v in class_samples.values())
        if total >= num_samples * 3:  # 收集 3 倍用于筛选
            break

    # 类别平衡采样
    if balance_classes:
        # 每个类别最多 num_samples/10 个，最少 1 个
        per_class = max(1, num_samples // len(CLASS_NAMES))
        selected = []
        for cls_id, samples in class_samples.items():
            random.shuffle(samples)
            selected.extend(samples[:per_class])
        # 如果不足，补充
        if len(selected) < num_samples:
            all_remaining = []
            for cls_id, samples in class_samples.items():
                all_remaining.extend(samples[per_class:])
            random.shuffle(all_remaining)
            selected.extend(all_remaining[:num_samples - len(selected)])
    else:
        all_samples = []
        for samples in class_samples.values():
            all_samples.extend(samples)
        random.shuffle(all_samples)
        selected = all_samples[:num_samples]

    return selected[:num_samples]


def build_gold_standard(
    yolo_dir: str, num_samples: int = 50, output_path: str = None
):
    """构建 gold standard CSV。"""
    print(f"[Gold Standard] 从 {yolo_dir} 收集 {num_samples} 个样本...")

    samples = collect_test_samples(yolo_dir, num_samples)

    if not samples:
        print("[错误] 未找到任何样本!")
        return

    # 统计类别分布
    class_dist = {}
    for _, cls_id, _ in samples:
        class_dist[cls_id] = class_dist.get(cls_id, 0) + 1

    print(f"[Gold Standard] 收集 {len(samples)} 个样本")
    print("[Gold Standard] 类别分布:")
    for cls_id in sorted(class_dist.keys()):
        name = CLASS_NAMES.get(cls_id, f"Class{cls_id}")
        print(f"  {cls_id}: {name:<20} {class_dist[cls_id]:>4} 个")

    # 写入 CSV
    if output_path is None:
        output_path = "data/test/gold_standard/gold_standard_morphotypes.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "phylum", "apsma_morphotype_id",
            "class_id", "bbox", "source"
        ])

        for img_path, cls_id, bbox in samples:
            phylum = CLASS_NAMES.get(cls_id, "Unknown")
            # apsma_morphotype_id 设为 unknown (FathomNet YOLO 标注只有门级)
            writer.writerow([
                img_path,
                phylum,
                "unknown",
                cls_id,
                bbox,
                "fathomnet_yolo_test"
            ])

    print(f"\n[Gold Standard] 保存: {output_path}")
    print(f"[Gold Standard] 总计: {len(samples)} 个样本")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 gold standard 测试集")
    parser.add_argument(
        "--yolo-dir", default="data/processed/fathomnet_yolo",
        help="FathomNet YOLO 数据目录"
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument(
        "--output", default="data/test/gold_standard/gold_standard_morphotypes.csv"
    )
    args = parser.parse_args()

    build_gold_standard(args.yolo_dir, args.num_samples, args.output)
