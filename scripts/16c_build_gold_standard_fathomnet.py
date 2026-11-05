#!/usr/bin/env python3
"""从 FathomNet 真实海底照片构建 Gold Standard 测试集 (v1.0)

解决 16b atlas 版的域偏移问题:
- APSMA 图集是科学插图 (手绘/合成), 与 VLM 训练数据 (FathomNet 真实海底照片) 分布完全不同
- 本版直接从 FathomNet YOLO val 集采样真实海底照片, 验证域内分类性能

数据源:
- data/processed/apsma_aligned_yolo/images/val/ (67K 真实海底照片)
- data/processed/apsma_aligned_yolo/labels/val/ (对应 YOLO 标注)

输出 CSV 字段:
- image_path: FathomNet 图像路径 (真实海底照片)
- phylum: 门级 (从 YOLO class_id 映射, 10 类)
- apsma_morphotype_id: "unknown" (FathomNet 只有门级标注, 无形态型 ID)
- class_id: YOLO class_id
- bbox: YOLO 格式 bbox (x_center y_center width height, 归一化)
- source: fathomnet_val

评估指标:
- Phylum Accuracy (主要指标, 目标 >= 0.80)
- Cross-phylum Error Rate (目标 <= 0.10)
- Morphotype Hit Rate: 不适用 (FathomNet 无形态型标注)

用法:
    python scripts/16c_build_gold_standard_fathomnet.py \\
        --num-samples 50 \\
        --output data/test/gold_standard/gold_standard_fathomnet.csv
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


def collect_fathomnet_samples(
    yolo_dir: str,
    num_samples: int = 50,
    per_class_max: int = 8,
    prefer_single_object: bool = True,
    min_bbox_area: float = 0.01,  # 归一化面积, 过滤太小的标注
    max_bbox_area: float = 0.80,  # 过滤整图标注
):
    """从 FathomNet YOLO val 集按门级平衡采样真实海底照片。

    Args:
        yolo_dir: apsma_aligned_yolo 目录 (含 images/val/ 和 labels/val/)
        num_samples: 目标样本数
        per_class_max: 每个门最多采样数
        prefer_single_object: 优先选择只有一个标注的图像 (评估更清晰)
        min_bbox_area: 标注面积下限 (归一化), 过滤太小的目标
        max_bbox_area: 标注面积上限 (归一化), 过滤整图标注
    """
    images_dir = Path(yolo_dir) / "images" / "val"
    labels_dir = Path(yolo_dir) / "labels" / "val"

    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标注目录不存在: {labels_dir}")

    # 按类别收集候选样本
    class_samples = {i: [] for i in range(10)}
    total_scanned = 0
    total_with_labels = 0

    image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    random.seed(42)
    random.shuffle(image_files)

    print(f"[FathomNet] 扫描 {len(image_files)} 个 val 图像...")
    print(f"[FathomNet] 采样策略: 单目标优先={prefer_single_object}, "
          f"bbox 面积=[{min_bbox_area}, {max_bbox_area}]")

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        total_scanned += 1

        # 读取所有标注
        annotations = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                area = w * h
                # 过滤过大或过小的标注
                if area < min_bbox_area or area > max_bbox_area:
                    continue
                annotations.append((cls_id, " ".join(parts[1:5]), area))

        if not annotations:
            continue

        total_with_labels += 1

        # 优先单目标图像
        if prefer_single_object and len(annotations) > 1:
            # 仍然收集, 但优先级低 (放在 class_samples 末尾)
            pass

        for cls_id, bbox, area in annotations:
            # 每个类别收集足够候选即可
            if len(class_samples[cls_id]) >= per_class_max * 5:  # 收集 5 倍用于筛选
                continue
            class_samples[cls_id].append({
                "image_path": str(img_path),
                "cls_id": cls_id,
                "bbox": bbox,
                "area": area,
                "n_objects": len(annotations),
            })

        # 检查是否已收集足够
        total = sum(len(v) for v in class_samples.values())
        if total >= num_samples * 5:
            break

    print(f"[FathomNet] 扫描 {total_scanned} 图像, {total_with_labels} 有标注")
    print(f"[FathomNet] 候选分布:")
    for cls_id, samples in class_samples.items():
        if samples:
            name = CLASS_NAMES.get(cls_id, f"Class{cls_id}")
            print(f"  {cls_id}: {name:<20} {len(samples):>4} 候选")

    # 类别平衡采样: 每个门最多 per_class_max 个, 优先单目标
    selected = []
    for cls_id in range(10):
        samples = class_samples[cls_id]
        if not samples:
            continue

        # 排序: 单目标优先, 然后按面积适中优先
        samples.sort(key=lambda x: (x["n_objects"] > 1, abs(x["area"] - 0.15)))
        selected.extend(samples[:per_class_max])

    # 如果不足 num_samples, 从剩余候选补充
    if len(selected) < num_samples:
        all_remaining = []
        for cls_id, samples in class_samples.items():
            taken = per_class_max
            all_remaining.extend(samples[taken:])
        random.shuffle(all_remaining)
        selected.extend(all_remaining[:num_samples - len(selected)])

    return selected[:num_samples]


def build_gold_standard(
    yolo_dir: str, num_samples: int = 50, output_path: str = None,
    per_class_max: int = 8,
):
    """构建 FathomNet 真实海底照片版 gold standard。"""
    print(f"\n[FathomNet Gold Standard] 从 {yolo_dir}/images/val 采样 {num_samples} 样本")
    print(f"[FathomNet Gold Standard] 每门最多 {per_class_max} 个")

    samples = collect_fathomnet_samples(yolo_dir, num_samples, per_class_max)

    if not samples:
        print("[错误] 未采样到任何样本!")
        return

    # 统计门级分布
    class_dist = {}
    for s in samples:
        cls_id = s["cls_id"]
        class_dist[cls_id] = class_dist.get(cls_id, 0) + 1

    print(f"\n[FathomNet Gold Standard] 采样 {len(samples)} 个样本")
    print("[FathomNet Gold Standard] 门级分布:")
    for cls_id in sorted(class_dist.keys()):
        name = CLASS_NAMES.get(cls_id, f"Class{cls_id}")
        print(f"  {cls_id}: {name:<20} {class_dist[cls_id]:>4} 个")

    # 写入 CSV
    if output_path is None:
        output_path = "data/test/gold_standard/gold_standard_fathomnet.csv"

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
            phylum = CLASS_NAMES.get(s["cls_id"], "Unknown")
            writer.writerow({
                "image_path": s["image_path"],
                "phylum": phylum,
                "apsma_morphotype_id": "unknown",  # FathomNet 只有门级标注
                "class_id": s["cls_id"],
                "bbox": s["bbox"],
                "source": "fathomnet_val",
                "original_morphotype_id": "",
                "class_name": "",
                "order_name": "",
            })

    print(f"\n[FathomNet Gold Standard] 保存: {output_path}")
    print(f"[FathomNet Gold Standard] 总计: {len(samples)} 个样本 (真实海底照片)")

    # 同时生成 10 样本快速验证子集
    quick_path = output_path.replace(".csv", "_10.csv")
    with open(quick_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples[:10]:
            phylum = CLASS_NAMES.get(s["cls_id"], "Unknown")
            writer.writerow({
                "image_path": s["image_path"],
                "phylum": phylum,
                "apsma_morphotype_id": "unknown",
                "class_id": s["cls_id"],
                "bbox": s["bbox"],
                "source": "fathomnet_val",
                "original_morphotype_id": "",
                "class_name": "",
                "order_name": "",
            })
    print(f"[FathomNet Gold Standard] 快速验证子集: {quick_path} (10 样本)")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 FathomNet 真实海底照片构建 Gold Standard")
    parser.add_argument(
        "--yolo-dir", default="data/processed/apsma_aligned_yolo",
        help="APSMA 对齐 YOLO 数据目录 (含 images/val/ 和 labels/val/)",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--per-class-max", type=int, default=8,
                        help="每个门最多采样数量")
    parser.add_argument(
        "--output", default="data/test/gold_standard/gold_standard_fathomnet.csv"
    )
    args = parser.parse_args()

    build_gold_standard(args.yolo_dir, args.num_samples, args.output, args.per_class_max)
