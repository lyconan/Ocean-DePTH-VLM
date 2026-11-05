#!/usr/bin/env python3
"""构建 FathomNet 细粒度 YOLO 数据集 (v1.0)

从 FathomNet COCO 标注中筛选样本数 ≥ 阈值的细粒度类别,
转换为 YOLO 格式, 完全消除 "Unknown" 类别。

改进点 (相比旧 03_convert_to_yolo.py):
1. 细粒度类别: 物种/属级 (非门级粗分类), 充分利用 2553 个类别
2. 无 Unknown: 检测不到就不标注, 不强行归为 Unknown
3. 类别平衡: 生成 class_weights.json 供 focal loss 使用
4. 频率排序: 高频类低 ID, 便于分析

用法:
    python scripts/06_build_finegrained_yolo.py \\
        --fathomnet-dir /mnt/.../FathomNet \\
        --output-dir data/processed/fathomnet_finegrained_yolo \\
        --min-samples 50
"""
import json
import os
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def load_coco(coco_json_path: str) -> dict:
    """加载 COCO 标注 JSON。"""
    print(f"加载 COCO: {coco_json_path}")
    with open(coco_json_path) as f:
        return json.load(f)


def select_finegrained_categories(
    train_coco: dict,
    val_coco: dict,
    min_samples: int = 50,
) -> dict:
    """选择训练集中样本数 ≥ min_samples 的细粒度类别。

    返回: {original_cat_id: new_class_id} 映射 (按频率降序)
    """
    cat_map = {c["id"]: c["name"] for c in train_coco["categories"]}

    # 统计训练集每个类别的标注数
    train_counts = Counter(ann["category_id"] for ann in train_coco["annotations"])

    # 筛选 ≥ min_samples 的类别
    selected = [(cat_id, cnt) for cat_id, cnt in train_counts.items() if cnt >= min_samples]
    # 按样本数降序排序 (高频类获得低 ID)
    selected.sort(key=lambda x: -x[1])

    # 构建 映射
    id_map = {cat_id: new_id for new_id, (cat_id, _) in enumerate(selected)}

    # 类别名称列表
    names = [cat_map[cat_id] for cat_id, _ in selected]

    print(f"\n=== 细粒度类别筛选 (min_samples={min_samples}) ===")
    print(f"  FathomNet 总类别: {len(cat_map)}")
    print(f"  选中类别: {len(selected)}")
    print(f"  覆盖训练标注: {sum(c for _, c in selected)} / {sum(train_counts.values())} "
          f"({sum(c for _, c in selected)/sum(train_counts.values())*100:.1f}%)")
    print(f"  样本数范围: {selected[-1][1]} ~ {selected[0][1]}")
    print(f"\n  Top10 类别:")
    for i, (cat_id, cnt) in enumerate(selected[:10]):
        print(f"    [{i:3d}] {cat_map[cat_id]:<40} {cnt:>6} 样本")
    print(f"  ...")
    print(f"  Bottom5 类别:")
    for i, (cat_id, cnt) in enumerate(selected[-5:]):
        print(f"    [{id_map[cat_id]:3d}] {cat_map[cat_id]:<40} {cnt:>6} 样本")

    return id_map, names, dict(train_counts)


def convert_split_to_yolo(
    coco: dict,
    id_map: dict,
    images_dir: str,
    output_labels_dir: str,
    output_images_dir: str,
    split_name: str,
    skip_images: bool = False,
) -> dict:
    """将一个 COCO split 转换为 YOLO 格式。

    只保留 id_map 中的类别标注, 其余跳过 (不归为 Unknown)。
    skip_images=True 时跳过逐个图像符号链接 (NAS 上太慢), 改用目录级符号链接。
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    if not skip_images:
        os.makedirs(output_images_dir, exist_ok=True)

    img_map = {img["id"]: img for img in coco["images"]}
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in id_map:  # 只保留选中类别
            ann_by_img[ann["image_id"]].append(ann)

    stats = {"images": 0, "annotations": 0, "skipped_no_ann": 0, "skipped_no_img": 0}
    linked = 0

    for img_id, anns in ann_by_img.items():
        img = img_map.get(img_id)
        if img is None:
            stats["skipped_no_img"] += 1
            continue

        w = img.get("width", 0)
        h = img.get("height", 0)
        if w == 0 or h == 0:
            stats["skipped_no_img"] += 1
            continue

        label_lines = []
        for ann in anns:
            class_id = id_map[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]
            cx = max(0.0, min(1.0, (x + bw / 2) / w))
            cy = max(0.0, min(1.0, (y + bh / 2) / h))
            nw = max(0.0, min(1.0, bw / w))
            nh = max(0.0, min(1.0, bh / h))
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats["annotations"] += 1

        if not label_lines:
            stats["skipped_no_ann"] += 1
            continue

        # 写标签文件
        file_name = img.get("file_name", f"{img_id}.png")
        stem = Path(file_name).stem
        label_file = os.path.join(output_labels_dir, stem + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(label_lines))

        # 链接图像 (符号链接, 避免复制大文件)
        # skip_images=True 时跳过 (NAS 上逐个符号链接太慢, 改用目录级符号链接)
        if not skip_images:
            src_img = os.path.join(images_dir, file_name)
            dst_img = os.path.join(output_images_dir, file_name)
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                try:
                    os.symlink(src_img, dst_img)
                    linked += 1
                except OSError:
                    pass  # 符号链接失败时跳过 (NAS 权限等)

        stats["images"] += 1

    print(f"\n=== {split_name} 转换完成 ===")
    print(f"  图像: {stats['images']}, 标注: {stats['annotations']}")
    print(f"  新链接图像: {linked}")
    print(f"  跳过(无标注/无图像): {stats['skipped_no_ann'] + stats['skipped_no_img']}")
    return stats


def compute_class_weights(train_counts: dict, id_map: dict) -> list:
    """计算类别权重 (用于 focal loss / class weights)。

    使用 inverse-frequency 平方根缩放, 避免极端权重。
    """
    import math
    max_id = max(id_map.values())
    weights = [1.0] * (max_id + 1)
    counts = [0] * (max_id + 1)
    for cat_id, new_id in id_map.items():
        counts[new_id] = train_counts.get(cat_id, 1)

    total = sum(counts)
    for i in range(max_id + 1):
        freq = counts[i] / total if total > 0 else 1.0
        # inverse-frequency sqrt 缩放
        weights[i] = 1.0 / math.sqrt(max(freq, 1e-6))

    # 归一化到均值 1.0
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    return weights


def write_yaml_config(
    output_dir: str,
    names: list,
    config_path: str,
):
    """写入 YOLO 训练 YAML 配置文件。"""
    nc = len(names)
    yaml_lines = [
        f"# FathomNet 细粒度 YOLO 训练配置 (v1.0) — {nc} 类, 无 Unknown",
        f"# 自动生成 by 06_build_finegrained_yolo.py",
        f"path: {os.path.abspath(output_dir)}",
        f"train: images/train",
        f"val: images/val",
        f"",
        f"nc: {nc}",
        f"names:",
    ]
    for i, name in enumerate(names):
        yaml_lines.append(f"  {i}: {name}")
    yaml_lines.extend([
        f"",
        f"# 训练参数",
        f"base_model: yolo11m.pt",
        f"imgsz: 640",
        f"epochs: 150",
        f"batch: 8",
        f"lr0: 0.005",
        f"lrf: 0.0001",
        f"warmup_epochs: 5",
        f"patience: 30",
        f"",
        f"# 数据增强 (深海特有)",
        f"mosaic: 1.0",
        f"mixup: 0.15",
        f"copy_paste: 0.15",
        f"flipud: 0.5",
        f"hsv_v: 0.4",
        f"hsv_s: 0.3",
        f"",
        f"# 推理阈值",
        f"conf_filter: 0.15",
    ])
    with open(config_path, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")
    print(f"\nYOLO 配置已写入: {config_path} (nc={nc})")


def main():
    parser = argparse.ArgumentParser(description="构建 FathomNet 细粒度 YOLO 数据集")
    parser.add_argument("--fathomnet-dir", default="/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet",
                        help="FathomNet 数据根目录")
    parser.add_argument("--output-dir", default="data/processed/fathomnet_finegrained_yolo",
                        help="输出 YOLO 数据集目录")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="类别最小样本数阈值 (默认 50)")
    parser.add_argument("--config-out", default="config/yolo_finegrained.yaml",
                        help="输出 YOLO 训练配置文件路径")
    parser.add_argument("--skip-images", action="store_true",
                        help="跳过逐个图像符号链接 (NAS上太慢), 改用目录级符号链接")
    args = parser.parse_args()

    fathomnet_dir = Path(args.fathomnet_dir)
    train_coco = load_coco(str(fathomnet_dir / "Annotations" / "fathomnet_train.json"))
    val_coco = load_coco(str(fathomnet_dir / "Annotations" / "fathomnet_val.json"))

    # 1. 选择细粒度类别
    id_map, names, train_counts = select_finegrained_categories(
        train_coco, val_coco, min_samples=args.min_samples
    )

    # 2. 转换训练集
    output_dir = Path(args.output_dir)
    train_stats = convert_split_to_yolo(
        train_coco, id_map,
        images_dir=str(fathomnet_dir / "Train"),
        output_labels_dir=str(output_dir / "labels" / "train"),
        output_images_dir=str(output_dir / "images" / "train"),
        split_name="train",
        skip_images=args.skip_images,
    )

    # 3. 转换验证集
    val_stats = convert_split_to_yolo(
        val_coco, id_map,
        images_dir=str(fathomnet_dir / "Val"),
        output_labels_dir=str(output_dir / "labels" / "val"),
        output_images_dir=str(output_dir / "images" / "val"),
        split_name="val",
        skip_images=args.skip_images,
    )

    # 3b. 目录级符号链接 (skip_images 模式下, 用单个符号链接代替逐个文件链接)
    if args.skip_images:
        images_base = output_dir / "images"
        images_base.mkdir(parents=True, exist_ok=True)
        for split, src_name in [("train", "Train"), ("val", "Val")]:
            link_path = images_base / split
            src_path = fathomnet_dir / src_name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            os.symlink(str(src_path), str(link_path))
            print(f"目录级符号链接: {link_path} → {src_path}")

    # 4. 计算类别权重 (focal loss)
    class_weights = compute_class_weights(train_counts, id_map)
    weights_path = output_dir / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump({
            "class_weights": class_weights,
            "class_names": names,
            "nc": len(names),
            "min_samples": args.min_samples,
        }, f, indent=2)
    print(f"\n类别权重已写入: {weights_path}")

    # 5. 写类别名称文件
    names_path = output_dir / "classes.names"
    with open(names_path, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"类别名称已写入: {names_path}")

    # 6. 写 YOLO 配置
    write_yaml_config(str(output_dir), names, args.config_out)

    # 7. 汇总
    print(f"\n{'='*60}")
    print(f"细粒度 YOLO 数据集构建完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  类别数: {len(names)} (min_samples={args.min_samples})")
    print(f"  训练集: {train_stats['images']} 图, {train_stats['annotations']} 标注")
    print(f"  验证集: {val_stats['images']} 图, {val_stats['annotations']} 标注")
    print(f"  配置文件: {args.config_out}")
    print(f"  类别权重: {weights_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
