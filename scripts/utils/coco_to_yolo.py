"""COCO格式 → YOLO格式转换器"""
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional, Callable


def convert_coco_to_yolo(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    class_mapper: Optional[Callable[[str], int]] = None,
    copy_images: bool = True,
):
    """将COCO JSON转换为YOLO格式。
    
    Args:
        coco_json_path: COCO标注JSON文件路径
        images_dir: 原始图像目录
        output_dir: 输出YOLO格式目录
        class_mapper: 类别名称→类别ID映射函数
        copy_images: 是否复制/链接图像到输出目录
    """
    import shutil
    
    with open(coco_json_path) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    labels_dir = os.path.join(output_dir, "labels")
    img_out_dir = os.path.join(output_dir, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)

    stats = {"total_images": 0, "total_annotations": 0, "skipped": 0}

    for img_id, anns in ann_by_img.items():
        img = img_map.get(img_id)
        if img is None:
            stats["skipped"] += 1
            continue

        w, h = img.get("width", 0) or 0, img.get("height", 0) or 0
        # 防止除零：宽或高为0时跳过该图像
        if w == 0 or h == 0:
            stats["skipped"] += 1
            continue
        label_lines = []

        for ann in anns:
            cat_name = cat_map.get(ann["category_id"], "unknown")

            if class_mapper:
                class_id = class_mapper(cat_name)
            else:
                class_id = ann["category_id"] - 1

            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            
            # 裁剪到[0,1]范围
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats["total_annotations"] += 1

        if label_lines:
            stem = Path(img.get("file_name", f"{img_id}.png")).stem
            label_file = os.path.join(labels_dir, stem + ".txt")
            with open(label_file, "w") as f:
                f.write("\n".join(label_lines))

            if copy_images:
                src_img = os.path.join(images_dir, img.get("file_name", ""))
                dst_img = os.path.join(img_out_dir, img.get("file_name", ""))
                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    os.symlink(src_img, dst_img)

            stats["total_images"] += 1

    print(f"转换完成: {stats['total_images']} 图像, {stats['total_annotations']} 标注, {stats['skipped']} 跳过")
    return stats


def stratified_split_coco(
    coco_json_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """按类别分层划分COCO数据集为train/val/test。"""
    import random
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    with open(coco_json_path) as f:
        coco = json.load(f)

    # 按图像收集类别
    img_categories = defaultdict(set)
    for ann in coco["annotations"]:
        img_categories[ann["image_id"]].add(ann["category_id"])

    img_ids = list(img_categories.keys())
    primary_cat = [min(img_categories[i]) for i in img_ids]  # 用最小类别ID做分层

    # 两步划分
    train_ids, temp_ids, _, temp_cats = train_test_split(
        img_ids, primary_cat, test_size=(val_ratio + test_ratio),
        random_state=seed, stratify=primary_cat
    )
    
    test_size_adjusted = test_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=test_size_adjusted,
        random_state=seed, stratify=temp_cats
    )

    splits = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    
    # 写入划分文件
    for split_name, split_ids in splits.items():
        split_file = os.path.join(output_dir, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for img in coco["images"]:
                if img["id"] in split_ids:
                    f.write(img.get("file_name", f"{img['id']}.png") + "\n")

    print(f"数据划分: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return splits
