"""为已生成的YOLO标签文件批量创建图像符号链接(高效版)。

应用场景:
    00c_align_annotations_to_apsma.py 生成 apsma_aligned_yolo/labels/{train,val,test}/
    但因为 image_path 是相对文件名,images/ 目录为空。
    本脚本直接基于 labels/ 下 .txt 文件名,在指定的图像根目录中
    通过 os.path.exists (O(1) stat 调用) 测试图像是否存在,找到则创建符号链接。

效率: 不会扫描整个目录,只对每个 label 文件做一次 stat 调用。
"""
import argparse
import os
from pathlib import Path
from typing import Iterable

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_image_for_stem(stem: str, image_dirs: list) -> str:
    """对一个 stem,尝试在 image_dirs 各目录中查找图像。

    返回找到的图像绝对路径,未找到返回 None。
    """
    for d in image_dirs:
        for ext in IMG_EXTS:
            candidate = os.path.join(d, f"{stem}{ext}")
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    return None


def link_images_for_split(
    labels_dir: Path,
    images_out_dir: Path,
    image_dirs: list,
) -> tuple:
    """为一个 split (train/val/test) 创建符号链接。

    Returns:
        (linked_count, missing_count, missing_examples)
    """
    images_out_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    missing = 0
    missing_examples = []

    for label_file in labels_dir.glob("*.txt"):
        stem = label_file.stem
        src = find_image_for_stem(stem, image_dirs)
        if src is None:
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(stem)
            continue

        dst = images_out_dir / Path(src).name
        if not dst.exists() and not dst.is_symlink():
            # 使用绝对路径作为 symlink 目标,避免相对路径解析问题
            os.symlink(src, dst)
        linked += 1

    print(f"  {labels_dir.name}: 已链接 {linked}, 缺失 {missing}")
    if missing_examples:
        print(f"    缺失样本示例: {missing_examples}")
    return linked, missing


def main():
    parser = argparse.ArgumentParser(
        description="为YOLO标签文件批量创建图像符号链接(高效版)"
    )
    parser.add_argument(
        "--yolo-root",
        required=True,
        help="YOLO 数据根目录(应包含 labels/ 和 images/ 子目录)",
    )
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        required=True,
        help="图像根目录列表(按顺序查找,找到即停止)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="要处理的 split 列表(默认 train val test)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("图像根目录:")
    for d in args.image_dirs:
        exists = "✓" if os.path.exists(d) else "✗"
        print(f"  {exists} {d}")

    yolo_root = Path(args.yolo_root)
    total_linked = 0
    total_missing = 0

    print("=" * 60)
    print(f"为 split {args.splits} 创建符号链接...")
    for split in args.splits:
        labels_dir = yolo_root / "labels" / split
        if not labels_dir.exists():
            print(f"  跳过 {split}: labels 目录不存在 {labels_dir}")
            continue
        images_out_dir = yolo_root / "images" / split
        linked, missing = link_images_for_split(labels_dir, images_out_dir, args.image_dirs)
        total_linked += linked
        total_missing += missing

    print("=" * 60)
    print(f"总计: 已链接 {total_linked} 图像, 缺失 {total_missing} 图像")
    if total_missing > 0:
        print(f"⚠ {total_missing} 个标签文件没有对应图像(YOLO 训练时会自动跳过)")


if __name__ == "__main__":
    main()
