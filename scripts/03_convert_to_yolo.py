"""COCO格式 → YOLO格式转换 (FathomNet + APSMA对齐)"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.coco_to_yolo import convert_coco_to_yolo, stratified_split_coco
from utils.taxonomy_mapper import map_to_coarse_class


def main():
    parser = argparse.ArgumentParser(description="COCO转YOLO格式")
    parser.add_argument("--coco-json", required=True, help="COCO JSON文件路径")
    parser.add_argument("--images-dir", required=True, help="图像目录")
    parser.add_argument("--output-dir", default="data/processed/fathomnet_yolo")
    parser.add_argument("--alignment-csv", default=None, help="WoRMS↔APSMA对齐矩阵CSV")
    parser.add_argument("--split", action="store_true", help="是否划分train/val/test")
    args = parser.parse_args()

    # 转换为YOLO格式
    convert_coco_to_yolo(
        coco_json_path=args.coco_json,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        class_mapper=lambda name: map_to_coarse_class(name, args.alignment_csv),
        copy_images=True,
    )

    # 划分数据集
    if args.split:
        stratified_split_coco(
            coco_json_path=args.coco_json,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
