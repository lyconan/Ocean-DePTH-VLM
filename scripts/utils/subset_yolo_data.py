"""从大型 YOLO 数据集中抽取小子集,用于快速验证训练 pipeline。

应用场景:
    apsma_aligned_yolo/ 有 28万标签 + 28万图像 symlink,
    ultralytics 首次扫描 NAS 上 28万文件需数小时,
    无法快速验证训练 pipeline 是否能正常工作。
    本脚本抽取 N 张图(及其对应标签)到新目录,并生成对应的 yolo yaml 配置。
"""
import argparse
import os
import shutil
from pathlib import Path
import random


def subset_yolo_data(
    src_root: str,
    dst_root: str,
    n_per_split: int = 500,
    seed: int = 42,
    splits=("train", "val", "test"),
):
    """从 src_root 抽取 n_per_split 张图/标签 到 dst_root。"""
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    total_imgs = 0
    total_lbls = 0

    for split in splits:
        src_imgs = src_root / "images" / split
        src_lbls = src_root / "labels" / split
        dst_imgs = dst_root / "images" / split
        dst_lbls = dst_root / "labels" / split
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        if not src_imgs.exists():
            print(f"  跳过 {split}: {src_imgs} 不存在")
            continue

        # 列出所有图像(可能是 symlink)
        all_imgs = sorted(src_imgs.iterdir())
        random.shuffle(all_imgs)
        selected = all_imgs[:n_per_split]
        print(f"  {split}: 选中 {len(selected)} / {len(all_imgs)}")

        n_imgs = 0
        n_lbls = 0
        for img_path in selected:
            # 复制图像(避免 symlink 解析问题)
            dst_img = dst_imgs / img_path.name
            try:
                # 如果是 symlink,用复制而非创建新 symlink(避免解析问题)
                if img_path.is_symlink():
                    shutil.copy2(img_path.resolve(), dst_img)
                else:
                    shutil.copy2(img_path, dst_img)
                n_imgs += 1
            except Exception as e:
                print(f"    复制图像失败 {img_path.name}: {e}")
                continue

            # 复制对应标签(同名 .txt)
            lbl_path = src_lbls / (img_path.stem + ".txt")
            if lbl_path.exists():
                dst_lbl = dst_lbls / lbl_path.name
                shutil.copy2(lbl_path, dst_lbl)
                n_lbls += 1

        total_imgs += n_imgs
        total_lbls += n_lbls
        print(f"    已复制: {n_imgs} 图像, {n_lbls} 标签")

    print(f"\n总计: {total_imgs} 图像, {total_lbls} 标签")
    print(f"输出: {dst_root}")
    return total_imgs, total_lbls


def write_subset_yaml(
    src_yaml: str,
    dst_yaml: str,
    new_path: str,
):
    """基于源 yaml 生成新 yaml,只替换 path 字段。"""
    with open(src_yaml) as f:
        content = f.read()
    # 简单替换 path: 行
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith("path:"):
            new_lines.append(f"path: {new_path}")
        else:
            new_lines.append(line)
    with open(dst_yaml, "w") as f:
        f.write("\n".join(new_lines))
    print(f"生成配置: {dst_yaml} (path={new_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从大型 YOLO 数据集抽取小子集快速验证 pipeline"
    )
    parser.add_argument(
        "--src",
        default="data/processed/apsma_aligned_yolo",
        help="源 YOLO 数据集根目录",
    )
    parser.add_argument(
        "--dst",
        default="data/processed/apsma_yolo_subset",
        help="输出小子集目录",
    )
    parser.add_argument(
        "--n-per-split",
        type=int,
        default=500,
        help="每个 split 抽取的图像数",
    )
    parser.add_argument(
        "--src-yaml",
        default="config/yolo_training.yaml",
    )
    parser.add_argument(
        "--dst-yaml",
        default="config/yolo_training_subset.yaml",
    )
    args = parser.parse_args()

    n_imgs, n_lbls = subset_yolo_data(
        src_root=args.src,
        dst_root=args.dst,
        n_per_split=args.n_per_split,
    )

    # 生成对应 yaml 配置
    abs_dst = os.path.abspath(args.dst)
    write_subset_yaml(args.src_yaml, args.dst_yaml, abs_dst)
