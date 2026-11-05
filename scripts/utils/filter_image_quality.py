"""图片质量筛选: 过滤低质量图片(图标/分隔符/示意图)。

策略:
1. 按文件大小过滤(<10KB 的可能是图标/分隔符)
2. 用 PIL 检查图片尺寸(宽高<100px 或长宽比>5:1 的过滤)
3. 生成质量报告

输入:
    data/processed/species_image_library/{Phylum}/{Species}/img_XXX.jpg

输出:
    data/processed/species_image_library/quality_report.json
    (不删除图片,只标记低质量图片)
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
import os

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data/processed/species_image_library"

# 质量阈值
MIN_FILE_SIZE = 10 * 1024  # 10KB
MIN_DIMENSION = 100  # 100px
MAX_ASPECT_RATIO = 5.0  # 5:1


def check_image_quality(img_path):
    """检查单个图片质量。

    Returns:
        (is_good, reason, file_size, width, height)
    """
    # 1. 检查文件大小
    try:
        file_size = img_path.stat().st_size
    except Exception:
        return False, "stat_failed", 0, 0, 0

    if file_size < MIN_FILE_SIZE:
        return False, "too_small", file_size, 0, 0

    # 2. 用 PIL 检查图片尺寸
    try:
        from PIL import Image
        # 解析符号链接
        real_path = img_path.resolve() if img_path.is_symlink() else img_path
        with Image.open(real_path) as img:
            width, height = img.size
    except Exception as e:
        # PIL 不可用或图片损坏,仅按大小判断
        return True, "size_ok_no_pil", file_size, 0, 0

    # 3. 检查尺寸
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        return False, "dimension_too_small", file_size, width, height

    # 4. 检查长宽比
    aspect = max(width, height) / min(width, height)
    if aspect > MAX_ASPECT_RATIO:
        return False, "aspect_ratio_too_high", file_size, width, height

    return True, "good", file_size, width, height


def main():
    print("=" * 70)
    print("图片质量筛选")
    print("=" * 70)

    results = []
    quality_stats = Counter()
    size_stats = {"min": float('inf'), "max": 0, "avg": 0, "total": 0}
    dimension_stats = {"min_width": float('inf'), "min_height": float('inf'),
                       "max_width": 0, "max_height": 0}
    by_phylum = defaultdict(lambda: {"good": 0, "bad": 0, "reasons": Counter()})

    total = 0
    for phylum_dir in IMAGE_LIBRARY_DIR.iterdir():
        if not phylum_dir.is_dir():
            continue
        phylum = phylum_dir.name

        # 遍历物种目录
        for species_dir in phylum_dir.iterdir():
            if not species_dir.is_dir():
                continue
            # 遍历图片
            for img_file in species_dir.iterdir():
                if not (img_file.is_symlink() or img_file.is_file()):
                    continue
                total += 1

                is_good, reason, fsize, w, h = check_image_quality(img_file)

                if is_good:
                    quality_stats["good"] += 1
                    by_phylum[phylum]["good"] += 1
                else:
                    quality_stats["bad"] += 1
                    by_phylum[phylum]["bad"] += 1
                    by_phylum[phylum]["reasons"][reason] += 1

                # 更新统计
                if fsize > 0:
                    size_stats["total"] += fsize
                    if fsize < size_stats["min"]:
                        size_stats["min"] = fsize
                    if fsize > size_stats["max"]:
                        size_stats["max"] = fsize

                if w > 0 and h > 0:
                    if w < dimension_stats["min_width"]:
                        dimension_stats["min_width"] = w
                    if h < dimension_stats["min_height"]:
                        dimension_stats["min_height"] = h
                    if w > dimension_stats["max_width"]:
                        dimension_stats["max_width"] = w
                    if h > dimension_stats["max_height"]:
                        dimension_stats["max_height"] = h

                # 记录低质量图片
                if not is_good:
                    results.append({
                        "path": str(img_file.relative_to(IMAGE_LIBRARY_DIR)),
                        "reason": reason,
                        "file_size": fsize,
                        "width": w,
                        "height": h,
                    })

                if total % 10000 == 0:
                    print(f"  已检查 {total:,} 张图片...")

    size_stats["avg"] = size_stats["total"] / total if total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"✅ 质量筛选完成!")
    print(f"  总图片数: {total:,}")
    print(f"  高质量: {quality_stats['good']:,} ({quality_stats['good']/total*100:.1f}%)")
    print(f"  低质量: {quality_stats['bad']:,} ({quality_stats['bad']/total*100:.1f}%)")

    print(f"\n文件大小统计:")
    print(f"  最小: {size_stats['min']/1024:.1f} KB")
    print(f"  最大: {size_stats['max']/1024/1024:.1f} MB")
    print(f"  平均: {size_stats['avg']/1024:.1f} KB")

    print(f"\n尺寸统计:")
    print(f"  最小宽×高: {dimension_stats['min_width']}×{dimension_stats['min_height']}")
    print(f"  最大宽×高: {dimension_stats['max_width']}×{dimension_stats['max_height']}")

    print(f"\n按门级分布:")
    print(f"  {'门级':25s} {'高质量':>8} {'低质量':>8} {'高质量%':>8}")
    for phylum in sorted(by_phylum.keys()):
        info = by_phylum[phylum]
        total_p = info["good"] + info["bad"]
        rate = info["good"]/total_p*100 if total_p > 0 else 0
        print(f"  {phylum:25s} {info['good']:>8,} {info['bad']:>8,} {rate:>7.1f}%")

    # 保存报告
    report = {
        "total_images": total,
        "good_quality": quality_stats["good"],
        "bad_quality": quality_stats["bad"],
        "quality_rate": quality_stats["good"] / total if total > 0 else 0,
        "size_stats": {k: v for k, v in size_stats.items() if k != "inf"},
        "dimension_stats": {k: v for k, v in dimension_stats.items()
                           if not (isinstance(v, float) and v == float('inf'))},
        "bad_images": results[:1000],  # 只保存前1000个低质量图片
        "total_bad": len(results),
    }
    report_path = IMAGE_LIBRARY_DIR / "quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告: {report_path}")
    print(f"  低质量图片总数: {len(results):,}")


if __name__ == "__main__":
    main()
