"""删除低质量图片(图标/分隔符/示意图)。

筛选规则(与 filter_image_quality.py 一致):
1. 文件大小 < 10KB
2. 尺寸 < 100px (宽或高)
3. 长宽比 > 5:1

输入:
    data/processed/species_image_library/{Phylum}/{Species}/img_XXX.jpg

输出:
    - 删除低质量图片(符号链接 unlink,不删除原始文件)
    - data/processed/species_image_library/cleanup_report.json
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
import os

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data" / "processed" / "species_image_library"

# 质量阈值(与 filter_image_quality.py 一致)
MIN_FILE_SIZE = 10 * 1024  # 10KB
MIN_DIMENSION = 100  # 100px
MAX_ASPECT_RATIO = 5.0  # 5:1


def check_image_quality(img_path):
    """检查单个图片质量。"""
    try:
        file_size = img_path.stat().st_size
    except Exception:
        return False, "stat_failed", 0, 0, 0

    if file_size < MIN_FILE_SIZE:
        return False, "too_small", file_size, 0, 0

    try:
        from PIL import Image
        real_path = img_path.resolve() if img_path.is_symlink() else img_path
        with Image.open(real_path) as img:
            width, height = img.size
    except Exception:
        return True, "size_ok_no_pil", file_size, 0, 0

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        return False, "dimension_too_small", file_size, width, height

    aspect = max(width, height) / min(width, height)
    if aspect > MAX_ASPECT_RATIO:
        return False, "aspect_ratio_too_high", file_size, width, height

    return True, "good", file_size, width, height


def main():
    print("=" * 70)
    print("删除低质量图片")
    print("=" * 70)

    deleted = []
    by_phylum = defaultdict(lambda: {"deleted": 0, "reasons": Counter()})
    by_reason = Counter()
    total_checked = 0
    species_with_deleted = defaultdict(int)  # phylum/species -> 删除数

    # 遍历所有图片
    for phylum_dir in IMAGE_LIBRARY_DIR.iterdir():
        if not phylum_dir.is_dir():
            continue
        phylum = phylum_dir.name

        if phylum == "中国动物志":
            # 中国动物志卷目录
            for vol_dir in phylum_dir.iterdir():
                if not vol_dir.is_dir():
                    continue
                for img_file in list(vol_dir.iterdir()):
                    if not (img_file.is_symlink() or img_file.is_file()):
                        continue
                    total_checked += 1
                    is_good, reason, fsize, w, h = check_image_quality(img_file)
                    if not is_good:
                        try:
                            img_file.unlink()
                            deleted.append({
                                "path": str(img_file.relative_to(IMAGE_LIBRARY_DIR)),
                                "reason": reason,
                                "file_size": fsize,
                            })
                            by_reason[reason] += 1
                            by_phylum[phylum]["deleted"] += 1
                            by_phylum[phylum]["reasons"][reason] += 1
                        except Exception as e:
                            print(f"  ⚠️ 删除失败 {img_file}: {e}")
                    if total_checked % 5000 == 0:
                        print(f"  已检查 {total_checked:,} 张 (已删除 {len(deleted):,})")
            continue

        # WoRMS 物种目录
        for species_dir in phylum_dir.iterdir():
            if not species_dir.is_dir():
                continue
            species_key = f"{phylum}/{species_dir.name}"
            for img_file in list(species_dir.iterdir()):
                if not (img_file.is_symlink() or img_file.is_file()):
                    continue
                total_checked += 1
                is_good, reason, fsize, w, h = check_image_quality(img_file)
                if not is_good:
                    try:
                        img_file.unlink()
                        deleted.append({
                            "path": str(img_file.relative_to(IMAGE_LIBRARY_DIR)),
                            "reason": reason,
                            "file_size": fsize,
                        })
                        by_reason[reason] += 1
                        by_phylum[phylum]["deleted"] += 1
                        by_phylum[phylum]["reasons"][reason] += 1
                        species_with_deleted[species_key] += 1
                    except Exception as e:
                        print(f"  ⚠️ 删除失败 {img_file}: {e}")
                if total_checked % 5000 == 0:
                    print(f"  已检查 {total_checked:,} 张 (已删除 {len(deleted):,})")

    print(f"\n{'=' * 70}")
    print(f"✅ 清理完成!")
    print(f"  总检查: {total_checked:,}")
    print(f"  已删除: {len(deleted):,} ({len(deleted)/total_checked*100:.2f}%)")

    print(f"\n按原因分布:")
    for reason, count in by_reason.most_common():
        print(f"  {reason:25s}: {count:,}")

    print(f"\n按门级分布:")
    for phylum in sorted(by_phylum.keys()):
        info = by_phylum[phylum]
        print(f"  {phylum:25s}: {info['deleted']:>6,} 张")
        for r, c in info["reasons"].most_common():
            print(f"    └─ {r:20s}: {c:,}")

    # 保存报告
    report = {
        "total_checked": total_checked,
        "total_deleted": len(deleted),
        "by_reason": dict(by_reason),
        "by_phylum": {p: {"deleted": v["deleted"], "reasons": dict(v["reasons"])}
                     for p, v in by_phylum.items()},
        "species_affected": len(species_with_deleted),
        "deleted_images": deleted,  # 完整列表
    }
    report_path = IMAGE_LIBRARY_DIR / "cleanup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告: {report_path}")
    print(f"  受影响物种数: {len(species_with_deleted)}")


if __name__ == "__main__":
    main()
