"""清理 262 个非 APSMA 21 门物种(硅藻/植物/未收录)。

从 species_image_library/Unknown/ 和 species_knowledge_base/descriptions/Unknown/
删除不属于 21 门的物种符号链接。

输入:
    data/processed/species_image_library/Unknown/  (262 物种)
    data/processed/species_knowledge_base/descriptions/Unknown/

输出:
    - 删除 Unknown/ 目录下的所有非21门物种
    - 如果 Unknown/ 变空,删除目录本身
    - 生成清理报告
"""
import json
import shutil
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data" / "processed" / "species_image_library"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "species_knowledge_base"


def clean_unknown_images():
    """清理图片库 Unknown 目录。"""
    unknown_dir = IMAGE_LIBRARY_DIR / "Unknown"
    if not unknown_dir.exists():
        print("  Unknown/ 目录不存在,跳过")
        return 0, 0

    removed_species = 0
    removed_files = 0

    for species_dir in list(unknown_dir.iterdir()):
        if species_dir.is_dir():
            n = sum(1 for _ in species_dir.iterdir())
            shutil.rmtree(species_dir)
            removed_species += 1
            removed_files += n

    # 如果 Unknown 目录为空,删除它
    if unknown_dir.exists() and not any(unknown_dir.iterdir()):
        unknown_dir.rmdir()
        print(f"  Unknown/ 目录已清空并删除")

    return removed_species, removed_files


def clean_unknown_descriptions():
    """清理知识库 Unknown 目录。"""
    unknown_dir = KNOWLEDGE_BASE_DIR / "descriptions" / "Unknown"
    if not unknown_dir.exists():
        print("  descriptions/Unknown/ 目录不存在,跳过")
        return 0

    removed = 0
    for f in list(unknown_dir.iterdir()):
        if f.is_symlink() or f.is_file():
            f.unlink()
            removed += 1

    if unknown_dir.exists() and not any(unknown_dir.iterdir()):
        unknown_dir.rmdir()
        print(f"  descriptions/Unknown/ 目录已清空并删除")

    return removed


def main():
    print("=" * 70)
    print("清理非 21 门 Unknown 物种(硅藻/植物/未收录)")
    print("=" * 70)

    # 先统计当前 Unknown 数量
    img_unknown = IMAGE_LIBRARY_DIR / "Unknown"
    desc_unknown = KNOWLEDGE_BASE_DIR / "descriptions" / "Unknown"

    img_count = sum(1 for _ in img_unknown.iterdir()) if img_unknown.exists() else 0
    desc_count = sum(1 for _ in desc_unknown.iterdir()) if desc_unknown.exists() else 0
    print(f"\n清理前:")
    print(f"  图片库 Unknown 物种: {img_count}")
    print(f"  知识库 Unknown 描述: {desc_count}")

    print(f"\n清理图片库...")
    rm_sp, rm_img = clean_unknown_images()
    print(f"  删除 {rm_sp} 物种, {rm_img} 图片")

    print(f"\n清理知识库...")
    rm_desc = clean_unknown_descriptions()
    print(f"  删除 {rm_desc} 描述")

    print(f"\n✅ 清理完成!")
    print(f"  删除图片: {rm_img}")
    print(f"  删除描述: {rm_desc}")

    # 生成报告
    report = {
        "cleaned_image_species": rm_sp,
        "cleaned_image_files": rm_img,
        "cleaned_description_files": rm_desc,
        "reason": "262 个 Unknown 物种为硅藻(Heterokontophyta)/植物(Tracheophyta)/未收录,不在 APSMA 21门中",
    }
    report_path = PROJECT_ROOT / "data/processed/species_image_library/unknown_cleanup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {report_path}")


if __name__ == "__main__":
    main()
