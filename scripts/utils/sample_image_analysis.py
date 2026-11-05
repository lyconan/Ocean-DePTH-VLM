"""采样分析图片内容类型,识别非生物图片.

策略:
1. 随机采样 N 张图片
2. 用 PIL 分析图片特征:
   - 颜色分布(彩色 vs 黑白)
   - 边缘密度(自然纹理 vs 规则线条)
   - 长宽比
   - 文件大小
3. 按特征分类: 生物照片 / 图表/地图 / 示意图
"""
import os
import random
import json
from pathlib import Path
from collections import defaultdict
import math

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
IMG_DIR = PROJECT_ROOT / "data/processed/species_image_library"

# 收集所有图片路径
all_imgs = []
for phylum_dir in IMG_DIR.iterdir():
    if not phylum_dir.is_dir() or phylum_dir.name == "中国动物志":
        continue
    for species_dir in phylum_dir.iterdir():
        if not species_dir.is_dir():
            continue
        for img in species_dir.iterdir():
            if img.is_symlink() or img.is_file():
                all_imgs.append(img)

print(f"总图片数: {len(all_imgs)}")

# 随机采样 500 张
random.seed(42)
SAMPLE_SIZE = 500
sample = random.sample(all_imgs, min(SAMPLE_SIZE, len(all_imgs)))

from PIL import Image
import numpy as np

# 边缘检测函数(简化版 Sobel)
def edge_density(img_array):
    """计算图片边缘密度(自然照片 ~0.05-0.15,图表 >0.2)."""
    if len(img_array.shape) == 3:
        gray = img_array.mean(axis=2)
    else:
        gray = img_array
    h, w = gray.shape
    if h < 10 or w < 10:
        return 0
    # Sobel 算子
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(gx**2 + gy**2)
    # 边缘强度大于阈值的像素比例
    threshold = edges.mean() * 2 + 10
    return (edges > threshold).mean()

# 颜色复杂度(独特颜色数)
def color_complexity(img):
    """采样 100x100 缩略图,计算独特颜色数."""
    thumb = img.convert("RGB").resize((100, 100))
    arr = np.array(thumb)
    # 量化到 8 级
    quantized = (arr // 32) * 32
    unique = len(np.unique(quantized.reshape(-1, 3), axis=0))
    return unique

stats = {
    "color": 0, "grayscale": 0,
    "very_small": 0, "small": 0, "medium": 0, "large": 0,
    "square": 0, "landscape": 0, "portrait": 0,
}

buckets = defaultdict(list)

for i, img_path in enumerate(sample):
    if i % 50 == 0:
        print(f"  处理 {i}/{len(sample)}...")
    try:
        real = img_path.resolve() if img_path.is_symlink() else img_path
        file_size = real.stat().st_size

        with Image.open(real) as img:
            w, h = img.size
            mode = img.mode
            # 转为 numpy 数组
            arr = np.array(img.convert("RGB"))
            # 计算颜色复杂度
            thumb = img.convert("RGB").resize((50, 50))
            arr_small = np.array(thumb)
            unique_colors = len(np.unique((arr_small // 32) * 32, axis=0).reshape(-1, 3))
            # 判断是否彩色(各通道差异)
            r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
            channel_diff = np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()
            is_color = channel_diff > 5
            # 边缘密度
            edge_d = edge_density(arr)

        if is_color:
            stats["color"] += 1
        else:
            stats["grayscale"] += 1

        if file_size < 30 * 1024:
            stats["very_small"] += 1
        elif file_size < 100 * 1024:
            stats["small"] += 1
        elif file_size < 500 * 1024:
            stats["medium"] += 1
        else:
            stats["large"] += 1

        aspect = w / h if h > 0 else 1
        if 0.9 <= aspect <= 1.1:
            stats["square"] += 1
        elif aspect > 1.5:
            stats["landscape"] += 1
        elif aspect < 0.67:
            stats["portrait"] += 1

        # 分类
        rel_path = str(img_path.relative_to(PROJECT_ROOT))
        info = {
            "path": rel_path, "size_kb": round(file_size/1024, 1),
            "w": w, "h": h, "mode": mode, "aspect": round(aspect, 2),
            "is_color": bool(is_color), "channel_diff": round(float(channel_diff), 2),
            "unique_colors": int(unique_colors), "edge_density": round(float(edge_d), 3),
        }

        # 启发式分类
        if aspect > 3 or aspect < 0.33:
            buckets["very_wide"].append(info)  # 极长宽比,可能分布图
        elif not is_color and unique_colors < 5:
            buckets["binary_chart"].append(info)  # 黑白简单图,可能表格/示意图
        elif not is_color and edge_d > 0.15:
            buckets["gray_line_chart"].append(info)  # 黑白高边缘,可能线条图
        elif is_color and file_size > 50 * 1024 and unique_colors > 50:
            buckets["likely_photo"].append(info)  # 彩色+大文件+多颜色,可能生物照片
        elif is_color and unique_colors < 20:
            buckets["color_simple"].append(info)  # 彩色但颜色少,可能示意图
        else:
            buckets["other"].append(info)
    except Exception as e:
        pass

print(f"\n=== 采样 {len(sample)} 张图片分析 ===")
print(f"彩色: {stats['color']} ({stats['color']/len(sample)*100:.1f}%)")
print(f"黑白: {stats['grayscale']} ({stats['grayscale']/len(sample)*100:.1f}%)")
print(f"\n文件大小分布:")
print(f"  <30KB: {stats['very_small']} ({stats['very_small']/len(sample)*100:.1f}%)")
print(f"  30-100KB: {stats['small']} ({stats['small']/len(sample)*100:.1f}%)")
print(f"  100-500KB: {stats['medium']} ({stats['medium']/len(sample)*100:.1f}%)")
print(f"  >500KB: {stats['large']} ({stats['large']/len(sample)*100:.1f}%)")
print(f"\n长宽比分布:")
print(f"  方形: {stats['square']} ({stats['square']/len(sample)*100:.1f}%)")
print(f"  横向: {stats['landscape']} ({stats['landscape']/len(sample)*100:.1f}%)")
print(f"  纵向: {stats['portrait']} ({stats['portrait']/len(sample)*100:.1f}%)")

print(f"\n=== 启发式分类 ===")
total_classified = sum(len(v) for v in buckets.values())
for k, v in sorted(buckets.items(), key=lambda x: -len(x[1])):
    pct = len(v) / len(sample) * 100
    print(f"  {k:20s}: {len(v):4d} ({pct:.1f}%)")

# 推算整体
print(f"\n=== 推算 130,579 张图片整体 ===")
for k, v in buckets.items():
    pct = len(v) / len(sample) * 100
    estimated = int(pct / 100 * 130579)
    print(f"  {k:20s}: ~{estimated:,} 张 ({pct:.1f}%)")

# 保存详细分析报告
report = {
    "sample_size": len(sample),
    "total_images": len(all_imgs),
    "stats": stats,
    "buckets": dict(buckets),
    "estimated_distribution": {
        k: int(len(v) / len(sample) * 130579)
        for k, v in buckets.items()
    },
}
report_path = PROJECT_ROOT / "data/processed/species_image_library/content_analysis.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n报告: {report_path}")

# 保存采样示例图(供人工核查)
print(f"\n=== 各类别示例(前5个) ===")
for k, v in buckets.items():
    print(f"\n[{k}] ({len(v)} 张)")
    for info in v[:5]:
        print(f"  {info['path'][:90]}")
        print(f"    size={info['size_kb']}KB, {info['w']}x{info['h']}, "
              f"aspect={info['aspect']}, colors={info['unique_colors']}, "
              f"edge={info['edge_density']}")
