"""BenthicNet数据下载和筛选 — 深渊区(>3500m)太平洋底栖图像"""
import os
import argparse
import pandas as pd
from pathlib import Path


def filter_benthicnet(
    metadata_csv: str,
    annotations_csv: str = None,
    output_dir: str = "data/raw/benthicnet_raw",
    min_depth: float = 3500,
    longitude_range: tuple = None,
    limit: int = None,
):
    """筛选BenthicNet深渊区底栖图像。

    Args:
        metadata_csv: BenthicNet标注CSV路径(benthicnet_labelled.csv)
        annotations_csv: BenthicNet分类学CSV路径(可选,benthicnet_taxonomies.csv)
        output_dir: 输出目录
        min_depth: 最小深度(米)，默认3500m(深渊区)
        longitude_range: 经度范围(可选,None表示不限制)
        limit: 最大图像数(可选)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取标注数据: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)
    print(f"总标注数: {len(metadata)}")

    # 深度筛选 — BenthicNet使用 gebco_bathymetry 列(负值表示海面以下深度)
    depth_col = None
    for col in ['depth_m', 'gebco_bathymetry', 'depth']:
        if col in metadata.columns:
            depth_col = col
            break
    if depth_col:
        # gebco_bathymetry 通常为负值(海平面以下),取绝对值
        depth_vals = metadata[depth_col].abs()
        deep = metadata[depth_vals > min_depth]
        print(f"深度>{min_depth}m (列={depth_col}): {len(deep)} 标注")
    else:
        print("警告: 未找到深度列(depth_m/gebco_bathymetry/depth)，跳过深度筛选")
        deep = metadata

    # 经度筛选(可选)
    if longitude_range and 'longitude' in deep.columns:
        pacific_mask = pd.Series(False, index=deep.index)
        for lon_min, lon_max in longitude_range:
            pacific_mask |= (deep['longitude'] >= lon_min) & (deep['longitude'] <= lon_max)
        pacific = deep[pacific_mask]
        print(f"指定经度区域: {len(pacific)} 标注")
    else:
        print("未限制经度范围(保留所有深海数据)")
        pacific = deep

    # 筛选有生物标注的图像 — BenthicNet使用 catami_biota 列
    biota_col = None
    for col in ['catami_biota', 'category_type']:
        if col in pacific.columns:
            biota_col = col
            break
    if biota_col:
        # 过滤掉空生物标注(非生物标签如NA/空)
        biota = pacific[pacific[biota_col].notna() & (pacific[biota_col] != '')]
        # 进一步过滤: 排除非生物标签(如"None","Unknown Geology"等)
        non_biota_labels = {'None', '', 'NA', 'Unknown Geology', 'Unknown'}
        biota = biota[~biota[biota_col].isin(non_biota_labels)]
        print(f"有生物标注 (列={biota_col}): {len(biota)} 标注")
        filtered = biota
    else:
        print("警告: 未找到catami_biota/category_type列，跳过生物标注筛选")
        filtered = pacific

    if limit and len(filtered) > limit:
        filtered = filtered.sample(n=limit, random_state=42)
        print(f"限制为: {limit} 图像")

    # 保存筛选结果
    output_csv = output_dir / "benthicnet_filtered.csv"
    filtered.to_csv(output_csv, index=False)
    print(f"筛选结果保存: {output_csv} ({len(filtered)} 图像)")

    # 统计
    for col in ['depth_m', 'gebco_bathymetry']:
        if col in filtered.columns:
            print(f"深度范围 (列={col}): {filtered[col].abs().min():.0f} - {filtered[col].abs().max():.0f} m")
            break

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="筛选BenthicNet数据")
    parser.add_argument("--metadata", required=True, help="BenthicNet元数据CSV")
    parser.add_argument("--annotations", default=None, help="BenthicNet标注CSV")
    parser.add_argument("--output", default="data/raw/benthicnet_raw")
    parser.add_argument("--min-depth", type=float, default=3500)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    filter_benthicnet(
        metadata_csv=args.metadata,
        annotations_csv=args.annotations,
        output_dir=args.output,
        min_depth=args.min_depth,
        limit=args.limit,
    )
