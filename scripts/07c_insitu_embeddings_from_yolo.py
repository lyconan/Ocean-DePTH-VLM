#!/usr/bin/env python3
"""从已下载的 FathomNet YOLO 数据提取 in-situ 嵌入 (v1.0)

替代 07b 的在线 FathomNet API 方案，用本地已下载的 FathomNet YOLO 图像裁剪：
1. 遍历 FathomNet YOLO 图像 + YOLO 标注 (YOLO 格式 bbox)
2. 裁剪标注区域 → CLIP 嵌入 (512维)
3. 用 Neo4j exsitu 向量检索匹配 APSMA 形态型
4. 将 in-situ 裁剪嵌入分配给最相似的形态型 (score >= 阈值)
5. 批量写入 Neo4j insitu_embedding_primary + insitu_embeddings_json + insitu_image_paths

用法:
    python scripts/07c_insitu_embeddings_from_yolo.py \\
        --yolo-dir data/processed/fathomnet_yolo \\
        --neo4j-uri bolt://localhost:7687 \\
        --neo4j-user neo4j --neo4j-password password \\
        --top-n 3 --score-threshold 0.45

依赖: torch, PIL, neo4j, transformers (CLIP 回退)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from neo4j import GraphDatabase
from utils.bioclip_embed import get_bioclip_model, extract_embedding as _extract_emb


# ═══════════════════════════════════════════════════════════
# CLIP 模型加载 (委托给 utils.bioclip_embed, 兼容 BioCLIP/open_clip)
# ═══════════════════════════════════════════════════════════

def get_clip_model():
    """懒加载 BioCLIP (或 CLIP 回退) 模型。委托给共享工具。"""
    return get_bioclip_model()


def extract_embedding(crop_pil: Image.Image) -> List[float]:
    """计算单张裁剪图片的 BioCLIP/CLIP 嵌入向量 (512维, 归一化)。"""
    return _extract_emb(crop_pil).tolist()


# ═══════════════════════════════════════════════════════════
# YOLO 标注解析 + 裁剪
# ═══════════════════════════════════════════════════════════

def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    """解析 YOLO 格式标注 → 像素坐标 bbox 列表。

    YOLO 格式: class_id x_center y_center width height (归一化)
    返回: [(class_id, x1, y1, x2, y2), ...] (像素坐标)
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            # 边界保护
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            if x2 > x1 and y2 > y1:
                bboxes.append((cls_id, x1, y1, x2, y2))
    return bboxes


def collect_insitu_crops(
    yolo_dir: str, max_crops: int = 500
) -> List[Tuple[str, Image.Image, int]]:
    """从 FathomNet YOLO 数据收集 in-situ 裁剪。

    返回: [(image_path, crop_pil, class_id), ...]
    """
    images_dir = Path(yolo_dir) / "images"
    labels_dir = Path(yolo_dir) / "labels"

    crops = []
    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))

    for img_path in image_files:
        if len(crops) >= max_crops:
            break

        label_path = labels_dir / (img_path.stem + ".txt")
        try:
            img = Image.open(str(img_path)).convert("RGB")
        except Exception:
            continue

        img_w, img_h = img.size
        bboxes = parse_yolo_label(str(label_path), img_w, img_h)

        for cls_id, x1, y1, x2, y2 in bboxes:
            if len(crops) >= max_crops:
                break
            try:
                crop = img.crop((x1, y1, x2, y2))
                # 过滤过小的裁剪
                if crop.width < 20 or crop.height < 20:
                    continue
                crops.append((str(img_path), crop, cls_id))
            except Exception:
                continue

    print(f"  收集 {len(crops)} 个 in-situ 裁剪 (上限 {max_crops})")
    return crops


# ═══════════════════════════════════════════════════════════
# Neo4j 向量匹配 + 嵌入写入
# ═══════════════════════════════════════════════════════════

def match_to_morphotype(
    driver, embedding: List[float], score_threshold: float = 0.45
) -> Optional[Dict]:
    """用 exsitu 向量检索找到最相似的 APSMA 形态型。"""
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('morphotype_exsitu_embeddings', 1, $embedding)
            YIELD node, score
            RETURN node.morphotype_id AS id,
                   node.name AS name,
                   node.phylum_name AS phylum,
                   score
            LIMIT 1
        """, embedding=embedding)
        record = result.single()
        if record and record["score"] >= score_threshold:
            return dict(record)
    return None


def write_insitu_embeddings_to_neo4j(
    driver, insitu_map: Dict[str, Dict]
) -> int:
    """将 in-situ 嵌入写入 Neo4j Morphotype 节点。

    insitu_map: {morphotype_id: {"embeddings": [[...], ...], "paths": [..., ...]}}
    返回: 更新的节点数
    """
    updated = 0
    with driver.session() as session:
        for mid, data in insitu_map.items():
            embeddings = data["embeddings"]
            paths = data["paths"]

            # insitu_embedding_primary = 第一个嵌入 (向量索引目标)
            primary = embeddings[0] if embeddings else None
            emb_json = json.dumps(embeddings)

            session.run("""
                MATCH (m:Morphotype {morphotype_id: $mid})
                SET m.insitu_embedding_primary = $primary,
                    m.insitu_embeddings_json = $emb_json,
                    m.insitu_image_paths = $paths
            """, mid=mid, primary=primary, emb_json=emb_json, paths=paths)
            updated += 1

    return updated


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="从 FathomNet YOLO 数据提取 in-situ 嵌入 (离线方案)"
    )
    parser.add_argument(
        "--yolo-dir", default="data/processed/fathomnet_yolo",
        help="FathomNet YOLO 数据目录 (含 images/ 和 labels/)"
    )
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--top-n", type=int, default=3, help="每个形态型保留的 in-situ 裁剪数")
    parser.add_argument("--score-threshold", type=float, default=0.45,
                        help="exsitu 向量匹配分数阈值")
    parser.add_argument("--max-crops", type=int, default=500,
                        help="最多处理的裁剪数 (控制时间)")
    parser.add_argument("--output", default="data/processed/fathomnet_insitu_crops/insitu_embeddings.pt",
                        help="嵌入 .pt 文件输出路径 (兼容 09 脚本)")
    args = parser.parse_args()

    print("=" * 70)
    print("In-situ 嵌入提取 (FathomNet YOLO 离线方案)")
    print("=" * 70)

    # 1. 加载 CLIP
    print("\n[1/5] 加载 CLIP 嵌入模型...")
    get_clip_model()

    # 2. 连接 Neo4j
    print("\n[2/5] 连接 Neo4j...")
    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )
    with driver.session() as session:
        count = session.run("MATCH (m:Morphotype) RETURN count(m) AS c").single()["c"]
        print(f"  Morphotype 节点: {count}")

    # 3. 收集 in-situ 裁剪
    print(f"\n[3/5] 收集 in-situ 裁剪 (上限 {args.max_crops})...")
    yolo_abs = str(PROJECT_ROOT / args.yolo_dir) if not Path(args.yolo_dir).is_absolute() else args.yolo_dir
    crops = collect_insitu_crops(yolo_abs, max_crops=args.max_crops)

    if not crops:
        print("[错误] 未收集到任何裁剪!")
        driver.close()
        return

    # 4. 提取嵌入 + 匹配形态型
    print(f"\n[4/5] 提取嵌入 + 匹配形态型 (阈值={args.score_threshold})...")
    insitu_map: Dict[str, Dict] = {}  # {mid: {"embeddings": [], "paths": []}}
    matched_count = 0
    skipped_count = 0

    for i, (img_path, crop, cls_id) in enumerate(crops):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(crops)} | 已匹配 {len(insitu_map)} 形态型")

        try:
            embedding = extract_embedding(crop)
        except Exception as e:
            print(f"  嵌入提取失败: {e}")
            continue

        match = match_to_morphotype(driver, embedding, args.score_threshold)
        if match:
            mid = match["id"]
            if mid not in insitu_map:
                insitu_map[mid] = {"embeddings": [], "paths": []}

            # 每个形态型最多保留 top_n 个
            if len(insitu_map[mid]["embeddings"]) < args.top_n:
                insitu_map[mid]["embeddings"].append(embedding)
                insitu_map[mid]["paths"].append(img_path)
                matched_count += 1
        else:
            skipped_count += 1

    print(f"  匹配: {matched_count} 裁剪 → {len(insitu_map)} 形态型")
    print(f"  跳过: {skipped_count} 裁剪 (score < {args.score_threshold})")

    # 5. 写入 Neo4j
    print(f"\n[5/5] 写入 Neo4j insitu_embedding_primary...")
    updated = write_insitu_embeddings_to_neo4j(driver, insitu_map)
    print(f"  更新 {updated} 个 Morphotype 节点")

    # 保存 .pt 文件 (兼容 09 脚本)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save(insitu_map, args.output)
        print(f"  嵌入保存: {args.output}")

    driver.close()

    # 汇总
    print("\n" + "=" * 70)
    print("In-situ 嵌入提取完成")
    print("=" * 70)
    print(f"处理裁剪:     {len(crops)}")
    print(f"匹配形态型:   {len(insitu_map)}")
    print(f"Neo4j 更新:   {updated} 节点")
    print(f"每个形态型:   最多 {args.top_n} 个 in-situ 锚点")
    print(f"匹配阈值:     {args.score_threshold}")

    # 检查向量索引填充率
    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )
    with driver.session() as session:
        filled = session.run(
            "MATCH (m:Morphotype) WHERE m.insitu_embedding_primary IS NOT NULL "
            "RETURN count(m) AS c"
        ).single()["c"]
        print(f"insitu 填充率: {filled}/{count} ({100*filled/count:.1f}%)")
    driver.close()

    print("\n✓ In-situ 嵌入提取完成!")


if __name__ == "__main__":
    main()
