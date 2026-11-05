#!/usr/bin/env python3
"""提取 FathomNet in-situ 嵌入 — 裁剪 bbox 区域的 BioCLIP 嵌入 (v1.0)

核心改进: 之前的 ex-situ 嵌入来自整张图片或论文插图, 与真实海底照片域偏移大。
本脚本从 FathomNet 训练集的真实海底照片中裁剪 bbox 区域, 提取 in-situ 嵌入,
存入 Neo4j 的 insitu_embedding 属性, 并创建独立的向量索引。

流程:
1. 加载 FathomNet COCO 标注 (fathomnet_train.json)
2. 对每个类别 (≥50 样本), 取最多 N 张图像, 裁剪 bbox 区域
3. 对裁剪后的 patch 提取 BioCLIP 嵌入, 取平均并归一化
4. 在 Neo4j 中更新 FN_ 节点的 insitu_embedding 属性
5. 创建 morphotype_insitu_embeddings 向量索引

用法:
    # 完整提取 + 导入
    python scripts/07f_extract_insitu_embeddings.py

    # 只提取不导入 (调试)
    python scripts/07f_extract_insitu_embeddings.py --dry-run

    # 自定义参数
    python scripts/07f_extract_insitu_embeddings.py --max-crops 10 --min-samples 50
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.bioclip_embed import get_bioclip_model, extract_embedding as _extract_emb

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

FATHOMNET_DIR = Path("/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet")
EMBEDDING_DIM = 512


def extract_insitu_embeddings(
    fathomnet_dir: Path,
    model,
    processor,
    min_samples: int = 50,
    max_crops_per_category: int = 10,
    min_crop_size: int = 32,
    padding_ratio: float = 0.1,
) -> Dict[str, dict]:
    """从 FathomNet 训练集裁剪 bbox 区域, 提取 in-situ BioCLIP 嵌入。

    Args:
        fathomnet_dir: FathomNet 数据目录
        model: BioCLIP 模型
        processor: BioCLIP 预处理器
        min_samples: 类别最小样本数阈值
        max_crops_per_category: 每个类别最多裁剪的 bbox 数
        min_crop_size: bbox 最小边长 (像素), 太小的裁剪跳过
        padding_ratio: bbox 裁剪时向外扩展的比例 (提供上下文)

    Returns:
        {FN_<cat_name>: {embedding, n_crops, phylum, ...}}
    """
    from utils.bioclip_embed import is_bioclip
    print(f"[In-situ嵌入] BioCLIP: {is_bioclip()}")
    print(f"[In-situ嵌入] FathomNet: {fathomnet_dir}")
    print(f"[In-situ嵌入] 参数: min_samples={min_samples}, max_crops={max_crops_per_category}, "
          f"min_crop_size={min_crop_size}, padding={padding_ratio}")

    # 加载 COCO 标注
    coco_path = fathomnet_dir / "Annotations" / "fathomnet_train.json"
    print(f"[In-situ嵌入] 加载 COCO: {coco_path}")
    with open(coco_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}

    # 统计类别样本数 + 收集每类标注 (包含 bbox 信息)
    cat_counts = defaultdict(int)
    cat_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        cat_counts[cat_id] += 1
        if len(cat_annotations[cat_id]) < max_crops_per_category:
            cat_annotations[cat_id].append(ann)

    # 筛选 ≥ min_samples 的类别
    selected_cats = [(cid, cnt) for cid, cnt in cat_counts.items() if cnt >= min_samples]
    selected_cats.sort(key=lambda x: -x[1])
    print(f"[In-situ嵌入] 选中 {len(selected_cats)} 个类别 (≥{min_samples} 样本)")

    train_dir = fathomnet_dir / "Train"
    results = {}
    processed = 0
    failed = 0
    skipped_small = 0

    for cat_idx, (cat_id, cnt) in enumerate(selected_cats):
        cat_name = cat_map[cat_id]
        annotations = cat_annotations[cat_id]

        embeddings = []
        n_crops = 0

        for ann in annotations:
            img_id = ann["image_id"]
            img_info = img_map.get(img_id)
            if not img_info:
                continue

            file_name = img_info.get("file_name", "")
            img_path = train_dir / file_name
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img_w, img_h = img.size

                # COCO bbox: [x, y, width, height]
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    continue

                x, y, bw, bh = bbox

                # 添加 padding (提供上下文)
                pad_w = bw * padding_ratio
                pad_h = bh * padding_ratio
                x1 = max(0, int(x - pad_w))
                y1 = max(0, int(y - pad_h))
                x2 = min(img_w, int(x + bw + pad_w))
                y2 = min(img_h, int(y + bh + pad_h))

                # 跳过太小的裁剪
                crop_w = x2 - x1
                crop_h = y2 - y1
                if crop_w < min_crop_size or crop_h < min_crop_size:
                    skipped_small += 1
                    continue

                # 裁剪 bbox 区域
                crop = img.crop((x1, y1, x2, y2))
                emb = _extract_emb(crop, model, processor)
                embeddings.append(emb)
                n_crops += 1

            except Exception as e:
                if failed < 5:
                    print(f"  [跳过] {img_path}: {e}")
                failed += 1
                continue

        if not embeddings:
            failed += 1
            continue

        # 平均嵌入并归一化
        avg_emb = torch.stack(embeddings).mean(dim=0)
        norm = avg_emb.norm()
        if norm > 0:
            avg_emb = avg_emb / norm

        # 推断门级分类
        try:
            from utils.taxonomy_mapper import infer_phylum_from_concept
            phylum = infer_phylum_from_concept(cat_name)
        except Exception:
            phylum = "Unknown"

        species_key = f"FN_{cat_name}"
        results[species_key] = {
            "embedding": avg_emb.tolist(),
            "phylum": phylum,
            "species_name": cat_name,
            "n_crops": n_crops,
            "n_train_samples": cnt,
        }
        processed += 1

        if processed % 50 == 0:
            print(f"  [进度] {processed}/{len(selected_cats)} 类别, "
                  f"裁剪 {n_crops} 个, 跳过小框 {skipped_small}, 失败 {failed}")

    print(f"\n[In-situ嵌入] 完成: {processed} 个类别, {failed} 个失败, "
          f"{skipped_small} 个太小框跳过")
    return results


def import_insitu_to_neo4j(embeddings_data: Dict[str, dict]):
    """将 in-situ 嵌入导入 Neo4j, 更新 FN_ 节点的 insitu_embedding 属性。"""
    from neo4j import GraphDatabase

    print(f"\n[Neo4j导入] 连接: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    updated = 0
    created = 0
    failed = 0

    try:
        with driver.session() as session:
            # 检查导入前状态
            before = session.run(
                "MATCH (m:Morphotype) WHERE m.insitu_embedding IS NOT NULL "
                "RETURN count(m) AS c"
            ).single()["c"]
            print(f"[Neo4j导入] 导入前有 insitu_embedding 的节点: {before}")

            # 批量更新
            items = list(embeddings_data.items())
            batch_size = 50

            for batch_start in range(0, len(items), batch_size):
                batch = items[batch_start:batch_start + batch_size]
                for mid, data in batch:
                    try:
                        # MERGE: 如果 FN_ 节点已存在则更新 insitu_embedding, 否则创建
                        result = session.run("""
                            MERGE (m:Morphotype {morphotype_id: $mid})
                            ON CREATE SET
                                m.name = $name,
                                m.phylum_name = $phylum,
                                m.source = 'fathomnet_insitu',
                                m.created_at = timestamp()
                            SET m.insitu_embedding = $emb,
                                m.insitu_n_crops = $n_crops,
                                m.insitu_n_train_samples = $n_train
                            RETURN m.morphotype_id AS id
                        """, mid=mid, name=data["species_name"],
                           phylum=data["phylum"],
                           emb=data["embedding"],
                           n_crops=data["n_crops"],
                           n_train=data["n_train_samples"])
                        if result.single():
                            updated += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        if failed <= 3:
                            print(f"  [错误] {mid}: {e}")

                if (batch_start // batch_size) % 5 == 0:
                    print(f"  [进度] {batch_start + len(batch)}/{len(items)}")

        # 验证导入后状态
        with driver.session() as session:
            after = session.run(
                "MATCH (m:Morphotype) WHERE m.insitu_embedding IS NOT NULL "
                "RETURN count(m) AS c"
            ).single()["c"]

            # 按来源统计
            by_source = session.run(
                "MATCH (m:Morphotype) WHERE m.insitu_embedding IS NOT NULL "
                "RETURN m.source AS source, count(m) AS c ORDER BY c DESC"
            )
            source_stats = {r["source"]: r["c"] for r in by_source}

    finally:
        driver.close()

    print(f"\n[Neo4j导入] 结果:")
    print(f"  处理: {updated + failed}, 成功: {updated}, 失败: {failed}")
    print(f"  导入前 insitu_embedding 节点: {before}")
    print(f"  导入后 insitu_embedding 节点: {after}")
    print(f"  按来源: {source_stats}")


def create_insitu_vector_index():
    """创建 in-situ 向量索引 (如果不存在)。"""
    from neo4j import GraphDatabase

    print(f"\n[向量索引] 创建 insitu 向量索引...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # 检查索引是否已存在
            existing = session.run("SHOW INDEXES YIELD name WHERE name = 'morphotype_insitu_embeddings' RETURN name")
            if existing.single():
                print("[向量索引] morphotype_insitu_embeddings 已存在, 跳过创建")
                return

            # 创建向量索引
            session.run("""
                CREATE VECTOR INDEX morphotype_insitu_embeddings IF NOT EXISTS
                FOR (m:Morphotype) ON (m.insitu_embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """, dim=EMBEDDING_DIM)
            print(f"[向量索引] 创建成功: morphotype_insitu_embeddings (dim={EMBEDDING_DIM}, cosine)")

            # 验证
            import time
            time.sleep(2)
            result = session.run("SHOW INDEXES YIELD name, type, state WHERE name = 'morphotype_insitu_embeddings' RETURN name, type, state")
            for r in result:
                print(f"  索引: {r['name']}, 类型: {r['type']}, 状态: {r['state']}")

    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="提取 FathomNet in-situ 嵌入 (bbox 裁剪)")
    parser.add_argument("--fathomnet-dir", default=str(FATHOMNET_DIR),
                        help="FathomNet 数据目录")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="类别最小样本数阈值 (默认50)")
    parser.add_argument("--max-crops", type=int, default=10,
                        help="每个类别最多裁剪的 bbox 数 (默认10)")
    parser.add_argument("--min-crop-size", type=int, default=32,
                        help="bbox 最小边长 (像素, 默认32)")
    parser.add_argument("--padding", type=float, default=0.1,
                        help="bbox 裁剪时向外扩展比例 (默认0.1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只提取嵌入不导入 Neo4j")
    args = parser.parse_args()

    print("=" * 60)
    print("FathomNet In-situ 嵌入提取 — bbox 裁剪 + BioCLIP")
    print("=" * 60)

    # 加载 BioCLIP 模型
    model, processor = get_bioclip_model()

    # 提取 in-situ 嵌入
    embeddings = extract_insitu_embeddings(
        Path(args.fathomnet_dir),
        model, processor,
        min_samples=args.min_samples,
        max_crops_per_category=args.max_crops,
        min_crop_size=args.min_crop_size,
        padding_ratio=args.padding,
    )

    # 保存嵌入到文件
    emb_file = Path("models/insitu_embeddings.pt")
    emb_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v["embedding"] for k, v in embeddings.items()}, emb_file)
    print(f"\n[保存] in-situ 嵌入 ({len(embeddings)} 个): {emb_file}")

    # 保存元数据
    meta_file = Path("models/insitu_embeddings_meta.json")
    with open(meta_file, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "embedding"}
                   for k, v in embeddings.items()}, f, indent=2)
    print(f"[保存] 元数据: {meta_file}")

    # 导入 Neo4j
    if not args.dry_run and embeddings:
        import_insitu_to_neo4j(embeddings)
        create_insitu_vector_index()

    # 统计
    if embeddings:
        phylum_counts = defaultdict(int)
        for v in embeddings.values():
            phylum_counts[v["phylum"]] += 1
        print(f"\n[统计] 门级分布:")
        for p, c in sorted(phylum_counts.items(), key=lambda x: -x[1]):
            print(f"  {p}: {c}")

    print(f"\n{'='*60}")
    print(f"FathomNet In-situ 嵌入提取完成!")
    print(f"  总类别数: {len(embeddings)}")
    print(f"  嵌入文件: {emb_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
