#!/usr/bin/env python3
"""扩充知识检索库 — 为物种图片库批量提取 BioCLIP 嵌入并导入 Neo4j (v1.0)

将检索库从 185 个 APSMA 形态型扩充到 7,000+ 个物种:
1. 遍历 species_image_library/<phylum>/<species>/ 目录
2. 每个物种取代表性图像 (最多 N 张) 提取 BioCLIP 嵌入 (取平均)
3. 在 Neo4j 中创建 Morphotype 节点 (morphotype_id = WORMS_<phylum>_<species>)
4. 节点自动加入现有 morphotype_exsitu_embeddings 向量索引

同时支持 FathomNet 细粒度类别嵌入提取:
- 从 FathomNet COCO 标注中每个类别取代表图像
- 创建 Morphotype 节点 (morphotype_id = FN_<category_name>)

用法:
    # 物种图片库嵌入
    python scripts/07e_expand_knowledge_base.py --mode species

    # FathomNet 类别嵌入
    python scripts/07e_expand_knowledge_base.py --mode fathomnet

    # 两者都做
    python scripts/07e_expand_knowledge_base.py --mode all
"""
import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Dict

import torch
from PIL import Image

# 添加 scripts 目录到 path
sys.path.insert(0, str(Path(__file__).parent))
from utils.bioclip_embed import get_bioclip_model, extract_embedding as _extract_emb

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

SPECIES_IMAGE_LIB = Path("data/processed/species_image_library")
FATHOMNET_DIR = Path("/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_representative_images(species_dir: Path, max_images: int = 3) -> List[Path]:
    """从物种目录中选取代表性图像 (最多 max_images 张)。"""
    images = sorted([
        f for f in species_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS and f.is_file()
    ])
    # 均匀采样 (如果超过 max_images)
    if len(images) > max_images:
        step = len(images) / max_images
        images = [images[int(i * step)] for i in range(max_images)]
    return images


def extract_species_embeddings(
    image_lib: Path,
    model,
    processor,
    max_images_per_species: int = 3,
    max_species: Optional[int] = None,
) -> Dict[str, dict]:
    """遍历物种图片库, 提取每个物种的 BioCLIP 嵌入。

    返回: {species_key: {embedding, image_path, phylum, species_name}}
    """
    from utils.bioclip_embed import is_bioclip
    print(f"[物种嵌入] BioCLIP: {is_bioclip()}")
    print(f"[物种嵌入] 图片库: {image_lib}")

    results = {}
    processed = 0
    failed = 0

    # 遍历门目录
    phylum_dirs = sorted([d for d in image_lib.iterdir() if d.is_dir()])
    print(f"[物种嵌入] 发现 {len(phylum_dirs)} 个门目录")

    for phylum_dir in phylum_dirs:
        phylum = phylum_dir.name
        # 跳过非物种目录 (如 中国动物志 有不同结构)
        species_dirs = sorted([d for d in phylum_dir.iterdir() if d.is_dir()])
        for species_dir in species_dirs:
            species_name = species_dir.name
            images = get_representative_images(species_dir, max_images_per_species)
            if not images:
                continue

            # 提取多张图像嵌入并取平均
            embeddings = []
            valid_image_path = None
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    emb = _extract_emb(img, model, processor)
                    embeddings.append(emb)
                    if valid_image_path is None:
                        valid_image_path = str(img_path)
                except Exception as e:
                    if failed < 5:
                        print(f"  [跳过] {img_path}: {e}")
                    failed += 1
                    continue

            if not embeddings:
                continue

            # 平均嵌入并归一化
            avg_emb = torch.stack(embeddings).mean(dim=0)
            norm = avg_emb.norm()
            if norm > 0:
                avg_emb = avg_emb / norm

            species_key = f"WORMS_{phylum}_{species_name}"
            results[species_key] = {
                "embedding": avg_emb.tolist(),
                "image_path": valid_image_path,
                "phylum": phylum,
                "species_name": species_name,
                "n_images": len(embeddings),
            }
            processed += 1

            if processed % 200 == 0:
                print(f"  [进度] 已处理 {processed} 个物种 (失败 {failed})")

            if max_species and processed >= max_species:
                print(f"  [限制] 达到 max_species={max_species}, 停止")
                return results

    print(f"\n[物种嵌入] 完成: {processed} 个物种, {failed} 个失败")
    return results


def extract_fathomnet_embeddings(
    fathomnet_dir: Path,
    model,
    processor,
    min_samples: int = 50,
    max_images_per_category: int = 5,
) -> Dict[str, dict]:
    """从 FathomNet 中提取每个类别 (≥min_samples) 的代表性 BioCLIP 嵌入。"""
    from utils.bioclip_embed import is_bioclip
    print(f"\n[FathomNet嵌入] BioCLIP: {is_bioclip()}")
    print(f"[FathomNet嵌入] 数据目录: {fathomnet_dir}")

    # 加载 COCO 标注
    coco_path = fathomnet_dir / "Annotations" / "fathomnet_train.json"
    print(f"[FathomNet嵌入] 加载 COCO: {coco_path}")
    with open(coco_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}

    # 统计类别样本数 + 收集每类图像
    cat_counts = defaultdict(int)
    cat_images = defaultdict(list)
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        cat_counts[cat_id] += 1
        if len(cat_images[cat_id]) < max_images_per_category:
            cat_images[cat_id].append(ann["image_id"])

    # 筛选 ≥ min_samples 的类别
    selected_cats = [(cid, cnt) for cid, cnt in cat_counts.items() if cnt >= min_samples]
    selected_cats.sort(key=lambda x: -x[1])
    print(f"[FathomNet嵌入] 选中 {len(selected_cats)} 个类别 (≥{min_samples} 样本)")

    results = {}
    processed = 0
    failed = 0

    train_dir = fathomnet_dir / "Train"

    for cat_id, cnt in selected_cats:
        cat_name = cat_map[cat_id]
        image_ids = cat_images[cat_id]

        embeddings = []
        valid_image_path = None
        for img_id in image_ids:
            img_info = img_map.get(img_id)
            if not img_info:
                continue
            file_name = img_info.get("file_name", "")
            img_path = train_dir / file_name
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                emb = _extract_emb(img, model, processor)
                embeddings.append(emb)
                if valid_image_path is None:
                    valid_image_path = str(img_path)
            except Exception as e:
                failed += 1
                continue

        if not embeddings:
            failed += 1
            continue

        avg_emb = torch.stack(embeddings).mean(dim=0)
        norm = avg_emb.norm()
        if norm > 0:
            avg_emb = avg_emb / norm

        # 推断门级分类 (复用 taxonomy_mapper)
        try:
            from utils.taxonomy_mapper import infer_phylum_from_concept
            phylum = infer_phylum_from_concept(cat_name)
        except Exception:
            phylum = "Unknown"

        species_key = f"FN_{cat_name}"
        results[species_key] = {
            "embedding": avg_emb.tolist(),
            "image_path": valid_image_path,
            "phylum": phylum,
            "species_name": cat_name,
            "n_images": len(embeddings),
            "n_train_samples": cnt,
        }
        processed += 1

        if processed % 100 == 0:
            print(f"  [进度] 已处理 {processed}/{len(selected_cats)} 个类别 (失败 {failed})")

    print(f"\n[FathomNet嵌入] 完成: {processed} 个类别, {failed} 个失败")
    return results


def import_to_neo4j(embeddings_data: Dict[str, dict], source_tag: str = "species"):
    """将嵌入数据导入 Neo4j, 创建/更新 Morphotype 节点。"""
    from neo4j import GraphDatabase

    print(f"\n[Neo4j导入] 连接: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    created = 0
    updated = 0
    failed = 0

    try:
        with driver.session() as session:
            # 检查导入前状态
            before = session.run(
                "MATCH (m:Morphotype) WHERE m.exsitu_embedding IS NOT NULL "
                "RETURN count(m) AS c"
            ).single()["c"]
            print(f"[Neo4j导入] 导入前有嵌入的 Morphotype: {before}")

            batch_size = 50
            items = list(embeddings_data.items())
            for batch_start in range(0, len(items), batch_size):
                batch = items[batch_start:batch_start + batch_size]
                for mid, data in batch:
                    try:
                        result = session.run("""
                            MERGE (m:Morphotype {morphotype_id: $mid})
                            ON CREATE SET
                                m.name = $name,
                                m.phylum_name = $phylum,
                                m.exsitu_image_path = $img_path,
                                m.source = $source,
                                m.created_at = timestamp()
                            ON MATCH SET
                                m.exsitu_embedding = $emb,
                                m.exsitu_image_path = $img_path,
                                m.source = $source
                            SET m.exsitu_embedding = $emb
                            RETURN m.morphotype_id AS id
                        """, mid=mid, name=data["species_name"],
                           phylum=data["phylum"],
                           img_path=data["image_path"],
                           source=source_tag,
                           emb=data["embedding"])
                        if result.single():
                            created += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        if failed <= 3:
                            print(f"  [错误] {mid}: {e}")

                if (batch_start // batch_size) % 10 == 0:
                    print(f"  [进度] {batch_start + len(batch)}/{len(items)}")

        # 验证导入后状态
        with driver.session() as session:
            after = session.run(
                "MATCH (m:Morphotype) WHERE m.exsitu_embedding IS NOT NULL "
                "RETURN count(m) AS c"
            ).single()["c"]

            # 按来源统计
            by_source = session.run(
                "MATCH (m:Morphotype) WHERE m.exsitu_embedding IS NOT NULL "
                "RETURN m.source AS source, count(m) AS c ORDER BY c DESC"
            )
            source_stats = {r["source"]: r["c"] for r in by_source}

    finally:
        driver.close()

    print(f"\n[Neo4j导入] 结果:")
    print(f"  处理: {created + failed}, 成功: {created}, 失败: {failed}")
    print(f"  导入前嵌入节点: {before}")
    print(f"  导入后嵌入节点: {after}")
    print(f"  按来源: {source_stats}")


def main():
    parser = argparse.ArgumentParser(description="扩充知识检索库 — 批量 BioCLIP 嵌入导入 Neo4j")
    parser.add_argument("--mode", choices=["species", "fathomnet", "all"], default="all",
                        help="嵌入来源: species=物种图片库, fathomnet=FathomNet类别, all=两者")
    parser.add_argument("--max-images-per-species", type=int, default=3,
                        help="每个物种最多采样图像数 (默认3)")
    parser.add_argument("--max-images-per-category", type=int, default=5,
                        help="每个FathomNet类别最多采样图像数 (默认5)")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="FathomNet类别最小样本数阈值 (默认50)")
    parser.add_argument("--max-species", type=int, default=None,
                        help="最大物种数 (调试用)")
    parser.add_argument("--species-lib", default=str(SPECIES_IMAGE_LIB),
                        help="物种图片库路径")
    parser.add_argument("--fathomnet-dir", default=str(FATHOMNET_DIR),
                        help="FathomNet数据目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="只提取嵌入不导入Neo4j")
    args = parser.parse_args()

    # 加载 BioCLIP 模型
    print("=" * 60)
    print("扩充知识检索库 — BioCLIP 嵌入批量提取")
    print("=" * 60)
    model, processor = get_bioclip_model()

    all_embeddings = {}

    # 1. 物种图片库嵌入
    if args.mode in ("species", "all"):
        species_emb = extract_species_embeddings(
            Path(args.species_lib),
            model, processor,
            max_images_per_species=args.max_images_per_species,
            max_species=args.max_species,
        )
        all_embeddings.update(species_emb)
        # 保存中间结果
        emb_cache = Path("data/processed/species_embeddings_cache.json")
        with open(emb_cache, "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "embedding"}
                       for k, v in species_emb.items()}, f, indent=2)
        print(f"  物种嵌入元数据缓存: {emb_cache}")

    # 2. FathomNet 类别嵌入
    if args.mode in ("fathomnet", "all"):
        fn_emb = extract_fathomnet_embeddings(
            Path(args.fathomnet_dir),
            model, processor,
            min_samples=args.min_samples,
            max_images_per_category=args.max_images_per_category,
        )
        all_embeddings.update(fn_emb)

    # 3. 保存完整嵌入到文件
    emb_file = Path("models/knowledge_base_embeddings.pt")
    emb_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v["embedding"] for k, v in all_embeddings.items()}, emb_file)
    print(f"\n[保存] 完整嵌入 ({len(all_embeddings)} 个): {emb_file}")

    # 4. 导入 Neo4j
    if not args.dry_run and all_embeddings:
        import_to_neo4j(all_embeddings, source_tag=args.mode)

    print(f"\n{'='*60}")
    print(f"知识检索库扩充完成!")
    print(f"  总嵌入数: {len(all_embeddings)}")
    print(f"  嵌入文件: {emb_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
