"""使用 BioCLIP 对图片进行零样本分类,识别生物照片 vs 非生物图片.

策略:
1. 加载 BioCLIP (open_clip)
2. 准备文本候选类别:
   - 生物照片 (整体标本/活体/局部特写)
   - 科学示意图 (解剖图/线条图)
   - 地图/分布图
   - 表格/文字页
   - 系统发育树
3. 对每张图片:
   - 提取图像嵌入
   - 计算与各类文本的相似度
   - 选择最相似的类别
4. 删除非生物图片
5. 每个物种保留 Top-K 高置信度的生物照片

输入:
    data/processed/species_image_library/{Phylum}/{Species}/img_XXX.jpg

输出:
    data/processed/species_image_library/bioclip_classification.json
    data/processed/species_image_library/bioclip_filter_report.json
    (默认只标记,不删除;加 --apply 后才删除)
"""
import os
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter

import torch
from PIL import Image

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data/processed/species_image_library"

# 文本候选类别 (BioCLIP 对自然语言描述敏感)
TEXT_CANDIDATES = {
    "photo_organism": [
        "a photo of a marine organism",
        "a photo of a sea creature",
        "a photo of an underwater animal",
        "a photo of a marine plant",
        "a close-up photo of a biological specimen",
        "a photograph of a living animal",
        "a photo of a fish",
        "a photo of a mollusk shell",
        "a photo of a crustacean",
        "a photo of a worm",
    ],
    "illustration_organism": [
        "a scientific illustration of an organism",
        "a drawing of a marine animal",
        "a colored illustration of a sea creature",
        "a hand-drawn picture of a biological specimen",
    ],
    "diagram_anatomy": [
        "an anatomical diagram of an organism",
        "a labeled diagram showing body parts",
        "a scientific drawing showing internal anatomy",
        "a line drawing of body structure",
    ],
    "map": [
        "a geographical map showing distribution",
        "a world map with marked locations",
        "a map showing where species are found",
    ],
    "table_text": [
        "a table with text and numbers",
        "a page of text from a book",
        "a list of taxonomic names",
        "a table of measurements",
    ],
    "phylogeny": [
        "a phylogenetic tree",
        "a taxonomic classification tree",
        "a cladogram showing evolutionary relationships",
    ],
    "chart": [
        "a bar chart or graph",
        "a scatter plot",
        "a data visualization chart",
    ],
}

# 类别映射: 哪些是生物相关(保留),哪些是非生物(删除)
BIO_CATEGORIES = {"photo_organism", "illustration_organism"}
NONBIO_CATEGORIES = {"diagram_anatomy", "map", "table_text", "phylogeny", "chart"}

# 全局模型缓存
_MODEL = None
_PROCESSOR = None
_TOKENIZER = None
_DEVICE = None
_TEXT_FEATURES = None


# BioCLIP 2.5 (ViT-H/14, 1024维) — 与 utils/bioclip_embed.py 保持一致
MODEL_HUB_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"


def load_model():
    """加载 BioCLIP 模型和文本特征."""
    global _MODEL, _PROCESSOR, _TOKENIZER, _DEVICE, _TEXT_FEATURES
    if _MODEL is not None:
        return

    import open_clip

    print("加载 BioCLIP 模型...")
    _MODEL, _, _PROCESSOR = open_clip.create_model_and_transforms(
        MODEL_HUB_NAME
    )
    _TOKENIZER = open_clip.get_tokenizer(MODEL_HUB_NAME)
    _MODEL.eval()

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL = _MODEL.to(_DEVICE)
    print(f"使用设备: {_DEVICE}")

    # 预计算文本特征
    all_texts = []
    text_labels = []
    for category, texts in TEXT_CANDIDATES.items():
        for t in texts:
            all_texts.append(t)
            text_labels.append(category)

    with torch.no_grad():
        text_tokens = _TOKENIZER(all_texts).to(_DEVICE)
        text_features = _MODEL.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    _TEXT_FEATURES = (text_features, text_labels)
    print(f"加载 {len(all_texts)} 个文本候选,覆盖 {len(TEXT_CANDIDATES)} 个类别")


def classify_image_batch(image_paths, batch_size=32):
    """批量分类图片.

    Returns:
        results: [(path, predicted_category, confidence), ...]
    """
    load_model()
    text_features, text_labels = _TEXT_FEATURES

    results = []
    valid_paths = []
    valid_images = []

    # 加载图片
    for p in image_paths:
        try:
            # 解析符号链接
            real_path = p.resolve() if p.is_symlink() else p
            # 限制图片大小避免 OOM
            img = Image.open(real_path)
            # 大图缩放 - 更激进 1024 加速
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            img = img.convert("RGB")
            valid_images.append(img)
            valid_paths.append(p)
        except Exception as e:
            results.append((str(p), "error", 0.0))

    if not valid_images:
        return results

    # 批量处理
    for i in range(0, len(valid_images), batch_size):
        batch_imgs = valid_images[i:i + batch_size]
        batch_paths = valid_paths[i:i + batch_size]

        try:
            # 预处理
            pixel_values = torch.stack([_PROCESSOR(img) for img in batch_imgs]).to(_DEVICE)

            with torch.inference_mode():
                image_features = _MODEL.encode_image(pixel_values)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # 计算相似度
                similarities = (image_features @ text_features.T)  # (B, num_texts)
                sims_list = similarities.cpu().tolist()

                for j, sims in enumerate(sims_list):
                    cat_scores = defaultdict(list)
                    for s, label in zip(sims, text_labels):
                        cat_scores[label].append(s)
                    cat_max = {k: max(v) for k, v in cat_scores.items()}
                    predicted = max(cat_max, key=cat_max.get)
                    confidence = cat_max[predicted]
                    results.append((str(batch_paths[j]), predicted, float(confidence)))
        except Exception as e:
            for p in batch_paths:
                results.append((str(p), "error", 0.0))
            print(f"  批次错误: {e}", flush=True)

    return results


def collect_all_images():
    """收集所有图片路径."""
    all_imgs = []
    for phylum_dir in IMAGE_LIBRARY_DIR.iterdir():
        if not phylum_dir.is_dir() or phylum_dir.name == "中国动物志":
            continue
        for species_dir in phylum_dir.iterdir():
            if not species_dir.is_dir():
                continue
            for img in species_dir.iterdir():
                if img.is_symlink() or img.is_file():
                    all_imgs.append(img)
    return all_imgs


def main():
    parser = argparse.ArgumentParser(description="BioCLIP 图片分类筛选")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--apply", action="store_true",
                        help="实际删除非生物图片(默认只生成报告)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="每个物种保留的生物照片数(按置信度排序)")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理图片数(0=全部,用于测试)")
    args = parser.parse_args()

    print("=" * 70)
    print("BioCLIP 图片分类筛选")
    print("=" * 70)

    # 收集图片
    all_imgs = collect_all_images()
    print(f"总图片数: {len(all_imgs)}")

    if args.limit > 0:
        all_imgs = all_imgs[:args.limit]
        print(f"限制处理: {len(all_imgs)} 张(测试模式)")

    # 批量分类
    print(f"\n开始分类 (batch_size={args.batch_size})...")
    start_time = time.time()

    all_results = []
    for i in range(0, len(all_imgs), args.batch_size):
        batch = all_imgs[i:i + args.batch_size]
        results = classify_image_batch(batch, args.batch_size)
        all_results.extend(results)

        if (i // args.batch_size) % 10 == 0:
            elapsed = time.time() - start_time
            done = i + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(all_imgs) - done) / rate if rate > 0 else 0
            print(f"  进度: {done}/{len(all_imgs)} "
                  f"({done/len(all_imgs)*100:.1f}%, "
                  f"速率: {rate:.1f} 张/秒, "
                  f"剩余: {remaining/60:.1f} 分钟)")

    elapsed = time.time() - start_time
    print(f"\n分类完成! 用时: {elapsed/60:.1f} 分钟, 速率: {len(all_imgs)/elapsed:.1f} 张/秒")

    # 统计
    by_category = Counter()
    by_phylum = defaultdict(lambda: Counter())
    by_species = defaultdict(lambda: [])  # species -> [(path, category, confidence)]

    for path, category, confidence in all_results:
        by_category[category] += 1
        # 解析门级和物种
        rel_path = Path(path).relative_to(IMAGE_LIBRARY_DIR)
        parts = rel_path.parts
        if len(parts) >= 3:
            phylum = parts[0]
            species = parts[1]
            by_phylum[phylum][category] += 1
            by_species[f"{phylum}/{species}"].append({
                "path": str(rel_path),
                "category": category,
                "confidence": confidence,
            })

    print(f"\n=== 分类结果 ===")
    print(f"总图片: {sum(by_category.values())}")
    for cat, count in by_category.most_common():
        is_bio = "✅" if cat in BIO_CATEGORIES else "❌"
        print(f"  {is_bio} {cat:25s}: {count:6d} ({count/sum(by_category.values())*100:.1f}%)")

    print(f"\n=== 按门级分布(生物照片比例) ===")
    for phylum in sorted(by_phylum.keys()):
        cat_count = by_phylum[phylum]
        total = sum(cat_count.values())
        bio = sum(cat_count.get(c, 0) for c in BIO_CATEGORIES)
        nonbio = sum(cat_count.get(c, 0) for c in NONBIO_CATEGORIES)
        error = cat_count.get("error", 0)
        print(f"  {phylum:25s}: {total:5d} 张, 生物 {bio} ({bio/total*100:.1f}%), "
              f"非生物 {nonbio} ({nonbio/total*100:.1f}%), 错误 {error}")

    # 保存分类结果
    classification_path = IMAGE_LIBRARY_DIR / "bioclip_classification.json"
    with open(classification_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(all_results),
            "categories": dict(by_category),
            "by_phylum": {p: dict(c) for p, c in by_phylum.items()},
            "results": [{"path": p, "category": c, "confidence": conf}
                       for p, c, conf in all_results],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n分类结果: {classification_path}")

    # 筛选报告
    nonbio_to_delete = []
    species_top_k_kept = defaultdict(list)

    for species_key, items in by_species.items():
        # 按 BioCLIP 置信度对生物照片排序
        bio_items = [it for it in items if it["category"] in BIO_CATEGORIES]
        nonbio_items = [it for it in items if it["category"] in NONBIO_CATEGORIES]
        error_items = [it for it in items if it["category"] == "error"]

        bio_items.sort(key=lambda x: -x["confidence"])
        kept = bio_items[:args.top_k]

        # 超过 top_k 的生物照片也被删除
        if len(bio_items) > args.top_k:
            nonbio_to_delete.extend([it["path"] for it in bio_items[args.top_k:]])

        # 非生物图片删除
        nonbio_to_delete.extend([it["path"] for it in nonbio_items])

        # 错误图片保留(避免误删)
        species_top_k_kept[species_key] = {
            "total": len(items),
            "kept": len(kept),
            "nonbio_deleted": len(nonbio_items),
            "extra_deleted": max(0, len(bio_items) - args.top_k),
            "error": len(error_items),
        }

    print(f"\n=== 筛选报告 ===")
    print(f"总图片: {len(all_results)}")
    print(f"保留: {len(all_results) - len(nonbio_to_delete)}")
    print(f"删除: {len(nonbio_to_delete)} ({len(nonbio_to_delete)/len(all_results)*100:.1f}%)")
    print(f"受影响物种: {len([s for s, info in species_top_k_kept.items() if info['nonbio_deleted'] > 0])}")

    # 保存筛选报告
    report_path = IMAGE_LIBRARY_DIR / "bioclip_filter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": len(all_results),
            "kept": len(all_results) - len(nonbio_to_delete),
            "to_delete": len(nonbio_to_delete),
            "delete_paths": nonbio_to_delete,
            "species_summary": dict(species_top_k_kept),
            "args": {
                "top_k": args.top_k,
                "apply": args.apply,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"筛选报告: {report_path}")

    # 实际删除
    if args.apply:
        print(f"\n=== 实际删除 {len(nonbio_to_delete)} 张非生物图片 ===")
        deleted = 0
        for rel_path in nonbio_to_delete:
            full_path = IMAGE_LIBRARY_DIR / rel_path
            try:
                if full_path.is_symlink() or full_path.exists():
                    full_path.unlink()
                    deleted += 1
            except Exception as e:
                print(f"  删除失败 {full_path}: {e}")
        print(f"✅ 已删除: {deleted} 张")
    else:
        print(f"\n(dry-run 模式,加 --apply 参数才实际删除)")


if __name__ == "__main__":
    main()
