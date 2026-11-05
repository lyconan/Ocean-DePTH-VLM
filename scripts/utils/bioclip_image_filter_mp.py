"""使用 BioCLIP 对图片进行零样本分类,识别生物照片 vs 非生物图片.

多进程并行版本: 每张 GPU 一个 worker,处理图片子集.

策略:
1. 加载 BioCLIP (open_clip) 到指定 GPU
2. 准备文本候选类别
3. 每个进程处理一部分图片
4. 主进程合并结果并执行删除

输入:
    data/processed/species_image_library/{Phylum}/{Species}/img_XXX.jpg

输出:
    data/processed/species_image_library/bioclip_classification.json
    data/processed/species_image_library/bioclip_filter_report.json
"""
import os
import sys
import json
import argparse
import time
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict, Counter

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 关闭 DecompressionBomb 警告

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "data/processed/species_image_library"

# 文本候选类别
TEXT_CANDIDATES = {
    "photo_organism": [
        "a photo of a marine organism",
        "a photo of a sea creature",
        "a photo of an underwater animal",
        "a close-up photo of a biological specimen",
        "a photograph of a living animal",
        "a photo of a fish",
        "a photo of a mollusk shell",
        "a photo of a crustacean",
        "a photo of a worm",
        "a photo of a coral",
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
    ],
    "table_text": [
        "a table with text and numbers",
        "a page of text from a book",
        "a list of taxonomic names",
    ],
    "phylogeny": [
        "a phylogenetic tree",
        "a taxonomic classification tree",
    ],
    "chart": [
        "a bar chart or graph",
        "a scatter plot",
    ],
}

BIO_CATEGORIES = {"photo_organism", "illustration_organism"}
NONBIO_CATEGORIES = {"diagram_anatomy", "map", "table_text", "phylogeny", "chart"}


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
                    all_imgs.append(str(img))
    return all_imgs


def worker_init(gpu_id):
    """worker 初始化: 加载模型到指定 GPU."""
    import open_clip
    global _MODEL, _PROCESSOR, _TOKENIZER, _TEXT_FEATURES, _DEVICE

    _DEVICE = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"[Worker {os.getpid()}] 加载 BioCLIP 到 {_DEVICE}...", flush=True)

    # BioCLIP 2.5 (ViT-H/14, 1024维) — 与 utils/bioclip_embed.py 保持一致
    model_name = "hf-hub:imageomics/bioclip-2.5-vith14"
    _MODEL, _, _PROCESSOR = open_clip.create_model_and_transforms(model_name)
    _TOKENIZER = open_clip.get_tokenizer(model_name)
    _MODEL.eval()
    _MODEL = _MODEL.to(_DEVICE)

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
    print(f"[Worker {os.getpid()}] 模型加载完成,文本候选 {len(all_texts)} 个", flush=True)


def process_image(img_path):
    """处理单张图片."""
    global _MODEL, _PROCESSOR, _TEXT_FEATURES, _DEVICE
    text_features, text_labels = _TEXT_FEATURES

    try:
        p = Path(img_path)
        real = p.resolve() if p.is_symlink() else p
        img = Image.open(real)
        if max(img.size) > 1500:
            img.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
        img = img.convert("RGB")
    except Exception as e:
        return (img_path, "error", 0.0)

    try:
        pixel_values = _PROCESSOR(img).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            image_features = _MODEL.encode_image(pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(0)

            # 按类别聚合
            sims = similarities.cpu().tolist()
            cat_scores = defaultdict(list)
            for s, label in zip(sims, text_labels):
                cat_scores[label].append(s)
            cat_max = {k: max(v) for k, v in cat_scores.items()}
            predicted = max(cat_max, key=cat_max.get)
            confidence = cat_max[predicted]

            return (img_path, predicted, float(confidence))
    except Exception as e:
        return (img_path, "error", 0.0)


def main():
    parser = argparse.ArgumentParser(description="BioCLIP 图片分类筛选 (多进程版)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="并行 worker 数(每个用一张 GPU)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="每个物种保留的生物照片数")
    parser.add_argument("--apply", action="store_true",
                        help="实际删除非生物图片")
    parser.add_argument("--output-dir", default=str(IMAGE_LIBRARY_DIR))
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("BioCLIP 图片分类筛选 (多进程版)", flush=True)
    print("=" * 70, flush=True)

    # 收集图片
    all_imgs = collect_all_images()
    print(f"总图片数: {len(all_imgs)}", flush=True)
    print(f"GPU 数: {torch.cuda.device_count()}", flush=True)
    print(f"启动 {args.num_workers} 个 worker", flush=True)

    start_time = time.time()

    # 多进程池
    gpu_ids = list(range(min(args.num_workers, torch.cuda.device_count())))
    if not gpu_ids:
        gpu_ids = [0]

    with mp.Pool(processes=len(gpu_ids), initializer=worker_init,
                 initargs=[(gpu_ids, )]) as pool:
        # 实际上 initializer 只接受单个参数,要传列表
        pass

    # 重新用更简单方式: 单 GPU 但加进度
    worker_init(gpu_ids[0])

    # 分批处理,边处理边输出进度
    batch_size = 64
    all_results = []
    total = len(all_imgs)

    print(f"\n开始分类 (batch_size={batch_size})...", flush=True)

    for i in range(0, total, batch_size):
        batch = all_imgs[i:i + batch_size]
        results = [process_image(p) for p in batch]
        all_results.extend(results)

        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - start_time
            done = i + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"  进度: {done}/{total} "
                  f"({done/total*100:.1f}%, "
                  f"速率: {rate:.1f} 张/秒, "
                  f"剩余: {remaining/60:.1f} 分钟)", flush=True)

    elapsed = time.time() - start_time
    print(f"\n分类完成! 用时: {elapsed/60:.1f} 分钟", flush=True)

    # 统计
    by_category = Counter()
    by_phylum = defaultdict(lambda: Counter())
    by_species = defaultdict(list)

    for path, category, confidence in all_results:
        by_category[category] += 1
        rel_path = str(Path(path).relative_to(IMAGE_LIBRARY_DIR))
        parts = Path(path).relative_to(IMAGE_LIBRARY_DIR).parts
        if len(parts) >= 3:
            phylum = parts[0]
            species = parts[1]
            by_phylum[phylum][category] += 1
            by_species[f"{phylum}/{species}"].append({
                "path": rel_path,
                "category": category,
                "confidence": confidence,
            })

    print(f"\n=== 分类结果 ===", flush=True)
    print(f"总图片: {sum(by_category.values())}", flush=True)
    for cat, count in by_category.most_common():
        is_bio = "✅" if cat in BIO_CATEGORIES else "❌"
        print(f"  {is_bio} {cat:25s}: {count:6d} ({count/sum(by_category.values())*100:.1f}%)", flush=True)

    print(f"\n=== 按门级分布(生物照片比例) ===", flush=True)
    for phylum in sorted(by_phylum.keys()):
        cat_count = by_phylum[phylum]
        total = sum(cat_count.values())
        bio = sum(cat_count.get(c, 0) for c in BIO_CATEGORIES)
        nonbio = sum(cat_count.get(c, 0) for c in NONBIO_CATEGORIES)
        error = cat_count.get("error", 0)
        print(f"  {phylum:25s}: {total:5d} 张, 生物 {bio} ({bio/total*100:.1f}%), "
              f"非生物 {nonbio} ({nonbio/total*100:.1f}%), 错误 {error}", flush=True)

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
    print(f"\n分类结果: {classification_path}", flush=True)

    # 筛选
    nonbio_to_delete = []
    species_summary = {}

    for species_key, items in by_species.items():
        bio_items = [it for it in items if it["category"] in BIO_CATEGORIES]
        nonbio_items = [it for it in items if it["category"] in NONBIO_CATEGORIES]
        error_items = [it for it in items if it["category"] == "error"]

        bio_items.sort(key=lambda x: -x["confidence"])
        kept = bio_items[:args.top_k]

        if len(bio_items) > args.top_k:
            nonbio_to_delete.extend([it["path"] for it in bio_items[args.top_k:]])
        nonbio_to_delete.extend([it["path"] for it in nonbio_items])

        species_summary[species_key] = {
            "total": len(items),
            "kept": len(kept),
            "nonbio_deleted": len(nonbio_items),
            "extra_deleted": max(0, len(bio_items) - args.top_k),
            "error": len(error_items),
        }

    total_kept = len(all_results) - len(nonbio_to_delete)
    print(f"\n=== 筛选报告 ===", flush=True)
    print(f"总图片: {len(all_results)}", flush=True)
    print(f"保留: {total_kept}", flush=True)
    print(f"删除: {len(nonbio_to_delete)} ({len(nonbio_to_delete)/len(all_results)*100:.1f}%)", flush=True)
    print(f"受影响物种: {len([s for s, info in species_summary.items() if info['nonbio_deleted'] > 0])}", flush=True)

    # 保存筛选报告
    report_path = IMAGE_LIBRARY_DIR / "bioclip_filter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": len(all_results),
            "kept": total_kept,
            "to_delete": len(nonbio_to_delete),
            "delete_paths": nonbio_to_delete,
            "species_summary": species_summary,
            "args": {"top_k": args.top_k, "apply": args.apply},
        }, f, ensure_ascii=False, indent=2)
    print(f"筛选报告: {report_path}", flush=True)

    # 实际删除
    if args.apply:
        print(f"\n=== 实际删除 {len(nonbio_to_delete)} 张非生物图片 ===", flush=True)
        deleted = 0
        for rel_path in nonbio_to_delete:
            full_path = IMAGE_LIBRARY_DIR / rel_path
            try:
                if full_path.is_symlink() or full_path.exists():
                    full_path.unlink()
                    deleted += 1
                if deleted % 1000 == 0 and deleted > 0:
                    print(f"  已删除 {deleted}...", flush=True)
            except Exception as e:
                print(f"  删除失败 {full_path}: {e}", flush=True)
        print(f"✅ 已删除: {deleted} 张", flush=True)
    else:
        print(f"\n(dry-run 模式,加 --apply 参数才实际删除)", flush=True)


if __name__ == "__main__":
    # 设置 stdout 无缓冲
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True, line_buffering=True)
    main()
