"""FathomNet数据下载 + 筛选CCZ相关底栖生物概念 (补充数据增强)"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# CCZ相关底栖生物概念列表
CCZ_CONCEPTS = [
    # Porifera
    "Hexactinellida", "Demospongiae", "Porifera",
    # Echinodermata
    "Ophiuroidea", "Asteroidea", "Holothuroidea", "Echinoidea", "Crinoidea",
    # Cnidaria
    "Ceriantharia", "Actiniaria", "Hydrozoa", "Alcyonacea", "Antipatharia",
    # Annelida
    "Serpulidae", "Sabellidae", "Polychaeta",
    # Arthropoda
    "Caridea", "Anomura", "Brachyura", "Amphipoda",
    # Xenophyophorea
    "Xenophyophorea",
    # Mollusca
    "Gastropoda", "Bivalvia", "Octopoda",
]


def download_fathomnet_data(
    output_dir: str,
    concepts: list = None,
    limit_per_concept: int = 1000,
    use_api: bool = True,
):
    """下载FathomNet数据。

    注意: 用户可能已经下载过数据，此脚本检查现有数据并跳过。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if concepts is None:
        concepts = CCZ_CONCEPTS

    if use_api:
        try:
            from fathomnet.api import images as fathomnet_api
            from fathomnet.models import GeoImageConstraints
            _download_via_api(output_dir, concepts, limit_per_concept)
        except ImportError:
            print("fathomnet包未安装，尝试使用fathomnet-generate CLI...")
            _download_via_cli(output_dir, concepts)
    else:
        _download_via_cli(output_dir, concepts)


def _download_via_api(output_dir, concepts, limit_per_concept):
    """通过FathomNet Python API下载数据。"""
    from fathomnet.api import images as fathomnet_api
    from fathomnet.models import GeoImageConstraints
    import requests

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    all_annotations = []
    categories = []
    cat_id_map = {}
    next_cat_id = 1

    for concept in tqdm(concepts, desc="下载FathomNet概念"):
        if concept not in cat_id_map:
            cat_id_map[concept] = next_cat_id
            categories.append({"id": next_cat_id, "name": concept})
            next_cat_id += 1

        try:
            constraints = GeoImageConstraints(concept=concept, limit=limit_per_concept)
            concept_images = fathomnet_api.find(constraints)
        except Exception as e:
            print(f"  查询失败 [{concept}]: {e}")
            continue

        for img_record in tqdm(concept_images, desc=f"  {concept}", leave=False):
            img_path = images_dir / f"{img_record.uuid}.png"
            if img_path.exists():
                continue

            try:
                resp = requests.get(img_record.url, timeout=30)
                with open(img_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                continue

            import hashlib
            img_id = int(hashlib.md5(str(img_record.uuid).encode()).hexdigest()[:8], 16)
            all_annotations.append({
                "image_id": img_id,
                "file_name": img_path.name,
                "width": getattr(img_record, 'width', None) or 0,
                "height": getattr(img_record, 'height', None) or 0,
                "bboxes": [
                    {
                        "category_id": cat_id_map[concept],
                        "bbox": [b.x, b.y, b.width, b.height],
                        "concept": b.concept,
                    }
                    for b in img_record.boundingBoxes
                    if b.concept == concept
                ]
            })

    # 保存COCO格式JSON
    coco = _build_coco_json(all_annotations, categories)
    coco_path = output_dir / "fathomnet_coco.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"下载完成: {len(coco['images'])} 图像, {len(coco['annotations'])} 标注")
    print(f"输出: {coco_path}")


def _download_via_cli(output_dir, concepts):
    """通过fathomnet-generate CLI下载数据。"""
    import subprocess

    concepts_file = output_dir / "ccx_concepts.txt"
    with open(concepts_file, "w") as f:
        f.write("\n".join(concepts))

    cmd = [
        "fathomnet-generate",
        "--output", str(output_dir / "coco"),
        "--concepts-file", str(concepts_file),
        "--taxa", "fathomnet",
        "--format", "coco"
    ]
    subprocess.run(cmd, check=True)


def _build_coco_json(annotations, categories):
    """构建COCO格式JSON。"""
    images = []
    anns = []
    ann_id = 1

    for img_ann in annotations:
        images.append({
            "id": img_ann["image_id"],
            "file_name": img_ann["file_name"],
            "width": img_ann["width"],
            "height": img_ann["height"],
        })
        for bbox in img_ann["bboxes"]:
            anns.append({
                "id": ann_id,
                "image_id": img_ann["image_id"],
                "category_id": bbox["category_id"],
                "bbox": bbox["bbox"],
                "area": bbox["bbox"][2] * bbox["bbox"][3],
                "iscrowd": 0,
            })
            ann_id += 1

    return {"images": images, "annotations": anns, "categories": categories}


def generate_test_uuid_blocklist(
    coco_json_path: str,
    output_path: str = "data/test/fathomnet_test_uuids.txt",
    test_size: float = 0.20,
    seed: int = 42,
):
    """生成测试集UUID阻止列表 — 防止数据泄漏。"""
    with open(coco_json_path) as f:
        coco = json.load(f)

    import random
    random.seed(seed)
    img_ids = [img["id"] for img in coco["images"]]
    random.shuffle(img_ids)
    n_test = int(len(img_ids) * test_size)
    test_uuids = img_ids[:n_test]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for uuid in test_uuids:
            f.write(f"{uuid}\n")

    print(f"生成阻止列表: {output_path} ({len(test_uuids)} UUIDs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载FathomNet数据")
    parser.add_argument("--output", default="data/raw/fathomnet_raw", help="输出目录")
    parser.add_argument("--limit", type=int, default=1000, help="每个概念最大图像数")
    parser.add_argument("--no-api", action="store_true", help="不使用API，使用CLI")
    args = parser.parse_args()

    download_fathomnet_data(
        output_dir=args.output,
        limit_per_concept=args.limit,
        use_api=not args.no_api,
    )

    # 下载数据后自动生成测试集UUID阻止列表
    coco_path = os.path.join(args.output, "fathomnet_coco.json")
    if os.path.exists(coco_path):
        generate_test_uuid_blocklist(coco_path)
