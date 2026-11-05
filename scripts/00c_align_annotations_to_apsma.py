"""将历史标注对齐到APSMA形态型分类体系 — 生成YOLO训练集"""
import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from utils.taxonomy_mapper import APSMA_COARSE_CLASS_MAP, infer_phylum_from_concept


def lookup_alignment_matrix(label: str, alignment_csv: str) -> Optional[Dict]:
    """查找WoRMS↔APSMA对齐矩阵。"""
    try:
        with open(alignment_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("matched_rank", "").lower() == label.lower():
                    return dict(row)
                if row.get("apsma_morphotype_id", "").lower() == label.lower():
                    return dict(row)
    except FileNotFoundError:
        pass
    return None


def map_to_apsma_aligned_class(apsma_match: Dict) -> int:
    """从APSMA匹配结果映射到YOLO类别ID。
    
    未知门映射到专门的unknown类别ID(=9)，与utils/taxonomy_mapper.py一致。
    """
    phylum = apsma_match.get("apsma_phylum", "")
    return APSMA_COARSE_CLASS_MAP.get(phylum, len(APSMA_COARSE_CLASS_MAP))


def align_to_apsma(
    unified_records: List[Dict],
    apsma_knowledge: str = None,
    alignment_csv: str = None,
) -> List[Dict]:
    """将每条标注的标签映射到APSMA形态型分类体系。
    
    映射优先级:
    1. EXACT匹配: 原始标签直接映射到特定APSMA形态型
    2. PARTIAL匹配: 映射到APSMA class/order级别
    3. COARSEST匹配: 回退到APSMA门级
    """
    # 加载APSMA知识库
    apsma_morphotypes = {}
    if apsma_knowledge and os.path.exists(apsma_knowledge):
        with open(apsma_knowledge) as f:
            for line in f:
                rec = json.loads(line)
                apsma_morphotypes[rec["morphotype_id"]] = rec
    
    aligned_records = []
    mapping_stats = defaultdict(int)
    
    for rec in unified_records:
        label = rec["original_label"]
        
        # 尝试通过对齐矩阵查找
        apsma_match = None
        if alignment_csv:
            apsma_match = lookup_alignment_matrix(label, alignment_csv)
        
        if apsma_match and apsma_match.get("mapping_type") == "EXACT":
            rec["apsma_morphotype_id"] = apsma_match.get("apsma_morphotype_id")
            rec["apsma_phylum"] = apsma_match.get("apsma_phylum", "")
            rec["apsma_class"] = apsma_match.get("apsma_class", "")
            rec["apsma_order"] = apsma_match.get("apsma_order", "")
            rec["yolo_class"] = map_to_apsma_aligned_class(apsma_match)
            rec["mapping_type"] = "EXACT"
            mapping_stats["EXACT"] += 1
        elif apsma_match and apsma_match.get("mapping_type") == "PARTIAL":
            rec["apsma_morphotype_id"] = None
            rec["apsma_phylum"] = apsma_match.get("apsma_phylum", "")
            rec["apsma_class"] = apsma_match.get("apsma_class", "")
            rec["apsma_order"] = apsma_match.get("apsma_order", "")
            rec["yolo_class"] = map_to_apsma_aligned_class(apsma_match)
            rec["mapping_type"] = "PARTIAL"
            mapping_stats["PARTIAL"] += 1
        else:
            phylum = infer_phylum_from_concept(label)
            rec["apsma_morphotype_id"] = None
            rec["apsma_phylum"] = phylum
            rec["apsma_class"] = ""
            rec["apsma_order"] = ""
            rec["yolo_class"] = APSMA_COARSE_CLASS_MAP.get(
                phylum, len(APSMA_COARSE_CLASS_MAP)
            )
            rec["mapping_type"] = "COARSEST" if phylum != "unknown" else "NULL"
            mapping_stats[rec["mapping_type"]] += 1
        
        aligned_records.append(rec)
    
    print(f"对齐统计: {dict(mapping_stats)}")
    return aligned_records


def generate_yolo_training_set(
    aligned_records: List[Dict],
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """从APSMA对齐标注生成YOLO格式训练集。"""
    from sklearn.model_selection import train_test_split
    import shutil
    
    output_dir = Path(output_dir)
    
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # 分层划分
    labels = [r.get("yolo_class", 0) for r in aligned_records]
    train_recs, temp_recs = train_test_split(
        aligned_records, test_size=(1 - train_ratio),
        random_state=42, stratify=labels
    )
    test_size = (1 - train_ratio - val_ratio) / (1 - train_ratio)
    temp_labels = [r.get("yolo_class", 0) for r in temp_recs]
    val_recs, test_recs = train_test_split(
        temp_recs, test_size=test_size,
        random_state=42, stratify=temp_labels
    )
    
    splits = {"train": train_recs, "val": val_recs, "test": test_recs}
    
    for split_name, recs in splits.items():
        # 按图像分组收集标签，避免追加模式导致重复
        labels_by_image = defaultdict(list)
        image_paths = {}

        for rec in recs:
            img_stem = Path(rec["image_path"]).stem
            label_file = output_dir / "labels" / split_name / f"{img_stem}.txt"

            # 检查是COCO格式(bbox)还是YOLO格式(bbox_yolo)
            if "bbox" in rec:
                # COCO格式: [x, y, w, h] 像素坐标
                bbox = rec["bbox"]
                w, h = rec.get("image_width", 1), rec.get("image_height", 1)
                if w == 0:
                    w = 1
                if h == 0:
                    h = 1
                cx = (bbox[0] + bbox[2] / 2) / w
                cy = (bbox[1] + bbox[3] / 2) / h
                nw = bbox[2] / w
                nh = bbox[3] / h
            elif "bbox_yolo" in rec:
                # YOLO格式: [cx, cy, nw, nh] 已归一化
                bbox_yolo = rec["bbox_yolo"]
                cx, cy, nw, nh = bbox_yolo
            else:
                continue

            class_id = rec.get("yolo_class", 0)
            labels_by_image[label_file].append(
                f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
            )

            # 记录图像路径（同一图像只记录一次）
            src_img = rec.get("image_path", "")
            if src_img:
                image_paths[label_file] = src_img

        # 一次性写入标签文件（使用"w"模式，避免追加导致重复）
        for label_file, lines in labels_by_image.items():
            with open(label_file, "w") as f:
                f.writelines(lines)

            # 链接图像
            src_img = image_paths.get(label_file, "")
            if src_img and os.path.exists(src_img):
                dst = output_dir / "images" / split_name / Path(src_img).name
                if not os.path.lexists(dst):
                    os.symlink(os.path.abspath(src_img), dst)
    
    print(f"YOLO训练集生成完成: {output_dir}")
    for split_name, recs in splits.items():
        print(f"  {split_name}: {len(recs)} 图像")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APSMA分类对齐")
    parser.add_argument("--annotations", required=True, help="统一标注JSON")
    parser.add_argument("--alignment-csv", default=None, help="对齐矩阵CSV")
    parser.add_argument("--apsma-knowledge", default=None, help="APSMA知识JSONL")
    parser.add_argument("--output-dir", default="data/processed/apsma_aligned_yolo")
    args = parser.parse_args()
    
    with open(args.annotations) as f:
        records = json.load(f)
    
    aligned = align_to_apsma(records, args.apsma_knowledge, args.alignment_csv)
    generate_yolo_training_set(aligned, args.output_dir)
