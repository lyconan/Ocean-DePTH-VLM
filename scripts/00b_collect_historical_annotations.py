"""收集COMRA/ISA DeepData历史标注数据并统一格式"""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def load_coco_annotations(path: str) -> List[Dict]:
    """加载COCO格式标注。"""
    with open(path) as f:
        coco = json.load(f)
    records = []
    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}
    
    for ann in coco["annotations"]:
        img = img_map.get(ann["image_id"])
        if img:
            records.append({
                "image_path": img.get("file_name", ""),
                "bbox": ann["bbox"],  # [x, y, w, h]
                "original_label": cat_map.get(ann["category_id"], "unknown"),
                "source": "coco",
                "image_width": img.get("width", 0),
                "image_height": img.get("height", 0),
            })
    return records


def load_voc_annotations(voc_dir: str) -> List[Dict]:
    """加载VOC格式标注。"""
    import xml.etree.ElementTree as ET
    records = []
    voc_dir = Path(voc_dir)
    
    for xml_file in voc_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        img_path = root.findtext("filename", "")
        size = root.find("size")
        w = int(size.findtext("width", "0")) if size is not None else 0
        h = int(size.findtext("height", "0")) if size is not None else 0
        
        for obj in root.findall("object"):
            name = obj.findtext("name", "unknown")
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))
                records.append({
                    "image_path": img_path,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "original_label": name,
                    "source": "voc",
                    "image_width": w,
                    "image_height": h,
                })
    return records


def load_yolo_annotations(yolo_dir: str, names_file: str) -> List[Dict]:
    """加载YOLO格式标注。"""
    records = []
    yolo_dir = Path(yolo_dir)
    
    with open(names_file) as f:
        names = [line.strip() for line in f if line.strip()]
    
    for label_file in yolo_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, nw, nh = [float(x) for x in parts[1:5]]
                    # 查找对应的图像文件（支持多种格式）
                    img_stem = label_file.stem
                    img_path = None
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                        candidate = label_file.parent / f"{img_stem}{ext}"
                        if candidate.exists():
                            img_path = str(candidate)
                            break
                    if img_path is None:
                        img_path = img_stem + ".png"  # 默认.png，后续会检查存在性
                    records.append({
                        "image_path": img_path,
                        "bbox_yolo": [cx, cy, nw, nh],
                        "class_id": class_id,
                        "original_label": names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}",
                        "source": "yolo",
                    })
    return records


def collect_historical_annotations(data_sources: List[Dict]) -> List[Dict]:
    """收集并统一所有历史标注源。"""
    unified = []
    for source in data_sources:
        fmt = source.get("format", "coco")
        path = source.get("path", "")
        name = source.get("name", "unknown")
        
        print(f"加载 [{name}] ({fmt}): {path}")
        
        if fmt == "coco":
            records = load_coco_annotations(path)
        elif fmt == "voc":
            records = load_voc_annotations(path)
        elif fmt == "yolo":
            records = load_yolo_annotations(path, source.get("names_file", ""))
        else:
            print(f"  跳过: 未知格式 '{fmt}'")
            continue
        
        for rec in records:
            rec["source_name"] = name
        unified.extend(records)
        print(f"  加载 {len(records)} 条标注")
    
    print(f"总计: {len(unified)} 条统一标注")
    return unified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集历史标注数据")
    parser.add_argument("--sources", type=str, default="[]",
                       help="JSON格式的数据源列表")
    parser.add_argument("--output", default="data/processed/historical_annotations.json")
    args = parser.parse_args()
    
    sources = json.loads(args.sources)
    records = collect_historical_annotations(sources)
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"保存: {args.output}")
