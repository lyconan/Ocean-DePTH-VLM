"""In-situ视觉锚点嵌入提取 — 从FathomNet验证图像 (v3.1)

桥接域间隙: in-situ查询图像将比ex-situ图集照片更好地匹配in-situ锚点。"""
import hashlib
import os
import json
import argparse
from pathlib import Path
from typing import Dict

from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.schemas import UNKNOWN_SENTINEL

from utils.module_loader import load_script_module

_clip_module = None
def _get_clip_module():
    global _clip_module
    if _clip_module is None:
        _clip_module = load_script_module('07_extract_clip_embeddings')
    return _clip_module


def _compute_uuid_hash(uuid_str: str) -> str:
    """与01_download_fathomnet.py一致的确定性UUID哈希。"""
    return str(int(hashlib.md5(uuid_str.encode()).hexdigest()[:8], 16))


def extract_insitu_embeddings(
    alignment_csv: str,
    fathomnet_coco_json: str = None,
    output_path: str = "data/processed/fathomnet_insitu_crops/insitu_embeddings.pt",
    top_n: int = 5,
    test_uuid_blocklist_path: str = "data/test/fathomnet_test_uuids.txt",
):
    """为每个有WoRMS匹配的APSMA形态型提取in-situ视觉锚点。
    
    数据泄漏防护(v3.1.1): 测试集UUID被阻止进入KG。
    阻止列表存储的是UUID的MD5哈希整数(与01_download_fathomnet.py一致)。
    """
    extract_single_clip_embedding = _get_clip_module().extract_single_clip_embedding
    import pandas as pd
    import torch
    
    # 加载测试集UUID阻止列表(存储的是MD5哈希整数字符串)
    blocked_uuid_hashes = set()
    if test_uuid_blocklist_path and os.path.exists(test_uuid_blocklist_path):
        with open(test_uuid_blocklist_path) as f:
            blocked_uuid_hashes = set(line.strip() for line in f if line.strip())
        print(f"加载 {len(blocked_uuid_hashes)} 个阻止UUID哈希")
    
    # 加载对齐矩阵
    alignment = pd.read_csv(alignment_csv)
    matched = alignment[alignment["mapping_type"].isin(["EXACT", "PARTIAL"])]
    print(f"有WoRMS匹配的形态型: {len(matched)}")
    
    insitu_embeddings_map = {}
    leaked_count = 0
    
    try:
        from fathomnet.api import images as fathomnet_api
        from fathomnet.models import GeoImageConstraints
    except ImportError:
        print("fathomnet包未安装，无法提取in-situ嵌入")
        return {}
    
    for _, row in matched.iterrows():
        morphotype_id = row["apsma_morphotype_id"]
        matched_rank = row["matched_rank"]
        
        try:
            constraints = GeoImageConstraints(concept=matched_rank, limit=top_n * 3)
            concept_images = fathomnet_api.find(constraints)
        except Exception as e:
            print(f"  FathomNet查询失败 [{matched_rank}]: {e}")
            continue
        
        crop_paths = []
        crop_embeddings = []
        
        for img_record in concept_images:
            # 数据泄漏检查: 用与01一致的MD5哈希比较
            raw_uuid = getattr(img_record, 'uuid', None)
            if not raw_uuid:
                url = getattr(img_record, 'url', None) or ''
                raw_uuid = url.split('/')[-1].split('.')[0] if url else ''
            if raw_uuid:
                img_uuid_hash = _compute_uuid_hash(str(raw_uuid))
                if img_uuid_hash in blocked_uuid_hashes:
                    leaked_count += 1
                    continue
            
            if len(crop_embeddings) >= top_n:
                break
            
            try:
                import requests
                from io import BytesIO
                resp = requests.get(img_record.url, timeout=30)
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception:
                continue
            
            for bbox in img_record.boundingBoxes:
                if bbox.concept == matched_rank:
                    x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                    if x + w > img.width or y + h > img.height:
                        continue
                    crop = img.crop((x, y, x + w, y + h))
                    
                    # 嵌入提取与路径保存保持原子性
                    try:
                        emb = extract_single_clip_embedding(crop)
                    except Exception as e:
                        print(f"  嵌入提取失败 [{morphotype_id}]: {e}")
                        continue
                    
                    crop_dir = Path("data/processed/fathomnet_insitu_crops")
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = crop_dir / f"{morphotype_id}_{len(crop_paths)}.png"
                    crop.save(str(crop_path))
                    crop_paths.append(str(crop_path))
                    crop_embeddings.append(emb)
        
        if crop_embeddings:
            insitu_embeddings_map[morphotype_id] = {
                "paths": crop_paths,
                "embeddings": crop_embeddings
            }
    
    if leaked_count > 0:
        print(f"数据泄漏防护: 阻止 {leaked_count} 个测试集图像进入KG")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(insitu_embeddings_map, output_path)
        print(f"In-situ嵌入保存: {output_path} ({len(insitu_embeddings_map)} 形态型)")
    
    return insitu_embeddings_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取in-situ嵌入")
    parser.add_argument("--alignment-csv", required=True)
    parser.add_argument("--fathomnet-coco", default=None)
    parser.add_argument("--output", default="data/processed/fathomnet_insitu_crops/insitu_embeddings.pt")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--blocklist", default="data/test/fathomnet_test_uuids.txt")
    args = parser.parse_args()
    
    extract_insitu_embeddings(
        alignment_csv=args.alignment_csv,
        fathomnet_coco_json=args.fathomnet_coco,
        output_path=args.output,
        top_n=args.top_n,
        test_uuid_blocklist_path=args.blocklist,
    )
