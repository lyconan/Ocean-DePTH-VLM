"""BioCLIP嵌入提取 — ex-situ(图集标本) + in-situ(FathomNet验证图像) (v3.2)

v3.2: 统一切换到 BioCLIP 2.5 (hf-hub:imageomics/bioclip-2.5-vith14, 1024维),
模型加载与嵌入计算走 utils/bioclip_embed.py 共享封装;
移除 512维 openai CLIP 回退 (维度不一致会污染向量库)。
"""
import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.bioclip_embed import (
    get_bioclip_model,
    get_embedding_dim,
    extract_embedding,
)


def _get_clip_model():
    """懒加载 BioCLIP 2.5 (1024维), 统一共享封装 (无 CLIP 回退)。"""
    return get_bioclip_model()


def extract_single_clip_embedding(image) -> torch.Tensor:
    """提取单个BioCLIP嵌入(1024维)。

    Args:
        image: PIL.Image或图像路径
    Returns:
        (1024,) 归一化嵌入向量
    """
    model, processor = _get_clip_model()
    return extract_embedding(image, model, processor)


def extract_clip_embeddings(
    image_paths: List[str],
    batch_size: int = 32,
    output_path: str = None,
) -> torch.Tensor:
    """批量提取BioCLIP嵌入。

    Args:
        image_paths: 图像路径列表 (损坏/不存在的文件自动跳过)
        batch_size: GPU batch
        output_path: 可选, 保存为 .pt
    Returns:
        (N, 1024) 归一化嵌入矩阵
    """
    from utils.bioclip_embed import extract_embeddings_batch

    model, processor = _get_clip_model()

    valid_paths = []
    for p in image_paths:
        if not os.path.exists(p):
            print(f"  跳过(不存在): {p}")
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            valid_paths.append(p)
        except Exception as e:
            print(f"  跳过(损坏): {p} ({e})")

    result = extract_embeddings_batch(valid_paths, model, processor, batch_size=batch_size)

    if output_path:
        torch.save(result, output_path)
        print(f"嵌入保存: {output_path} {tuple(result.shape)}")

    return result


def extract_exsitu_embeddings(
    jsonl_path: str,
    output_path: str = None,
) -> Dict[str, torch.Tensor]:
    """提取APSMA图集标本的ex-situ嵌入。"""
    from tqdm import tqdm

    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))

    embeddings_map = {}
    for rec in tqdm(records, desc="提取ex-situ嵌入"):
        img_path = rec.get("image", {}).get("path", "")
        if img_path and os.path.exists(img_path):
            try:
                embeddings_map[rec["morphotype_id"]] = extract_single_clip_embedding(img_path)
            except Exception as e:
                print(f"  失败 [{rec['morphotype_id']}]: {e}")

    if output_path:
        torch.save(embeddings_map, output_path)
        print(f"Ex-situ嵌入保存: {output_path} ({len(embeddings_map)} 形态型)")

    return embeddings_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取BioCLIP 2.5嵌入")
    parser.add_argument("--jsonl", default="data/processed/apsma_parsed/apsma_knowledge_validated.jsonl")
    parser.add_argument("--output", default="models/clip_morphotype_embeddings.pt")
    # --batch-size 当前未使用: extract_exsitu_embeddings 采用逐条提取方式,
    # 以便与返回的 dict(morphotype_id -> embedding) 直接映射。
    parser.add_argument("--batch-size", type=int, default=32, help="(未使用)保留以兼容旧调用")
    args = parser.parse_args()

    extract_exsitu_embeddings(args.jsonl, args.output)
