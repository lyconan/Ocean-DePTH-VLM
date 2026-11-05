#!/usr/bin/env python3
"""BioCLIP 嵌入提取共享工具 (v2.0 — BioCLIP 2.5 ViT-H/14)

模型: imageomics/bioclip-2.5-vith14 (https://huggingface.co/imageomics/bioclip-2.5-vith14)
- 嵌入维度 1024 (ViT-H/14, 由模型 embed_dim 动态获取)
- 输出始终 L2 归一化

v2.0 变更 (相对 v1.0):
- 默认模型从 BioCLIP 1.0 (hf-hub:imageomics/bioclip, 512维) 升级为 BioCLIP 2.5 (1024维)
- 移除 openai/clip 回退 (维度不一致会污染向量库; 模型加载失败直接抛错)
- 新增 batch 提取 extract_embeddings_batch (GPU 批量推理)
- extract_embedding 返回维度变为 1024 — 调用方禁止硬编码 512

用法:
    from utils.bioclip_embed import get_bioclip_model, extract_embedding
    model, processor = get_bioclip_model(device="cuda:0")
    emb = extract_embedding(pil_image, model, processor)  # (1024,) tensor

v2.1 变更 (2026-11-03):
- get_bioclip_model 默认加载 LoRA v2 微调适配器 (data/models/bioclip_lora/lora_adapter_v2)
  与 Neo4j 向量库 (entryimage_embeddings, v2 重提后) 保持同一嵌入空间
- 显式 lora=False 可退回基座 (仅用于仍查询基座索引的路径, 如 morphotype_exsitu_embeddings)
- 环境变量 BIOCLIP_LORA=0 全局禁用 (排障用)
- 适配器加载失败 (peft 缺失/目录缺失) 打印醒目警告并退回基座, 不中断服务
"""
from __future__ import annotations

import os
import torch
from PIL import Image
from typing import List, Optional, Tuple, Union

MODEL_HUB_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"
EXPECTED_DIM = 1024  # ViT-H/14; 实际以 model.embed_dim 为准

# LoRA v2 适配器目录 (相对工作区根: scripts/../data/models/bioclip_lora/lora_adapter_v2)
ADAPTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "models", "bioclip_lora", "lora_adapter_v2")

# 全局模型缓存 (支持按 device 缓存多份)
_CLIP_MODELS: dict = {}


def _resolve_lora_flag(lora: Optional[bool]) -> bool:
    if lora is not None:
        return bool(lora)
    return os.environ.get("BIOCLIP_LORA", "1") not in ("0", "false", "False")


def get_bioclip_model(device: Optional[str] = None,
                      lora: Optional[bool] = None) -> Tuple:
    """懒加载 BioCLIP 2.5 (ViT-H/14, 1024维), 默认叠加 LoRA v2 适配器。

    Args:
        device: "cuda:0" / "cpu" 等; None 则自动 (cuda 可用即 cuda:0)
        lora:   None=按环境变量 BIOCLIP_LORA (默认开);
                True=强制加载; False=强制基座 (用于基座嵌入空间的索引)
    Returns:
        (model, processor) tuple; model 已 eval() 且移到指定设备 (fp32)
    """
    use_lora = _resolve_lora_flag(lora)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cache_key = (device, use_lora)
    if cache_key in _CLIP_MODELS:
        return _CLIP_MODELS[cache_key]

    import open_clip
    model, _, processor = open_clip.create_model_and_transforms(MODEL_HUB_NAME)
    model.eval()
    model = model.float()  # fp32 推理 (与同事 Jin 的 b25 流程一致)

    if use_lora:
        try:
            import peft
            if not os.path.isdir(ADAPTER_DIR):
                raise FileNotFoundError(f"适配器目录不存在: {ADAPTER_DIR}")
            model.visual = peft.PeftModel.from_pretrained(
                model.visual, ADAPTER_DIR).eval()
            model = model.to(device)
            print(f"[BioCLIP2.5] 模型加载成功 ({MODEL_HUB_NAME}, dim={EXPECTED_DIM}, "
                  f"device={device}) + LoRA v2 适配器已加载 ({ADAPTER_DIR})")
        except Exception as e:
            # 关键: 检索库已是 v2 空间, 退回基座会导致查询/库嵌入失配
            print(f"[BioCLIP2.5] !!! 警告: LoRA v2 适配器加载失败 ({e}), "
                  f"当前为基座模型 — 查询向量与库 (v2) 失配, 检索效果将严重下降!")
            print(f"[BioCLIP2.5] 模型加载成功 (基座, dim={EXPECTED_DIM}, device={device})")
    else:
        model = model.to(device)
        print(f"[BioCLIP2.5] 模型加载成功 (基座 {MODEL_HUB_NAME}, dim={EXPECTED_DIM}, "
              f"device={device}, lora=False)")

    _CLIP_MODELS[cache_key] = (model, processor)
    return _CLIP_MODELS[cache_key]


def is_bioclip() -> bool:
    """兼容旧接口; v2.0 起恒为 True (无 CLIP 回退)。"""
    return True


def get_embedding_dim(model) -> int:
    """动态获取嵌入维度 (1024)。

    open_clip 3.x 的 CLIP 对象没有 embed_dim 属性, 用 visual.output_dim。
    """
    dim = getattr(model, "embed_dim", None)
    if not dim:
        visual = getattr(model, "visual", None)
        dim = getattr(visual, "output_dim", None)
    return int(dim) if dim else EXPECTED_DIM


def _load_pil(image: Union[Image.Image, str]) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return image.convert("RGB")


def _preprocess_image(image: Union[Image.Image, str], processor) -> torch.Tensor:
    """open_clip preprocess (torchvision Compose) -> (1, C, H, W) cpu tensor."""
    return processor(_load_pil(image)).unsqueeze(0)


def extract_embedding(
    image: Union[Image.Image, str],
    model=None,
    processor=None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """提取单张 BioCLIP 2.5 嵌入 (1024维, L2归一化, cpu)。

    Args:
        image: PIL.Image 或图像路径
        model/processor: 已加载的模型; 不传则自动加载 (device 生效)
        device: 仅在自动加载时生效
    Returns:
        (1024,) 归一化嵌入 tensor (cpu)
    """
    if model is None or processor is None:
        model, processor = get_bioclip_model(device)
        device = next(model.parameters()).device
    elif device is None:
        device = next(model.parameters()).device

    pixel_values = _preprocess_image(image, processor).to(device)

    with torch.no_grad():
        features = model.encode_image(pixel_values)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).float().cpu()


def extract_embeddings_batch(
    images: List[Union[Image.Image, str]],
    model,
    processor,
    batch_size: int = 32,
    device: Optional[str] = None,
    num_workers: int = 0,
) -> torch.Tensor:
    """批量提取 BioCLIP 2.5 嵌入 (GPU 批量推理, 加速全量提取)。

    Args:
        images: PIL.Image 或路径列表
        model/processor: 已加载模型
        batch_size: GPU batch
        device: 推理设备 (默认取模型当前设备)
        num_workers: DataLoader 预处理进程数
    Returns:
        (N, 1024) 归一化嵌入 tensor (cpu, fp32)
    """
    from torch.utils.data import DataLoader, Dataset

    if device is None:
        device = next(model.parameters()).device

    class _ImgDataset(Dataset):
        def __init__(self, items, proc):
            self.items = items
            self.proc = proc

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            img = self.items[idx]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            return self.proc(img.convert("RGB"))

    loader = DataLoader(
        _ImgDataset(images, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu())
    return torch.cat(feats, dim=0)


def extract_embedding_list(
    image: Union[Image.Image, str],
    model=None,
    processor=None,
    device: Optional[str] = None,
) -> list:
    """与 extract_embedding 相同, 但返回 list[float] (便于 Neo4j 序列化)。"""
    emb = extract_embedding(image, model, processor, device)
    return emb.tolist()
