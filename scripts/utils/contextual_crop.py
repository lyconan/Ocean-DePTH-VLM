"""上下文裁剪 — 自适应BBox填充 + 紧密裁剪 (v3.1.1)"""
from PIL import Image
from typing import Tuple


def contextual_crop(
    image_pil: Image.Image,
    bbox: list,
    confidence: float,
    padding_ratio: float = 0.25,
) -> Image.Image:
    """带自适应填充的裁剪，保留低置信度检测的边缘特征。
    
    Args:
        image_pil: 完整海底图像
        bbox: [x1, y1, x2, y2] YOLO坐标
        confidence: YOLO检测置信度
        padding_ratio: BBox扩展比例(默认25%)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    img_w, img_h = image_pil.size

    if confidence < 0.3:
        pad = max(padding_ratio, 0.30)
    elif confidence < 0.5:
        pad = padding_ratio
    else:
        pad = 0.10

    px = int(w * pad)
    py = int(h * pad)

    new_x1 = max(0, int(x1 - px))
    new_y1 = max(0, int(y1 - py))
    new_x2 = min(img_w, int(x2 + px))
    new_y2 = min(img_h, int(y2 + py))

    return image_pil.crop((new_x1, new_y1, new_x2, new_y2))


def tight_crop(
    image_pil: Image.Image,
    bbox: list,
    shrink_ratio: float = 0.05,
) -> Image.Image:
    """紧密裁剪 — BioCLIP嵌入提取用 (v3.1.1)。
    
    BioCLIP的全局池化对背景像素极度敏感，紧密裁剪确保
    只包含生物体像素，零背景污染。
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    sx = int(w * shrink_ratio)
    sy = int(h * shrink_ratio)

    new_x1 = int(x1 + sx)
    new_y1 = int(y1 + sy)
    new_x2 = int(x2 - sx)
    new_y2 = int(y2 - sy)

    if new_x2 <= new_x1:
        new_x1, new_x2 = int(x1), int(x2)
    if new_y2 <= new_y1:
        new_y1, new_y2 = int(y1), int(y2)

    return image_pil.crop((new_x1, new_y1, new_x2, new_y2))
