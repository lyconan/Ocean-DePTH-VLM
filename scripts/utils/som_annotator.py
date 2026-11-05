"""Set-of-Mark (SoM) 图像标注 — 自适应渲染防止遮挡 (v3.1.1)"""
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict


def draw_som_annotations(
    image_pil: Image.Image,
    detections: List[Dict],
) -> Image.Image:
    """自适应SoM渲染 — 大目标用矩形标注，小/密集目标用中心点+引线。
    
    Args:
        image_pil: 原始海底图像
        detections: 检测列表，每个含bbox, target_id, confidence
    """
    annotated = image_pil.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    overlap_map = _compute_overlap_map(detections)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        target_id = det["target_id"]
        conf = det["confidence"]

        if conf >= 0.5:
            color = (0, 255, 0)
        elif conf >= 0.3:
            color = (255, 255, 0)
        else:
            color = (255, 0, 0)

        is_small = (x2 - x1) < 50 or (y2 - y1) < 50
        max_overlap = overlap_map.get(target_id, 0)
        is_dense = max_overlap > 0.15

        if is_small or is_dense:
            # 策略A: 中心点 + 引线 + 外部标签
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

            dot_r = 4
            draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                        fill=color, outline=(0, 0, 0), width=1)

            label_x, label_y = _find_label_position(det, detections, annotated.size)
            draw.line([(cx, cy), (label_x, label_y)], fill=color, width=1)

            label = f"#{target_id}"
            label_bbox = draw.textbbox((label_x, label_y), label, font=font_small)
            draw.rectangle(label_bbox, fill=(*color, 153))
            draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font_small)
        else:
            # 策略B: 标准SoM + 半透明标签
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"#{target_id}"
            label_y_pos = max(0, y1 - 24)
            label_bbox = draw.textbbox((x1, label_y_pos), label, font=font)
            draw.rectangle(label_bbox, fill=(*color, 153))
            draw.text((x1, label_y_pos), label, fill=(0, 0, 0), font=font)

    return annotated


def _compute_overlap_map(detections: List[Dict]) -> Dict[int, float]:
    """计算每个检测与其他检测的最大IoU。"""
    overlap_map = {}
    for i, det_a in enumerate(detections):
        max_iou = 0.0
        ax1, ay1, ax2, ay2 = det_a["bbox"]
        for j, det_b in enumerate(detections):
            if i == j:
                continue
            bx1, by1, bx2, by2 = det_b["bbox"]
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_a + area_b - inter
            iou = inter / union if union > 0 else 0
            max_iou = max(max_iou, iou)
        overlap_map[det_a["target_id"]] = max_iou
    return overlap_map


def _find_label_position(det: Dict, all_detections: List[Dict], image_size: tuple) -> tuple:
    """寻找不与其他BBox重叠的标签位置。"""
    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
    img_w, img_h = image_size

    candidates = [
        (x2 + 8, y1),
        (max(0, x1 - 40), y1),
        (x1, max(0, y1 - 30)),
        (x1, y2 + 8),
    ]
    for cx, cy in candidates:
        if 0 <= cx < img_w - 30 and 0 <= cy < img_h - 20:
            return cx, cy
    return min(x2 + 8, img_w - 30), y1
