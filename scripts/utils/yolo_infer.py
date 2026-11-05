#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双路 YOLO 推理模块: 同一权重, 两套推理视图合并, 零重训提升检出率。

实测背景 (2,848 图验证子集, 真值 7,253 框, recall@IoU0.5):
  基线 imgsz=960 conf=0.25            78.7%
  单路 960+TTA conf=0.01              88.3%
  单路 1920 原生 conf=0.01            83.1%
  双路并集 (960+TTA ∪ 1920) conf0.05  90.8%
  三路并集 (+1280 切片) conf0.01      91.3% (推理侧饱和; 95% 需重训)

设计要点:
  - 宁多勿漏: 检测层低 conf 多检出, 误检交由检索层拒识阈值 (~0.90)
    与 VLM 细判兜底。
  - 1920 单路反而更差 (训练 imgsz=960, 1920 属尺度分布外), 但与
    960+TTA 高度互补, 并集显著高于任一单路。
  - 高分辨率路 (imgsz>=1536) 默认 batch=16 防 OOM; 每路推理后清空显存。

用法 (scripts 目录需在 sys.path, utils 为命名空间包):
    from utils.yolo_infer import DualPathDetector
    det = DualPathDetector("runs/detect/yolo11m_class34-2/weights/best.pt")
    results = det.predict(img_paths, conf=0.05)
    # List[List[dict]]: 每图检测列表, dict = {box, conf, cls, name}
"""


def _iou(a, b):
    """框 [x1,y1,x2,y2] 之间的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def merge_detections(path_dets, conf=0.05, iou_nms=0.55):
    """合并多路原始检测: conf 过滤 + 逐类贪心 NMS + conf 降序。

    path_dets: List[List[dict]] — 各路对同一张图的原始检测列表,
        每项含 box[x1,y1,x2,y2] / conf / cls (cls 可为类名或 id)。
    跨路重复框由逐类贪心 NMS 去重 (保留高 conf), 不同类之间不互斥。
    """
    cands = sorted((d for dets in path_dets for d in dets if d["conf"] >= conf),
                   key=lambda d: -d["conf"])
    out = []
    for d in cands:
        if all(d["cls"] != o["cls"] or _iou(d["box"], o["box"]) <= iou_nms
               for o in out):
            out.append(d)
    return out


class DualPathDetector:
    """双路 YOLO 检测器: 同一权重, 两套推理视图 (960+TTA / 1920 原生)。

    weights: best.pt 路径 (34 纲模型, 训练 imgsz=960)
    device : "cuda:0" / "cuda:1" / "cpu"
    paths  : 每路 (imgsz, augment) 配置, 默认 ((960, True), (1920, False))
    """

    def __init__(self, weights, device="cuda:1",
                 paths=((960, True), (1920, False))):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.paths = paths

    def _raw(self, imgs, conf, imgsz, augment, batch):
        """单路批量推理 -> 每图原始检测列表 (不做跨路合并)。"""
        pb = batch or (16 if imgsz >= 1536 else 64)
        out = [[] for _ in imgs]
        for i in range(0, len(imgs), pb):
            preds = self.model.predict(imgs[i:i + pb], conf=conf, imgsz=imgsz,
                                       augment=augment, device=self.device,
                                       verbose=False)
            for j, r in enumerate(preds):
                dets = []
                if r.boxes is not None:
                    names = r.names
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        dets.append({
                            "box": [round(x1), round(y1), round(x2), round(y2)],
                            "conf": round(float(b.conf[0]), 4),
                            "cls": int(b.cls[0]),
                            "name": names[int(b.cls[0])],
                        })
                out[i + j] = dets
        return out

    def predict(self, imgs, conf=0.05, iou_nms=0.55, batch=None):
        """生产入口: 双路推理 + merge_detections 合并。

        imgs: 图像路径列表 (也接受 PIL/ndarray, 同 ultralytics)
        返回 List[List[dict]] — 每图合并后检测列表 (conf 降序)。
        """
        per_path = [self._raw(imgs, conf, imgsz, aug, batch)
                    for imgsz, aug in self.paths]
        if str(self.device).startswith("cuda"):
            import torch
            torch.cuda.empty_cache()
        return [merge_detections([p[i] for p in per_path], conf=conf,
                                 iou_nms=iou_nms) for i in range(len(imgs))]
