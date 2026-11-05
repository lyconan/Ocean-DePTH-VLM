#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo34_classes.py — 34 纲级 YOLO 检测模型类别表 (单一事实来源)

顺序即 data.yaml 的 class id 顺序 (Other 永远最后), 与以下产物保持一致:
  - data/processed/fathomnet_class_yolo/data.yaml (06d_build_class_yolo.py 生成)
  - runs/detect/yolo11m_class34-2/weights/best.pt 的类别顺序

用法:
    from utils.yolo34_classes import YOLO_CLASSES

更换类别体系时只需改本文件, 并同步 data/processed/taxonomy/yolo34_class_alias_map.json。
"""

YOLO_CLASSES = ["Octocorallia", "Malacostraca", "Holothuroidea", "Hexacorallia",
                "Hexactinellida", "Actinopterygii", "Echinoidea", "Asteroidea",
                "Hydrozoa", "Polychaeta", "Demospongiae", "Cephalopoda", "Bivalvia",
                "Scyphozoa", "Crinoidea", "Elasmobranchii", "Gastropoda", "Ophiuroidea",
                "Thaliacea", "Appendicularia", "Tentaculata", "Ascidiacea", "Nuda",
                "Thecostraca", "Myxini", "Rhynchonellata", "Holocephali", "Sagittoidea",
                "Clitellata", "Pycnogonida", "Enteropneusta", "Gymnolaemata",
                "Polyplacophora", "Other"]

assert len(YOLO_CLASSES) == 34, f"34 纲体系应为 34 类, 当前 {len(YOLO_CLASSES)}"
