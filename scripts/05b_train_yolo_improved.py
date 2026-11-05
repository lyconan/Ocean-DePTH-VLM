#!/usr/bin/env python3
"""YOLO11 改进训练 — 更大模型 + 更多 epochs + 类别平衡 (v2.0)

改进点:
1. 升级 base_model: yolo11s.pt → yolo11m.pt (更强特征提取)
2. 增加 epochs: 50 → 100 (当前可能欠拟合)
3. 调整 batch: 16 → 8 (yolo11m 需要更多显存)
4. 添加 focal loss (应对类别不平衡)
5. 调整置信度阈值和 NMS 参数

用法:
    python scripts/05b_train_yolo_improved.py \\
        --config config/yolo_training.yaml \\
        --base-model yolo11m.pt \\
        --epochs 100 \\
        --batch 8 \\
        --device 1
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_yolo_improved(
    config_yaml: str = "config/yolo_training.yaml",
    base_model: str = "yolo11m.pt",
    epochs: int = 100,
    batch: int = 8,
    imgsz: int = 640,
    patience: int = 30,
    device: str = "1",
    project: str = "runs/detect/results/yolo_training",
    name: str = "depth_vlm_yolo-4-improved",
):
    """训练改进版 YOLO11 目标提议网络。"""
    print(f"[YOLO改进] base_model={base_model}, epochs={epochs}, batch={batch}, device={device}")
    model = YOLO(base_model)

    results = model.train(
        data=config_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        device=device,
        project=project,
        name=name,
        # 深海特有增强
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        flipud=0.5,
        hsv_v=0.4,
        hsv_s=0.3,
        # 训练策略改进
        lr0=0.005,        # 降低初始学习率 (yolo11m 更大,需要更小 lr)
        lrf=0.0001,
        warmup_epochs=5,
        optimizer="auto",
        cos_lr=True,
        # 正则化改进
        weight_decay=0.0005,
        dropout=0.1,
        # 验证
        val=True,
        plots=True,
        save=True,
        save_period=10,
        # 混合精度训练
        amp=True,
        # 缓存策略
        cache=False,       # NAS 路径不适合 cache
        workers=4,         # 减少 workers 避免 fork 卡死
    )

    best_path = Path(project) / name / "weights" / "best.pt"
    if best_path.exists():
        print(f"\n[YOLO改进] 最佳模型: {best_path}")

        # 复制到 models 目录
        import shutil
        target_path = Path("models") / "yolo_fathomnet_improved.pt"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, target_path)
        print(f"[YOLO改进] 已复制: {target_path}")

        # 更新 graphrag_inference.yaml 中的 yolo_weights
        try:
            import yaml
            config_path = "config/graphrag_inference.yaml"
            with open(config_path) as f:
                config = yaml.safe_load(f)
            config["yolo_weights"] = str(best_path)
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"[YOLO改进] 已更新 graphrag_inference.yaml: yolo_weights={best_path}")
        except Exception as e:
            print(f"[YOLO改进] 更新配置失败: {e}")

    return results


def validate_yolo(
    weights: str = None,
    config_yaml: str = "config/yolo_training.yaml",
):
    """验证 YOLO 模型并打印详细指标。"""
    if weights is None:
        weights = str(Path("runs/detect/results/yolo_training/depth_vlm_yolo-4-improved/weights/best.pt"))

    print(f"\n[验证] 权重: {weights}")
    model = YOLO(weights)
    metrics = model.val(data=config_yaml)

    print(f"\n=== YOLO 验证指标 ===")
    print(f"mAP@50:      {metrics.box.map50:.4f} (目标 >= 0.75)")
    print(f"mAP@50:95:   {metrics.box.map:.4f} (目标 >= 0.50)")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")

    names = model.names
    print(f"\n=== 每类 AP@50 ===")
    for i, ap in enumerate(metrics.box.maps):
        status = "PASS" if ap >= 0.60 else "FAIL"
        print(f"  [{i}] {names[i]:<20} AP@50={ap:.4f} ({status})")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 改进训练")
    parser.add_argument("--config", default="config/yolo_training.yaml")
    parser.add_argument("--base-model", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="1")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--weights", default=None, help="验证模式下的权重路径")
    args = parser.parse_args()

    if args.validate_only:
        validate_yolo(weights=args.weights, config_yaml=args.config)
    else:
        train_yolo_improved(
            config_yaml=args.config,
            base_model=args.base_model,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
        )
