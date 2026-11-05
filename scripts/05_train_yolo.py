"""YOLO11训练 — 深海底栖生物目标提议网络 (Object Proposal Network)"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_yolo(
    config_yaml: str = "config/yolo_training.yaml",
    base_model: str = "yolo11s.pt",
    epochs: int = 150,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 25,
    project: str = "results/yolo_training",
    name: str = "depth_vlm_yolo",
):
    """训练YOLO11目标提议网络。"""
    model = YOLO(base_model)

    results = model.train(
        data=config_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        project=project,
        name=name,
        # 深海特有增强
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        flipud=0.5,       # 垂直翻转(海底生物无重力方向偏好)
        hsv_v=0.4,        # 亮度增强(模拟不同ROV高度照明)
        # 训练策略
        lr0=0.01,
        lrf=0.0001,
        warmup_epochs=5,
        optimizer="auto",
        cos_lr=True,
        # 验证
        val=True,
        plots=True,
        save=True,
        save_period=10,
    )

    # 保存最佳模型
    best_path = Path(project) / name / "weights" / "best.pt"
    target_path = Path("models") / "yolo_fathomnet.pt"
    if best_path.exists():
        import shutil
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, target_path)
        print(f"最佳模型保存: {target_path}")

    return results


def validate_yolo(
    weights: str = "models/yolo_fathomnet.pt",
    config_yaml: str = "config/yolo_training.yaml",
):
    """验证YOLO模型。"""
    model = YOLO(weights)
    metrics = model.val(data=config_yaml)

    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50:95: {metrics.box.map:.4f}")
    print(f"Per-class AP@50: {metrics.box.maps}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练YOLO11")
    parser.add_argument("--config", default="config/yolo_training.yaml")
    parser.add_argument("--base-model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate_yolo(config_yaml=args.config)
    else:
        train_yolo(
            config_yaml=args.config,
            base_model=args.base_model,
            epochs=args.epochs,
            batch=args.batch,
        )
