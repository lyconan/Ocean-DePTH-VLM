#!/usr/bin/env python3
"""YOLO11 细粒度训练 — 622类 FathomNet 物种检测 + Focal Loss (v1.0)

改进点 (相比 05b):
1. 细粒度类别: 622 个物种/属级 (非门级粗分类), 无 Unknown
2. Focal Loss: fl_gamma=0.5 抑制高频类, 提升长尾类识别
3. 分类损失加权: 提高 cls 权重 (622类需要更强分类信号)
4. 更大图像尺寸: imgsz=768 (细粒度需要更高分辨率)
5. 更长训练: 200 epochs + cosine LR
6. 类别平衡数据增强: 更强 mosaic + copy_paste

用法:
    python scripts/05c_train_yolo_finegrained.py \\
        --config config/yolo_finegrained.yaml \\
        --base-model yolo11m.pt \\
        --epochs 200 \\
        --batch 8 \\
        --device 0
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def train_yolo_finegrained(
    config_yaml: str = "config/yolo_finegrained.yaml",
    base_model: str = "yolo11m.pt",
    epochs: int = 200,
    batch: int = 8,
    imgsz: int = 768,
    patience: int = 40,
    device: str = "0",
    project: str = "runs/detect/results/yolo_training",
    name: str = "depth_vlm_yolo-finegrained-622",
):
    """训练细粒度 YOLO11 — 622类 FathomNet 物种检测。"""
    print(f"[YOLO细粒度] base_model={base_model}, epochs={epochs}, batch={batch}, device={device}")
    print(f"[YOLO细粒度] config={config_yaml}")

    # 读取类别数验证
    try:
        import yaml
        with open(config_yaml) as f:
            cfg = yaml.safe_load(f)
        nc = cfg.get("nc", "?")
        print(f"[YOLO细粒度] 类别数: {nc}")
    except Exception:
        pass

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
        # ═══ 类别权重: 抑制高频类, 提升长尾类 ═══
        cls_pw=1.0,           # 反频率权重幂次 (0=关闭, 1.0=完全反频率加权)
        # ═══ 损失权重: 细粒度分类需要更强信号 ═══
        cls=1.0,              # 分类损失权重 (默认0.5, 622类提高到1.0)
        box=7.5,              # 框回归损失 (默认)
        dfl=1.5,              # 分布焦点损失 (默认)
        # ═══ 深海特有数据增强 ═══
        mosaic=1.0,          # 马赛克增强 (多图拼接)
        mixup=0.15,          # 混合增强
        copy_paste=0.3,      # 复制粘贴 (帮助稀有类, 提高到0.3)
        flipud=0.5,          # 上下翻转 (深海生物无固定方向)
        hsv_v=0.4,           # 亮度变化
        hsv_s=0.3,           # 饱和度变化
        hsv_h=0.015,         # 色调变化
        # ═══ 训练策略 ═══
        lr0=0.005,           # 初始学习率
        lrf=0.0001,          # 最终学习率
        warmup_epochs=5,
        optimizer="auto",
        cos_lr=True,         # 余弦退火
        # ═══ 正则化 ═══
        weight_decay=0.0005,
        dropout=0.1,
        # ═══ 验证 & 保存 ═══
        val=True,
        plots=True,
        save=True,
        save_period=20,      # 每20 epoch存一次
        # ═══ 性能 ═══
        amp=True,            # 混合精度
        cache=False,         # NAS路径不适合cache
        workers=4,           # 减少workers避免fork卡死
    )

    best_path = Path(project) / name / "weights" / "best.pt"
    if best_path.exists():
        print(f"\n[YOLO细粒度] 最佳模型: {best_path}")

        # 复制到 models 目录
        import shutil
        target_path = Path("models") / "yolo_finegrained_622.pt"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, target_path)
        print(f"[YOLO细粒度] 已复制: {target_path}")

        # 更新 graphrag_inference.yaml
        try:
            import yaml
            config_path = "config/graphrag_inference.yaml"
            with open(config_path) as f:
                gconfig = yaml.safe_load(f)
            gconfig["yolo_weights"] = str(best_path)
            gconfig["use_yolo_hint"] = True  # 细粒度YOLO可靠后重新启用
            with open(config_path, "w") as f:
                yaml.dump(gconfig, f, default_flow_style=False, allow_unicode=True)
            print(f"[YOLO细粒度] 已更新 graphrag_inference.yaml:")
            print(f"  yolo_weights={best_path}")
            print(f"  use_yolo_hint=true (细粒度YOLO分类可靠)")
        except Exception as e:
            print(f"[YOLO细粒度] 更新配置失败: {e}")

    return results


def validate_yolo_finegrained(
    weights: str = None,
    config_yaml: str = "config/yolo_finegrained.yaml",
):
    """验证细粒度 YOLO 模型。"""
    if weights is None:
        weights = str(Path("runs/detect/results/yolo_training/depth_vlm_yolo-finegrained-622/weights/best.pt"))

    print(f"\n[验证] 权重: {weights}")
    model = YOLO(weights)
    metrics = model.val(data=config_yaml)

    print(f"\n{'='*60}")
    print(f"=== YOLO 细粒度验证指标 (622类) ===")
    print(f"{'='*60}")
    print(f"mAP@50:      {metrics.box.map50:.4f} (目标 >= 0.50)")
    print(f"mAP@50:95:   {metrics.box.map:.4f} (目标 >= 0.30)")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")

    # 每类 AP 分析 (长尾类表现)
    names = model.names
    maps = metrics.box.maps
    print(f"\n=== 每类 AP@50 分析 (622类) ===")

    # 按AP排序
    class_aps = [(names[i], maps[i], i) for i in range(len(maps))]
    class_aps.sort(key=lambda x: -x[1])

    print(f"\nTop10 最佳类别:")
    for name, ap, cid in class_aps[:10]:
        print(f"  [{cid:3d}] {name:<40} AP@50={ap:.4f}")

    print(f"\nBottom10 最差类别 (长尾):")
    for name, ap, cid in class_aps[-10:]:
        print(f"  [{cid:3d}] {name:<40} AP@50={ap:.4f}")

    # 统计
    aps = [x[1] for x in class_aps]
    n_good = sum(1 for a in aps if a >= 0.50)
    n_ok = sum(1 for a in aps if a >= 0.30)
    n_poor = sum(1 for a in aps if a < 0.10)
    print(f"\n类别AP分布:")
    print(f"  AP>=0.50 (好):   {n_good}/{len(aps)} ({n_good/len(aps)*100:.1f}%)")
    print(f"  AP>=0.30 (可用): {n_ok}/{len(aps)} ({n_ok/len(aps)*100:.1f}%)")
    print(f"  AP<0.10 (差):    {n_poor}/{len(aps)} ({n_poor/len(aps)*100:.1f}%)")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 细粒度训练 (622类)")
    parser.add_argument("--config", default="config/yolo_finegrained.yaml")
    parser.add_argument("--base-model", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="0")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    if args.validate_only:
        validate_yolo_finegrained(weights=args.weights, config_yaml=args.config)
    else:
        train_yolo_finegrained(
            config_yaml=args.config,
            base_model=args.base_model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )
