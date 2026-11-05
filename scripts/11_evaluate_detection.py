"""YOLO检测指标评估"""
import os
import json
import argparse
from pathlib import Path
from ultralytics import YOLO


def evaluate_detection(
    weights: str = "models/yolo_fathomnet.pt",
    config_yaml: str = "config/yolo_training.yaml",
    output_dir: str = "results/detection_evaluation",
):
    """评估YOLO检测指标。"""
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(weights)
    metrics = model.val(data=config_yaml, save_json=True, plots=True)

    results = {
        "mAP_50": float(metrics.box.map50),
        "mAP_50_95": float(metrics.box.map),
        "per_class_AP_50": {model.names[i]: float(v) for i, v in enumerate(metrics.box.maps)},
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    # 检查指标目标
    targets = {"mAP_50": 0.75, "mAP_50_95": 0.50}
    for metric, target in targets.items():
        actual = results[metric]
        status = "PASS" if actual >= target else "FAIL"
        print(f"  {metric}: {actual:.4f} (target >= {target}, {status})")

    # 每类AP@50
    per_class_ap = {}
    for i, r in enumerate(metrics.box.maps):
        class_name = model.names[i]
        per_class_ap[class_name] = float(r)
        status = "PASS" if r >= 0.60 else "FAIL"
        print(f"  AP@50 [{class_name}]: {r:.4f} (target >= 0.60, {status})")

    results["per_class_ap"] = per_class_ap

    output_path = Path(output_dir) / "detection_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果保存: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO检测评估")
    parser.add_argument("--weights", default="models/yolo_fathomnet.pt")
    parser.add_argument("--config", default="config/yolo_training.yaml")
    parser.add_argument("--output", default="results/detection_evaluation")
    args = parser.parse_args()

    evaluate_detection(args.weights, args.config, args.output)
