"""消融实验 A0-A10 — 评估各模块贡献"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))
from utils.module_loader import load_script_module

import pandas as pd
from tqdm import tqdm


def run_ablation(
    gold_standard_csv: str,
    config_path: str = "config/graphrag_inference.yaml",
    output_dir: str = "results/ablation",
):
    """运行消融实验A0-A10。

    A0: Baseline (YOLO only, conf>=0.5)
    A1: YOLO + VLM (no RAG)
    A2: YOLO + VectorOnly (exsitu)
    A3: YOLO + VectorOnly (multi-view)
    A5: YOLO + GraphRAG (K=5) [proposed method]
    """
    os.makedirs(output_dir, exist_ok=True)

    gold = pd.read_csv(gold_standard_csv)
    print(f"金标准: {len(gold)} 样本")

    ablation_results = {}

    # A5: 完整管线(提议方法)
    print("\n=== A5: YOLO + GraphRAG (K=5) [提议方法] ===")
    ablation_results["A5"] = _run_full_pipeline(gold, config_path, top_k=5)

    # A2: 仅向量检索(ex-situ)
    print("\n=== A2: YOLO + VectorOnly (exsitu) ===")
    ablation_results["A2"] = _run_vector_only(gold, config_path, use_insitu=False)

    # A3: 多视图向量检索
    print("\n=== A3: YOLO + VectorOnly (multi-view) ===")
    ablation_results["A3"] = _run_vector_only(gold, config_path, use_insitu=True)

    # A1: 无RAG
    print("\n=== A1: YOLO + VLM (no RAG) ===")
    ablation_results["A1"] = _run_no_rag(gold, config_path)

    # A0: YOLO only baseline
    print("\n=== A0: Baseline (YOLO only) ===")
    ablation_results["A0"] = _run_yolo_only(gold, config_path)

    # 保存结果
    output_path = Path(output_dir) / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    # 打印对比表
    print(f"\n=== 消融实验结果对比 ===")
    print(f"{'Experiment':<10} {'Phylum Acc':>12} {'Morphotype HR':>14} {'Cross-Phylum':>13}")
    for name, res in sorted(ablation_results.items()):
        pa = res.get("phylum_accuracy", 0)
        mhr = res.get("morphotype_hit_rate", 0)
        cpe = res.get("cross_phylum_error_rate", 0)
        print(f"{name:<10} {pa:>12.4f} {mhr:>14.4f} {cpe:>13.4f}")

    # 生成可视化（转换数据格式以匹配visualization.py的期望）
    from utils.visualization import plot_ablation_results
    viz_data = _convert_ablation_data_for_viz(ablation_results)
    viz_path = Path(output_dir) / "ablation_results.png"
    plot_ablation_results(viz_data, str(viz_path))
    print(f"可视化: {viz_path}")

    return ablation_results


def _get_pipeline_module():
    """动态加载10_graphrag_pipeline模块(文件名以数字开头)。"""
    return load_script_module('10_graphrag_pipeline')


def _run_full_pipeline(gold, config_path, top_k=5):
    """运行完整GraphRAG管线。"""
    pipeline_mod = _get_pipeline_module()
    config = pipeline_mod.load_config(config_path)
    config["top_k"] = top_k
    pipeline = pipeline_mod.DePTHGraphRAGPipeline(config)
    try:
        return _evaluate_pipeline(gold, pipeline)
    finally:
        pipeline.close()


def _run_vector_only(gold, config_path, use_insitu=True):
    """仅向量检索(无图谱丰富)。"""
    pipeline_mod = _get_pipeline_module()
    config = pipeline_mod.load_config(config_path)
    config["skip_graph_enrichment"] = True  # 添加跳过图谱遍历的标志
    if not use_insitu:
        config["skip_insitu_channel"] = True  # 跳过insitu通道
    pipeline = pipeline_mod.DePTHGraphRAGPipeline(config)
    try:
        return _evaluate_pipeline(gold, pipeline)
    finally:
        pipeline.close()


def _run_no_rag(gold, config_path):
    """无RAG: YOLO + VLM零样本。"""
    pipeline_mod = _get_pipeline_module()
    config = pipeline_mod.load_config(config_path)
    config["skip_rag"] = True  # 跳过整个RAG检索
    pipeline = pipeline_mod.DePTHGraphRAGPipeline(config)
    try:
        return _evaluate_pipeline(gold, pipeline)
    finally:
        pipeline.close()


def _run_yolo_only(gold, config_path):
    """仅YOLO基线。"""
    import yaml
    from ultralytics import YOLO

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = YOLO(config.get("yolo_weights", "models/yolo_fathomnet.pt"))

    correct_phylum = 0
    total = 0

    for _, row in tqdm(gold.iterrows(), desc="A0: YOLO only"):
        img_path = row.get("image_path", "")
        if not os.path.exists(img_path):
            continue

        results = model(img_path, conf=0.5)
        if results and results[0].boxes and len(results[0].boxes) > 0:
            class_id = int(results[0].boxes[0].cls)
            pred_phylum = model.names[class_id]
        else:
            pred_phylum = "unknown"

        gt_phylum = row.get("phylum", "")
        total += 1
        if pred_phylum == gt_phylum:
            correct_phylum += 1

    return {
        "phylum_accuracy": correct_phylum / total if total > 0 else 0,
        "morphotype_hit_rate": 0,
        "cross_phylum_error_rate": 0,
    }


def _evaluate_pipeline(gold, pipeline):
    """通用管线评估。"""
    correct_phylum = 0
    correct_morphotype = 0
    cross_phylum_error = 0
    total = 0

    for _, row in tqdm(gold.iterrows(), total=len(gold)):
        img_path = row.get("image_path", "")
        if not os.path.exists(img_path):
            continue

        results = pipeline.run(img_path)
        if not results:
            continue

        det = results[0]
        vlm_cls = det.get("vlm_classification", {})

        gt_phylum = row.get("phylum", "")
        gt_morphotype = row.get("apsma_morphotype_id", "")

        total += 1
        pred_phylum = vlm_cls.get("phylum", "unknown")
        pred_morphotype = vlm_cls.get("morphotype", "unknown")

        if pred_phylum == gt_phylum:
            correct_phylum += 1
        elif pred_phylum != "unknown":
            cross_phylum_error += 1
        if pred_morphotype == gt_morphotype:
            correct_morphotype += 1

    return {
        "phylum_accuracy": correct_phylum / total if total > 0 else 0,
        "morphotype_hit_rate": correct_morphotype / total if total > 0 else 0,
        "cross_phylum_error_rate": cross_phylum_error / total if total > 0 else 0,
    }


def _convert_ablation_data_for_viz(ablation_results: Dict) -> Dict:
    """将消融实验结果转换为visualization.py所需格式。

    输入: {"A0": {"phylum_accuracy": 0.5, ...}, "A1": {...}}
    输出: {"phylum_accuracy": {"A0": 0.5, "A1": ...}, ...}
    """
    viz_data = {
        "phylum_accuracy": {},
        "morphotype_hit_rate": {},
        "cross_phylum_error": {},
    }
    for name, res in sorted(ablation_results.items()):
        viz_data["phylum_accuracy"][name] = res.get("phylum_accuracy", 0)
        viz_data["morphotype_hit_rate"][name] = res.get("morphotype_hit_rate", 0)
        viz_data["cross_phylum_error"][name] = res.get("cross_phylum_error_rate", 0)
    return viz_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--gold-standard", default="data/test/gold_standard/gold_standard_morphotypes.csv")
    parser.add_argument("--config", default="config/graphrag_inference.yaml")
    parser.add_argument("--output", default="results/ablation")
    args = parser.parse_args()

    run_ablation(args.gold_standard, args.config, args.output)
