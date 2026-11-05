"""结果可视化工具"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from typing import List, Dict, Optional


def plot_detection_results(
    image_path: str,
    detections: List[Dict],
    output_path: Optional[str] = None,
    show_confidence: bool = True,
):
    """绘制YOLO检测结果。"""
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(img)

    colors = plt.cm.Set3(np.linspace(0, 1, 9))
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det.get("class_name", "unknown")
        conf = det.get("confidence", 0)
        class_id = det.get("class_id", 0)
        
        color = colors[class_id % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        if show_confidence:
            label = f"{class_name} {conf:.2f}"
        else:
            label = class_name
        ax.text(x1, y1 - 5, label, fontsize=10, color='white',
                bbox=dict(facecolor=color, alpha=0.7))

    ax.set_title(f"Detections: {len(detections)}")
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_graphrag_results(
    image_path: str,
    results: List[Dict],
    output_path: Optional[str] = None,
):
    """绘制GraphRAG管线完整结果。"""
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(img)

    conf_colors = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}
    
    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        vlm_cls = det.get("vlm_classification", {})
        phylum = vlm_cls.get("phylum", "unknown")
        confidence = vlm_cls.get("confidence", "LOW")
        color = conf_colors.get(confidence, "gray")
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        label = f"#{det.get('target_id', '?')} {phylum} [{confidence}]"
        ax.text(x1, y1 - 5, label, fontsize=9, color='white',
                bbox=dict(facecolor=color, alpha=0.7))

    ax.set_title(f"GraphRAG Results: {len(results)} targets")
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation_results(
    ablation_data: Dict,
    output_path: Optional[str] = None,
):
    """绘制消融实验结果对比表。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Phylum Accuracy
    if "phylum_accuracy" in ablation_data:
        axes[0].bar(ablation_data["phylum_accuracy"].keys(),
                    ablation_data["phylum_accuracy"].values())
        axes[0].set_title("Phylum Accuracy")
        axes[0].set_ylim(0, 1)
    
    # Morphotype Hit Rate
    if "morphotype_hit_rate" in ablation_data:
        axes[1].bar(ablation_data["morphotype_hit_rate"].keys(),
                    ablation_data["morphotype_hit_rate"].values())
        axes[1].set_title("Morphotype Hit Rate")
        axes[1].set_ylim(0, 1)
    
    # Cross-phylum Error Rate
    if "cross_phylum_error" in ablation_data:
        axes[2].bar(ablation_data["cross_phylum_error"].keys(),
                    ablation_data["cross_phylum_error"].values())
        axes[2].set_title("Cross-phylum Error Rate")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
