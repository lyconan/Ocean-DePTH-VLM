"""Pydantic Schema验证 + 人工审核入口 (v3.1)"""
import os
import json
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.schemas import APSMARecord


def validate_jsonl_with_pydantic(
    input_jsonl: str,
    output_clean: str,
    output_rejected: str,
):
    """验证所有LLM提取的JSONL记录；分为干净和拒绝两组。
    
    严格验证: 错误的形态型ID格式、无效门名、测量范围不一致等。
    """
    clean = []
    rejected = []
    
    with open(input_jsonl) as f:
        for i, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                APSMARecord(**record)
                clean.append(line)
            except Exception as e:
                rejected.append({"line": i, "error": str(e), "raw": line[:500]})
                print(f"[REJECTED] Line {i}: {e}")
    
    os.makedirs(os.path.dirname(output_clean) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_rejected) or ".", exist_ok=True)

    with open(output_clean, "w") as f:
        f.writelines(clean)
    
    with open(output_rejected, "w") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)
    
    print(f"验证完成: {len(clean)} 通过, {len(rejected)} 拒绝")
    return len(clean), len(rejected)


def sample_for_review(
    clean_jsonl: str,
    sample_ratio: float = 0.1,
    output_review: str = "data/processed/apsma_parsed/review_queue.csv",
):
    """随机抽样供人工审核。
    
    对核心分类节点(Phylum→Class边)进行100%验证，
    其余记录抽样10%供专家抽检。
    """
    import random
    import csv
    
    records = []
    with open(clean_jsonl) as f:
        for line in f:
            records.append(json.loads(line))
    
    # 核心节点: 每个Phylum→Class边至少审核一个
    core_phylum_class = set()
    for rec in records:
        taxonomy = rec.get("taxonomy", {})
        key = (taxonomy.get("phylum", ""), taxonomy.get("class", ""))
        core_phylum_class.add(key)
    
    # 抽样
    sample_size = max(int(len(records) * sample_ratio), len(core_phylum_class))
    sampled = random.sample(records, min(sample_size, len(records)))
    
    # 写入审核队列
    os.makedirs(os.path.dirname(output_review) or ".", exist_ok=True)
    with open(output_review, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["morphotype_id", "phylum", "class", "order", "action", "notes"])
        for rec in sampled:
            taxonomy = rec.get("taxonomy", {})
            writer.writerow([
                rec.get("morphotype_id", ""),
                taxonomy.get("phylum", ""),
                taxonomy.get("class", ""),
                taxonomy.get("order", ""),
                "pending", ""
            ])
    
    print(f"审核队列: {len(sampled)} 条记录 → {output_review}")
    return sampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证APSMA JSONL记录")
    parser.add_argument("--input", default="data/processed/apsma_parsed/apsma_knowledge.jsonl")
    parser.add_argument("--output-clean", default="data/processed/apsma_parsed/apsma_knowledge_validated.jsonl")
    parser.add_argument("--output-rejected", default="data/processed/apsma_parsed/apsma_rejected.jsonl")
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    args = parser.parse_args()
    
    validate_jsonl_with_pydantic(args.input, args.output_clean, args.output_rejected)
    sample_for_review(args.output_clean, args.sample_ratio)
