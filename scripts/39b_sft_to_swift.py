#!/usr/bin/env python3
"""39b: 系统现成 SFT 数据 (data/processed/sft/) -> ms-swift QLoRA 训练格式.

把方案 D 净化后的三类任务统一转成 ms-swift messages 格式:
  - species/class/genus 任务 (11 图选择题: 1 查询图 + 10 参考图 -> 选项字母)
  - name 任务 (单图开放命名 -> 物种名文本)

用法:
  python3 scripts/39b_sft_to_swift.py \
      --src data/processed/sft --out data/processed/sft_swift
  # 默认转换全部 sft_*.jsonl; --files sft_train,sft_val 可指定子集
"""
import argparse
import json
from pathlib import Path

LETTERS = "ABCDEFGHIJ"

Q_SPECIES = (
    "第1张图为待鉴定的深海底栖生物裁剪图。随后10张参考图依次对应候选物种 "
    "A-J。请判断第1张图与哪个候选是同一物种，只输出一个选项字母。\n")

Q_BY_TASK = {
    "species": Q_SPECIES,
    "genus": ("第1张图为待鉴定的深海底栖生物裁剪图。随后10张参考图依次对应候选属 "
              "A-J。请判断第1张图属于哪个候选属，只输出一个选项字母。\n"),
    "class": ("第1张图为待鉴定的深海底栖生物裁剪图。随后10张参考图依次对应候选纲 "
              "A-J。请判断第1张图属于哪个候选纲，只输出一个选项字母。\n"),
}


def opt_label(o: dict) -> str:
    return o.get("latin") or o.get("genus") or o.get("class") or o.get("mid", "")


def mcq_to_swift(rec: dict) -> dict:
    opts = rec["options"][:10]
    opt_lines = "\n".join(
        f"{o['letter']}. {opt_label(o)}（参考图{refs_i}）"
        for refs_i, o in enumerate(opts, start=2))
    question = Q_BY_TASK.get(rec["task"], Q_SPECIES) + opt_lines
    return {
        "messages": [
            {"role": "user", "content": f"<img>{rec['query_path']}\n" +
             "".join(f"<img>{r}" for r in rec["refs"][:10]) + "\n" + question},
            {"role": "assistant", "content": rec["answer"]},
        ],
        "images": [rec["query_path"]] + rec["refs"][:10],
    }


def name_to_swift(rec: dict) -> dict:
    question = "识别图中的深海底栖生物，直接输出它的分类单元名称（拉丁名），不要输出其他内容。"
    return {
        "messages": [
            {"role": "user", "content": f"<img>{rec['query_path']}\n{question}"},
            {"role": "assistant", "content": rec["answer"]},
        ],
        "images": [rec["query_path"]],
    }


def convert_file(src: Path, out: Path) -> tuple:
    n, skip = 0, 0
    with open(out, "w", encoding="utf-8") as fo:
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["task"] == "name":
                out_rec = name_to_swift(rec)
            elif rec.get("options") and rec.get("refs"):
                out_rec = mcq_to_swift(rec)
            else:
                skip += 1
                continue
            fo.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n += 1
    return n, skip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/processed/sft")
    ap.add_argument("--out", default="data/processed/sft_swift")
    ap.add_argument("--files", default="sft_train,sft_crops_train,sft_name_train,sft_val",
                    help="逗号分隔, 不带扩展名")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in args.files.split(","):
        f = src / f"{name.strip()}.jsonl"
        if not f.exists():
            print(f"跳过(不存在): {f}")
            continue
        n, skip = convert_file(f, out / f"{name.strip()}_swift.jsonl")
        total += n
        print(f"{name}: {n} 条 (跳过 {skip})")
    print(f"合计 {total} 条 -> {out.resolve()}")


if __name__ == "__main__":
    main()
