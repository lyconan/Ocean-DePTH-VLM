#!/usr/bin/env python3
"""39: 自建数据集 -> VLM 物种鉴定微调数据 (SFT JSONL) 生成器.

面向第三方部署: 把"按物种分文件夹"的图像数据集转成与本系统一致的
11 图选择题微调格式 (1 张查询图 + 10 张候选参考图, 输出选项字母)。

输入数据集布局 (最简约定, 文件夹名 = 物种名):
    dataset_root/
      My_species_A/  img001.jpg img002.png ...
      My_species_B/  a.jpeg b.jpg ...
      ...            (每类建议 >=20 张, 上限 --cap-per-class 默认 100)

可选: 某类文件夹下放 _meta.json 填谱系字段:
    {"latin": "Heterochone calyx", "genus": "Heterochone", "class": "Hexactinellida"}

输出 (默认 ./sft_out/):
    sft_train.jsonl / sft_val.jsonl   -- 本系统格式 (39a 过滤后同构)
    swift_train.jsonl / swift_val.jsonl -- ms-swift QLoRA 训练直读格式
    dataset_stats.json               -- 每类样本数/划分摘要

用法:
    python3 scripts/39_build_sft_dataset.py --src /path/to/dataset_root --out ./sft_out
    # 常用参数: --val-ratio 0.1 --cap-per-class 100 --variants 1 --seed 42
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LETTERS = "ABCDEFGHIJK"


def load_class_meta(folder: Path):
    meta_path = folder / "_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"警告: {meta_path} 解析失败({e}), 忽略")
    return {}


def collect_dataset(src: Path, cap: int, rng: random.Random):
    """返回 {class_name: {"latin","genus","class","images":[Path,...]}}"""
    classes = {}
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        imgs = sorted(p for p in folder.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXT)
        if not imgs:
            continue
        meta = load_class_meta(folder)
        name = folder.name
        classes[name] = {
            "latin": meta.get("latin", name),
            "genus": meta.get("genus", name.split()[0] if " " in name else name),
            "class": meta.get("class", "Unknown"),
            "images": rng.sample(imgs, min(cap, len(imgs))),
        }
    return classes


def split_train_val(classes: dict, val_ratio: float, rng: random.Random):
    train, val = [], []  # [(cls, img)]
    for name, c in classes.items():
        imgs = c["images"][:]
        rng.shuffle(imgs)
        n_val = 1 if (len(imgs) >= 5 and val_ratio > 0) else 0
        n_val = max(n_val, round(len(imgs) * val_ratio)) if len(imgs) >= 5 else 0
        n_val = min(n_val, len(imgs) // 2)  # 训练侧至少留一半
        val.extend((name, p) for p in imgs[:n_val])
        train.extend((name, p) for p in imgs[n_val:])
    return train, val


def build_record(query_cls, query_img, classes, split_pool, rng, task="species"):
    """一条 11 图选择题: 候选 = GT + 10 干扰项 (优先同类易混, 再随机)。

    split_pool: 本 split 可用的全部 (cls, img) —— 干扰项与参考图只取自本 split,
    保证 val 记录不引用 train 图 (反之亦然), 防泄漏。
    """
    by_cls = split_pool["by_cls"]
    imgs_of = split_pool["imgs_of"]

    # ---- 选干扰类: 同 class 谱系优先 (难例), 再全局随机 ----
    # 协议: 1 张查询图 + 10 个选项 (A-J) = 11 图
    gt_meta = classes[query_cls]
    same_tax = [c for c in by_cls if c != query_cls
                and classes[c]["class"] == gt_meta["class"] and gt_meta["class"] != "Unknown"]
    others = [c for c in by_cls if c != query_cls]
    pool = same_tax[:]; rng.shuffle(pool)
    rest = [c for c in others if c not in pool]; rng.shuffle(rest)
    distr_cls = (pool + rest)[:9]
    if len(distr_cls) < 9:
        return None  # 类别数不足 10, 无法构造选择题

    cand_cls = [query_cls] + distr_cls

    # ---- 每个候选配 1 张参考图 (排除查询图自身) ----
    options, refs = [], []
    order = list(range(len(cand_cls)))
    rng.shuffle(order)  # GT 位置随机 -> 防位置偏置
    letters = LETTERS[:len(order)]
    answer = None
    for slot, ci in enumerate(order):
        cls = cand_cls[ci]
        avail = [p for p in imgs_of[cls] if p != query_img]
        if not avail:
            return None  # 该类只有查询图自己, 跳过
        ref = rng.choice(avail)
        refs.append(str(ref))
        options.append({
            "letter": letters[slot],
            "mid": cls,
            "latin": classes[cls]["latin"],
            "genus": classes[cls]["genus"],
            "class": classes[cls]["class"],
            "ref": str(ref),
        })
        if cls == query_cls:
            answer = letters[slot]

    return {
        "task": task,
        "variant": "v1",
        "query_path": str(query_img),
        "mid": query_cls,
        "refs": refs,
        "options": options,
        "answer": answer,
    }


def to_swift(rec: dict):
    """转 ms-swift messages 格式 (11 张图: 查询 + 10 参考图)。"""
    opt_lines = "\n".join(
        f"{o['letter']}. {o['latin']}（参考图{refs_i}）"
        for refs_i, o in enumerate(rec["options"], start=2))
    question = (
        "第1张图为待鉴定的深海底栖生物裁剪图。随后10张参考图依次对应候选物种 "
        "A-J。请判断第1张图与哪个候选是同一物种，只输出一个选项字母。\n"
        + opt_lines)
    return {
        "messages": [
            {"role": "user", "content": f"<img>{rec['query_path']}\n" +
             "".join(f"<img>{r}" for r in rec["refs"]) + "\n" + question},
            {"role": "assistant", "content": rec["answer"]},
        ],
        "images": [rec["query_path"]] + rec["refs"],
    }


def make_split_records(pairs, classes, rng, variants):
    """先建本 split 索引, 再逐查询出题。"""
    pool = {"by_cls": sorted({c for c, _ in pairs}),
            "imgs_of": defaultdict(list)}
    for c, p in pairs:
        pool["imgs_of"][c].append(p)
    records = []
    for cls, img in pairs:
        for _ in range(variants):
            rec = build_record(cls, img, classes, pool, rng)
            if rec:
                records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="数据集根目录 (每类一个文件夹)")
    ap.add_argument("--out", default="./sft_out", help="输出目录")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--cap-per-class", type=int, default=100)
    ap.add_argument("--variants", type=int, default=1,
                    help="每张查询图生成几条题目(不同字母排列)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    classes = collect_dataset(Path(args.src), args.cap_per_class, rng)
    if len(classes) < 10:
        raise SystemExit(f"类别数不足: 仅 {len(classes)} 类, 10 选项选择题至少需要 10 类。"
                         "可先合并相近类或扩充数据集。")
    n_imgs = sum(len(c["images"]) for c in classes.values())
    print(f"数据集: {len(classes)} 类 / {n_imgs} 张图")

    train_pairs, val_pairs = split_train_val(classes, args.val_ratio, rng)
    print(f"划分: train {len(train_pairs)} 张 / val {len(val_pairs)} 张")

    train_recs = make_split_records(train_pairs, classes, rng, args.variants)
    val_recs = make_split_records(val_pairs, classes, rng, args.variants)
    print(f"题目: train {len(train_recs)} 条 / val {len(val_recs)} 条")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_recs) + "\n",
        encoding="utf-8")
    (out / "sft_val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_recs) + "\n",
        encoding="utf-8")
    (out / "swift_train.jsonl").write_text(
        "\n".join(json.dumps(to_swift(r), ensure_ascii=False) for r in train_recs) + "\n",
        encoding="utf-8")
    (out / "swift_val.jsonl").write_text(
        "\n".join(json.dumps(to_swift(r), ensure_ascii=False) for r in val_recs) + "\n",
        encoding="utf-8")
    (out / "dataset_stats.json").write_text(json.dumps({
        "n_classes": len(classes),
        "per_class": {k: len(v["images"]) for k, v in classes.items()},
        "train_images": len(train_pairs), "val_images": len(val_pairs),
        "train_records": len(train_recs), "val_records": len(val_recs),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出 -> {out.resolve()}")


if __name__ == "__main__":
    main()
