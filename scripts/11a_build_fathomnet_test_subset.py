#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11a_build_fathomnet_test_subset.py — 从 FathomNet Val 测试集构建形态种级验证子集

需求 (2026-09-21):
  - 验证整个系统的物种识别能力 (YOLO34 -> BioCLIP2.5 kNN -> VLM)
  - 只覆盖 FathomNet 来源的 morphospecies 类别 (1,481 个)
  - 纯度: 图内所有 bbox 物种 concept 都必须能映射到 2,641 库内类别
    (器械/库外物种图自动剔除)
  - 每个类别最多 5 张图
  - 测试集原为粗类别 (34纲), 本子集带精细 concept 级标注

流程:
  1. morphospecies_index -> latin(小写归一) -> msp_id 映射 (全部 2,641, 图内物种可属任意来源)
  2. 多线程扫描 Val/*.xml, 正则提取 <name>concepts</name>
  3. 纯度过滤: 图内全部 concept 可映射
  4. round-robin 每类最多 5 张分配
  5. 输出 JSON (图路径 + concept->msp_id + 统计)

用法: python3 scripts/11a_build_fathomnet_test_subset.py
"""
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")
VAL_DIR = "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet/Val"
OUT_JSON = os.path.join(BASE, "data/processed/eval_subset/fathomnet_test_subset.json")
MAX_PER_CLASS = 5
NAME_RE = re.compile(r"<name>([^<]+)</name>")

# FathomNet 来源标记 (entries[].source 含这些子串的条目才算 FathomNet 来源类别)
FN_MARKS = ("FathomNet",)


def norm(s):
    return " ".join(s.split()).lower()


def main():
    # 1) latin -> msp_id 全量索引
    idx = json.load(open(MSP_JSON, encoding="utf-8"))
    msp = idx["morphospecies"]
    msp = list(msp.values()) if isinstance(msp, dict) else msp
    latin2id = {}
    fn_ids = set()
    for m in msp:
        latin2id.setdefault(norm(m.get("latin") or ""), m["morphospecies_id"])
        srcs = " ".join(e.get("source", "") for e in m.get("entries", [])
                        if isinstance(e, dict))
        if any(k in srcs for k in FN_MARKS):
            fn_ids.add(m["morphospecies_id"])
    print(f"图谱条目 {len(msp)}, latin 索引 {len(latin2id)}, FathomNet 来源类别 {len(fn_ids)}")

    # 2) 扫描 Val XML
    xmls = [f for f in os.listdir(VAL_DIR) if f.endswith(".xml")]
    print(f"Val XML: {len(xmls)}")

    def parse(fn):
        try:
            txt = open(os.path.join(VAL_DIR, fn), encoding="utf-8", errors="ignore").read()
            return fn[:-4], list(dict.fromkeys(NAME_RE.findall(txt)))
        except OSError:
            return fn[:-4], []

    images = {}
    with ThreadPoolExecutor(32) as ex:
        for uuid, concepts in ex.map(parse, xmls, chunksize=64):
            if concepts:
                images[uuid] = concepts
    print(f"有标注图: {len(images)}")

    # 3) 纯度过滤 + concept 映射
    pure = {}       # uuid -> {concept: msp_id}
    unmapped_cnt = 0
    for uuid, concepts in images.items():
        mp = {}
        ok = True
        for c in concepts:
            mid = latin2id.get(norm(c))
            if mid is None:
                ok = False
                break
            mp[c] = mid
        if ok:
            pure[uuid] = mp
        else:
            unmapped_cnt += 1
    print(f"纯图 (全部concept可映射): {len(pure)}, 含库外概念剔除: {unmapped_cnt}")

    # 4) 配额分配: 只统计 FathomNet 来源类别; round-robin 每类 <=5
    used_images = set()
    per_class = defaultdict(int)
    # 类 -> 含该类的纯图列表 (图内 concept 越少越干净优先)
    cls_imgs = defaultdict(list)
    for uuid, mp in pure.items():
        fn_hit = [mid for mid in mp.values() if mid in fn_ids]
        if not fn_hit:
            continue
        cls_imgs[None].append(uuid)  # 占位防空
        for mid in fn_hit:
            cls_imgs[mid].append((len(mp), uuid))
    covered = [c for c in cls_imgs if c is not None and cls_imgs[c]]
    print(f"FathomNet 来源类别有纯图覆盖: {len(covered)}")

    # 贪心: 每类按图内 concept 数升序取, 图可服务多类
    selected = {}   # uuid -> {concept: msp_id}
    for mid in covered:
        for _, uuid in sorted(cls_imgs[mid])[:MAX_PER_CLASS]:
            if per_class[mid] >= MAX_PER_CLASS:
                break
            selected.setdefault(uuid, pure[uuid])
            per_class[mid] += 1
    print(f"入选图: {len(selected)}, 覆盖类别: {len([c for c,v in per_class.items() if v>0])}")

    # 5) 输出
    records = []
    for uuid, mp in sorted(selected.items()):
        rel = f"Val/{uuid}.png"
        records.append({
            "image": rel,
            "png_exists": os.path.exists(os.path.join(VAL_DIR, uuid + ".png")),
            "concepts": mp,
            "n_objects": len(mp),
        })
    stats = {
        "per_class": {k: v for k, v in sorted(per_class.items()) if v},
        "n_images": len(records),
        "n_classes_covered": len([v for v in per_class.values() if v]),
        "n_fn_classes": len(fn_ids),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"stats": stats, "records": records}, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"已写出: {OUT_JSON}")
    print(f"图片存在性检查: {sum(r['png_exists'] for r in records)}/{len(records)}")


if __name__ == "__main__":
    main()
