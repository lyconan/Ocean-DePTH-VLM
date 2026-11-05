#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_consolidate_species_kb.py — 全量物种知识库汇总
合并: DeepSeek采集结果(results jsonl) + 中文名(zh_names.jsonl) + 图集元数据(tasks jsonl)
输出: consolidated_species_kb.jsonl (每行一个物种完整档案) + consolidated_summary.csv (Excel可开)
可随时运行（部分数据亦可），采集完成后重跑即全量。
"""
import argparse
import csv
import json
import os

FIELDS = ["mid", "latin", "latin_full", "rank", "rank_cn", "phylum",
          "zh_atlas", "zh_gbif", "zh_worms", "zh_best",
          "ok", "total_chars", "n_sections", "split_ok", "n_citations",
          "citations", "collected", "error",
          "sec_分类地位", "sec_形态特征", "sec_分布与栖息深度", "sec_生态角色", "sec_命名与分类历史"]


def pick_zh(t, z):
    cands = []
    if z:
        cands += (z.get("zh_atlas") or "").split("/")
        cands += z.get("zh_gbif") or []
        cands += z.get("zh_worms") or []
    cands = [c.strip() for c in cands if c and c.strip()]
    return "/".join(dict.fromkeys(cands)) if cands else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="full_2641 目录")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    d = args.dir
    results_path = args.results or os.path.join(d, "batch_results.jsonl")
    tasks = {t["mid"]: t for t in (json.loads(l) for l in open(os.path.join(d, "all_2641_tasks.jsonl")) if l.strip())}
    zh = {}
    zh_path = os.path.join(d, "zh_names.jsonl")
    if os.path.exists(zh_path):
        for l in open(zh_path):
            try:
                r = json.loads(l)
                zh[r["mid"]] = r
            except Exception:
                pass
    recs = {}
    if os.path.exists(results_path):
        for l in open(results_path):
            try:
                r = json.loads(l)
                recs[r["mid"]] = r  # 后写的覆盖先写的
            except Exception:
                pass
    out_rows = []
    for mid, t in tasks.items():
        r = recs.get(mid, {})
        z = zh.get(mid, {})
        secs = {s.get("title", ""): s.get("text", "") for s in (r.get("sections") or [])}
        # 全文模式下也填充 sec_*: 全文放分类地位列便于检索? 不——保持空, 用 full_text 单列
        row = {
            "mid": mid, "latin": t["latin"], "latin_full": t["latin_full"],
            "rank": t["rank"], "rank_cn": t.get("rank_cn", ""), "phylum": t["phylum"],
            "zh_atlas": z.get("zh_atlas", ""), "zh_gbif": "/".join(z.get("zh_gbif") or []),
            "zh_worms": "/".join(z.get("zh_worms") or []),
            "zh_best": pick_zh(t, z),
            "ok": r.get("ok", False),
            "total_chars": r.get("total_chars", 0),
            "n_sections": len(r.get("sections") or []),
            "split_ok": r.get("split_ok", False),
            "n_citations": len(r.get("citations") or []),
            "citations": ";".join(r.get("citations") or []),
            "collected": r.get("collected", ""),
            "error": r.get("error", "") or "",
        }
        for std in ["分类地位", "形态特征", "分布与栖息深度", "生态角色", "命名与分类历史"]:
            row[f"sec_{std}"] = secs.get(std, "") if r.get("split_ok") else ""
        if not r.get("split_ok") and r.get("ok"):
            row["full_text"] = (secs.get("全文") or "").replace("\n", "\\n")
        else:
            row["full_text"] = ""
        out_rows.append(row)
    out_rows.sort(key=lambda x: x["mid"])
    jp = os.path.join(d, "consolidated_species_kb.jsonl")
    with open(jp, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cp = os.path.join(d, "consolidated_summary.csv")
    with open(cp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["full_text"], extrasaction="ignore")
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    ok_n = sum(1 for r in out_rows if r["ok"])
    zh_n = sum(1 for r in out_rows if r["zh_best"])
    print(f"consolidated: {len(out_rows)} rows | ok {ok_n} | zh_name {zh_n} | -> {jp} , {cp}")


if __name__ == "__main__":
    main()
