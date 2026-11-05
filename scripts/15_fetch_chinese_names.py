#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_fetch_chinese_names.py — 为全量形态种抓取中文名（三源汇总）
源1: 图集自带 name_cn (atlas)
源2: GBIF species/match(verbose=true 扫描主结果+alternatives) -> /species/{key}/vernacularNames 过滤 zh*
源3: WoRMS AphiaRecordsByName -> AphiaVernacularsByID 过滤 Chinese
输出: --out JSONL 追加式断点续跑 {"mid","zh_atlas","zh_gbif":[],"zh_worms":[]}
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import urllib.parse

UA = {"User-Agent": "OceanDePTH-KB/1.0 (research; contact: local)"}
ZH_PREFIX = ("zh",)


def http_json(url, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def gbif_zh(latin):
    try:
        u = ("https://api.gbif.org/v1/species/match?verbose=true&name="
             + urllib.parse.quote(latin))
        m = http_json(u)
    except Exception:
        return []
    cands = []
    if isinstance(m, dict):
        if m.get("matchType") not in ("HIGHERRANK", "NONE") and m.get("usageKey"):
            cands.append(m)
        for alt in m.get("alternatives") or []:
            if alt.get("usageKey"):
                cands.append(alt)
    out = []
    for c in cands[:3]:
        try:
            v = http_json(f"https://api.gbif.org/v1/species/{c['usageKey']}/vernacularNames?limit=100")
        except Exception:
            continue
        for it in v.get("results") or []:
            lang = (it.get("language") or "").lower()
            if lang.startswith(ZH_PREFIX):
                n = (it.get("vernacularName") or "").strip()
                if n and n not in out and re.search(r"[\u4e00-\u9fff]", n):
                    out.append(n)
        if out:
            break
    return out[:10]


def worms_zh(latin):
    try:
        u = ("https://www.marinespecies.org/rest/AphiaRecordsByName/"
             + urllib.parse.quote(latin) + "?marine_only=false")
        recs = http_json(u)
    except Exception:
        return []
    if not recs:
        return []
    aid = recs[0].get("valid_AphiaID") or recs[0].get("AphiaID")
    if not aid:
        return []
    try:
        verns = http_json(f"https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/{aid}")
    except Exception:
        return []
    out = []
    for v in verns or []:
        lang = (v.get("language_code") or v.get("language") or "").lower()
        if lang.startswith(ZH_PREFIX) or lang in ("chinese", "zho", "chi"):
            n = (v.get("vernacular") or "").strip()
            if n and n not in out and re.search(r"[\u4e00-\u9fff]", n):
                out.append(n)
    return out[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    tasks = [json.loads(l) for l in open(args.tasks) if l.strip()]
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                r = json.loads(l)
                done[r["mid"]] = r
            except Exception:
                pass
    todo = [t for t in tasks if t["mid"] not in done]
    print(f"total {len(tasks)}, done {len(done)}, todo {len(todo)}", flush=True)

    def work(t):
        g = gbif_zh(t["query_name"])
        time.sleep(0.2)
        w = worms_zh(t["query_name"])
        return {"mid": t["mid"], "latin": t["latin"], "zh_atlas": t.get("name_cn_atlas", ""),
                "zh_gbif": g, "zh_worms": w}

    cnt = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(work, todo):
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush(); os.fsync(f.fileno())
            cnt += 1
            if cnt % 200 == 0:
                print(f"  {cnt}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
