#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶元骨干树构建: 补全 14 个门下的完整 WoRMS 分类树 (门→纲→目→科)

背景: morphospecies_index.json 的阶元链只沿图集观察谱系生长, 各门下的
      兄弟分支(纲/目/科)缺失。本脚本补全完整骨干树。

技术路线:
  - WoRMS REST 的 AphiaChildrenByID/AphiaClassificationByID 已下线(404),
    无法直接枚举子节点
  - 改用 GBIF backbone /species/{key}/children 枚举树结构 (稳定可分页)
  - 每个节点再用 WoRMS AphiaRecordsByName/{name} 回填 AphiaID/接受名/状态
    (名称查询端点仍可用; 404/204/HTML 空页 = 未命中)
  - GBIF 与 WoRMS 概念冲突时以 WoRMS 为准: GBIF 拆分名(如 Scleralcyonacea)
    查 WoRMS 得到未接受状态 -> 合并回 WoRMS 接受名 (Alcyonacea), 与图集
    传统系统一致
  - 全程请求缓存 (data/processed/taxon_backbone/cache.json), 可断点续跑

输出: data/processed/taxon_backbone/taxon_backbone.json
  nodes: {node_id: {name, rank, worms_aphia_id, worms_status, valid_name,
                    is_marine, gbif_keys, aliases, phylum}}
  edges: [{parent, child}]
"""
import os
import re
import json
import time
import threading
import urllib.request
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data/processed/taxon_backbone")
OUT_JSON = os.path.join(OUT_DIR, "taxon_backbone.json")
CACHE_JSON = os.path.join(OUT_DIR, "cache.json")
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")

GBIF = "https://api.gbif.org/v1"
WORMS = "https://www.marinespecies.org/rest"
UA = {"User-Agent": "Ocean-DePTH-VLM/2.0 (taxon backbone; academic)"}

# 下探停止阶元: 属级及以下不收 (属/种只保留图集已有的, 且规避 GBIF
# 骨干树数万化石属的膨胀)
STOP_RANKS = {"genus", "subgenus", "species", "subspecies", "tribe",
              "subtribe", "variety", "form"}
# 收进骨干树的正式阶元白名单 (门→纲→目→科 完整树形结构)
KEEP_RANKS = {"phylum", "subphylum", "infraphylum", "superclass", "class",
              "subclass", "infraclass", "superorder", "order", "suborder",
              "infraorder", "superfamily", "family", "subfamily"}
MAX_DEPTH = 9  # unranked 等透明中间层消耗深度, 放宽以防误截

WORKERS_GBIF = 10
WORKERS_WORMS = 12  # WoRMS 单请求延迟 ~4s, worker 数决定实际吞吐 (_pace 仍限 4 req/s)


def load_cache():
    if os.path.exists(CACHE_JSON):
        try:
            with open(CACHE_JSON) as f:
                c = json.load(f)
            c.setdefault("gbif_children", {})
            c.setdefault("worms_name", {})
            c.setdefault("worms_by_id", {})
            return c
        except Exception:
            pass
    return {"gbif_children": {}, "worms_name": {}, "worms_by_id": {}}


CACHE = load_cache()
_cache_dirty = 0
_lock = threading.Lock()


def save_cache(force=False):
    global _cache_dirty
    with _lock:  # 写线程也在锁内改 CACHE, 防止 dump 期间字典并发修改
        if not force and _cache_dirty < 200:
            return
        tmp = CACHE_JSON + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(CACHE, f, ensure_ascii=False)
        os.replace(tmp, CACHE_JSON)
        _cache_dirty = 0


def http_json(url, tries=6):
    """返回 (status, data): ok / not_found / fail。fail 不可缓存"""
    host = GBIF if "gbif" in url else WORMS
    for i in range(tries):
        _pace(host)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=60) as r:
                return "ok", json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 204):
                return "not_found", None
            if e.code in (429, 502, 503, 504):
                time.sleep(min(60, (2, 5, 10, 20, 40, 60)[min(i, 5)]))
                continue
            return "fail", None
        except Exception:
            # 200 + HTML 空页等 JSON 解析失败 (WoRMS 限流常见) -> 可重试
            time.sleep(min(30, 3 * (i + 1)))
    return "fail", None


_pace_lock = threading.Lock()
_pace_last = {}


def _pace(host, min_interval=0.25):
    """全局限速: 每主机最小请求间隔 0.25s (~4 req/s)"""
    with _pace_lock:
        last = _pace_last.get(host, 0.0)
        now = time.time()
        wait = last + min_interval - now
        if wait > 0:
            _pace_last[host] = now + wait
        else:
            _pace_last[host] = now
    if wait > 0:
        time.sleep(wait)


def gbif_children(key):
    """GBIF 子节点枚举 (全分页)。仅在全部页请求成功时缓存 (防限流污染)"""
    ck = f"gc:{key}"
    if ck in CACHE["gbif_children"]:
        return CACHE["gbif_children"][ck]
    out, off, all_ok = [], 0, True
    while True:
        st, r = http_json(f"{GBIF}/species/{key}/children?limit=1000&offset={off}")
        if st != "ok":
            all_ok = False
            break
        results = r.get("results") or []
        out += results
        off += len(results)
        if r.get("endOfRecords") or not results:
            break
    slim = [{"key": c["key"], "name": c.get("scientificName") or "",
             "rank": (c.get("rank") or "").lower()}
            for c in out if c.get("key")]
    if all_ok:
        global _cache_dirty
        with _lock:
            CACHE["gbif_children"][ck] = slim
            _cache_dirty += 1
        save_cache()
    return slim


def worms_by_name(name):
    """WoRMS 名称查询 (缓存)。返回标准化记录或 None。仅 ok/not_found 缓存"""
    ck = name.strip().lower()
    if ck in CACHE["worms_name"]:
        return CACHE["worms_name"][ck]
    st, recs = http_json(f"{WORMS}/AphiaRecordsByName/{urllib.request.quote(name)}?marine_only=false")
    rec = None
    if st == "ok" and isinstance(recs, list) and recs:
        # 优先接受名, 否则第一条
        first = next((x for x in recs if x.get("status") == "accepted"), recs[0])
        rec = {"aphia_id": first.get("AphiaID"), "name": first.get("scientificname"),
               "status": first.get("status"), "rank": (first.get("rank") or "").lower(),
               "valid_aphia_id": first.get("valid_AphiaID"),
               "valid_name": first.get("valid_name"),
               "parent_aphia_id": first.get("parentNameUsageID"),
               "is_marine": first.get("isMarine"), "is_brackish": first.get("isBrackish"),
               "is_freshwater": first.get("isFreshwater"), "is_terrestrial": first.get("isTerrestrial")}
    elif st == "fail":
        return None  # 限流/解析失败: 不缓存, 下次重跑自动重试
    global _cache_dirty
    with _lock:
        CACHE["worms_name"][ck] = rec
        _cache_dirty += 1
    save_cache()
    return rec


def worms_by_ids(aids):
    """WoRMS 批量 ID 查询 (缓存)。返回 {aphia_id: 标准化记录}。仅 ok 缓存, fail 重试"""
    out = {}
    missing = []
    for a in aids:
        ck = str(a)
        if ck in CACHE["worms_by_id"]:
            if CACHE["worms_by_id"][ck] is not None:
                out[a] = CACHE["worms_by_id"][ck]
        else:
            missing.append(a)
    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        q = "&".join(f"aphiaids%5B%5D={a}" for a in batch)
        st, recs = http_json(f"{WORMS}/AphiaRecordsByAphiaIDs?{q}")
        if st != "ok" or not isinstance(recs, list):
            continue  # fail: 不缓存, 下次重跑自动重试
        got = {x.get("AphiaID"): x for x in recs if x.get("AphiaID")}
        for a in batch:
            first = got.get(a)
            rec = None
            if first:
                rec = {"aphia_id": first.get("AphiaID"), "name": first.get("scientificname"),
                       "status": first.get("status"), "rank": (first.get("rank") or "").lower(),
                       "valid_aphia_id": first.get("valid_AphiaID"),
                       "valid_name": first.get("valid_name"),
                       "parent_aphia_id": first.get("parentNameUsageID"),
                       "is_marine": first.get("isMarine"), "is_brackish": first.get("isBrackish"),
                       "is_freshwater": first.get("isFreshwater"),
                       "is_terrestrial": first.get("isTerrestrial")}
            global _cache_dirty
            with _lock:
                CACHE["worms_by_id"][str(a)] = rec
                _cache_dirty += 1
            if rec:
                out[a] = rec
    return out


def norm_rank(r):
    return (r or "").lower().replace("_", " ").strip()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    idx = json.load(open(MSP_JSON))["morphospecies"]
    phyla = {}
    for m in idx:
        ph = (m.get("taxonomy_worms") or {}).get("phylum")
        if ph and ph.get("name"):
            phyla.setdefault(ph["name"], ph.get("aphia_id"))

    print(f"根: {len(phyla)} 门", flush=True)

    # 1) GBIF 根 key: verbose match 扫描主结果+alternatives
    #    接受条件: rank=PHYLUM 且返回名与查询名完全一致 (大小写不敏感)
    #    - Arthropoda 主结果 matchType=HIGHERRANK 但 usageKey 正确 -> 接受
    #    - Chaetognatha/Ctenophora 主结果是动物界/NONE, 正确节点在 alternatives
    #    - 同名异属 (如 Ctenophora Meigen, 1803 昆虫属) 靠 rank=PHYLUM + 精确名排除
    #    任一门匹配失败则中止 (静默跳过会产出不完整树)
    roots = []
    failed = []
    for name in sorted(phyla):
        nid = None
        st, r = http_json(f"{GBIF}/species/match?name={urllib.request.quote(name)}"
                          f"&strict=true&verbose=true")
        cands = []
        if st == "ok" and isinstance(r, dict):
            cands = [r] + list(r.get("alternatives") or [])
        for c in cands:
            if c.get("usageKey") and (c.get("rank") or "").upper() == "PHYLUM" and \
                    (c.get("scientificName") or "").strip().lower() == name.lower():
                nid = c["usageKey"]
                break
        if nid:
            roots.append((name, nid))
            mt = next((c.get("matchType") for c in cands
                       if c.get("usageKey") == nid), "?")
            print(f"  {name}: GBIF key={nid} ({mt})", flush=True)
        else:
            failed.append(name)
            print(f"  {name}: GBIF 匹配失败", flush=True)
    if failed:
        raise SystemExit(f"根匹配失败: {failed} — 骨干树不完整, 请先解决再跑")

    # 2) BFS 枚举树结构 (GBIF, 前沿并行 + WoRMS 海洋剪枝)
    gbif_nodes = {}   # key -> {name, rank, parent_key, phylum}
    edges = []        # (parent_key, child_key)
    pruned = 0

    # 显式排除: 纯陆生/淡水纲 (WoRMS 的 isMarine 只表示"含海洋成员",
    # Insecta/Arachnida 因海黾海螨等少数海洋种而 isMarine=1, 标志位无法剪枝)
    EXCLUDE_CLASSES = {
        "insecta", "arachnida", "entognatha", "chilopoda", "diplopoda",
        "symphyla", "pauropoda", "onychophora", "amphibia", "reptilia",
        "aves", "mammalia", "branchiopoda"}

    def marine_ok(name, rank):
        """非海洋/不在 WoRMS 的枝条剪掉。只检查纲级节点 (EXCLUDE 为纲名),
        更深层继承其祖先的海洋属性, 避免数千次 WoRMS 逐节点查询"""
        if rank != "class":
            return True
        if name.strip().lower() in EXCLUDE_CLASSES:
            return False
        w = worms_by_name(name)
        if w is None:
            # 缓存中确认为 None = WoRMS 真无此名 -> 剪枝;
            # 查询临时失败不会写入缓存 -> 保留待查
            ck = name.strip().lower()
            return not (ck in CACHE["worms_name"] and CACHE["worms_name"][ck] is None)
        return True

    seen = set()  # 已处理过的 GBIF key (含透明中间节点)
    with ThreadPoolExecutor(max_workers=WORKERS_GBIF) as exg, \
            ThreadPoolExecutor(max_workers=WORKERS_WORMS) as exw:
        for ph, root_key in roots:
            gbif_nodes[root_key] = {"name": ph, "rank": "phylum", "parent_key": None,
                                    "phylum": ph}
            seen.add(root_key)
            # frontier: (key, name, rank, depth, eff_parent)
            # eff_parent = 最近保留阶元祖先; unranked 等中间层透明穿越
            frontier = [(root_key, ph, "phylum", 0, root_key)]
            while frontier:
                # 剪枝检查: 仅纲级节点查 WoRMS (透明中间节点不入树, 不查)
                keep = [f for f in frontier if f[2] in KEEP_RANKS]
                keep = [f for f in keep if marine_ok(f[1], f[2])]
                dropped = {f[0] for f in frontier if f[2] in KEEP_RANKS} - \
                          {f[0] for f in keep}
                pruned += len(dropped)
                frontier = keep + [f for f in frontier if f[2] not in KEEP_RANKS]
                if not frontier:
                    break
                alive_keys = [f[0] for f in frontier]
                children_map = dict(zip(alive_keys, exg.map(gbif_children, alive_keys)))
                nxt = []
                for (pkey, _nm, _rk, depth, effp), kids in zip(frontier, children_map.values()):
                    for c in kids:
                        if c["name"].upper().startswith(("BOLD:", "GBIF:")):
                            continue  # GBIF 骨干树混入的 BOLD BIN / 占位节点, 无 WoRMS 对应物
                        crk = norm_rank(c["rank"])
                        if c["key"] in seen:
                            continue
                        seen.add(c["key"])
                        if crk in KEEP_RANKS:
                            gbif_nodes[c["key"]] = {"name": c["name"], "rank": crk,
                                                    "parent_key": effp, "phylum": ph}
                            edges.append((effp, c["key"]))
                            # 科/亚科不再展开: 其子级全是属(STOP), 展开纯属浪费查询
                            if depth + 1 < MAX_DEPTH and crk not in ("family", "subfamily"):
                                nxt.append((c["key"], c["name"], crk, depth + 1, c["key"]))
                        elif crk in STOP_RANKS:
                            continue  # 属/种级不收
                        else:
                            # unranked/division 等中间层: 不入树, 子节点上挂到 effp
                            if depth + 1 < MAX_DEPTH:
                                nxt.append((c["key"], c["name"], crk, depth + 1, effp))
                frontier = nxt
            n_rank = sum(1 for v in gbif_nodes.values() if v["phylum"] == ph)
            print(f"  枚举 {ph}: {n_rank} 节点 (累计剪枝 {pruned} 枝)", flush=True)
    print(f"GBIF 枚举完成: {len(gbif_nodes)} 节点 {len(edges)} 边, 剪枝 {pruned}, {time.time()-t0:.0f}s", flush=True)
    save_cache(force=True)

    # 3) WoRMS 回填 (名称查询, 并行), 未接受名合并到接受概念
    node_of_aphia = {}     # aphia_id -> node_id
    nodes = {}             # node_id -> 属性 dict
    key2node = {}          # gbif key -> node_id
    stats = defaultdict(int)

    def backfill(key):
        g = gbif_nodes[key]
        w = worms_by_name(g["name"]) if g["name"] else None
        return key, g, w

    sorted_keys = sorted(gbif_nodes)
    with ThreadPoolExecutor(max_workers=WORKERS_WORMS) as ex:
        for i, (key, g, w) in enumerate(ex.map(backfill, sorted_keys)):
            if w and w.get("valid_aphia_id"):
                nid = f"aphia:{w['valid_aphia_id']}"
            elif w and w.get("aphia_id"):
                nid = f"aphia:{w['aphia_id']}"
            else:
                nid = f"gbif:{key}"
                stats["gbif_only"] += 1
            if nid not in nodes:
                nodes[nid] = {
                    "node_id": nid,
                    "name": (w.get("valid_name") if w and w.get("valid_name") else
                             (w.get("name") if w else g["name"])),
                    "rank": w["rank"] if w and w.get("rank") else g["rank"],
                    "worms_aphia_id": w.get("valid_aphia_id") or w.get("aphia_id") if w else None,
                    "worms_status": w.get("status") if w else None,
                    "parent_aphia_id": w.get("parent_aphia_id") if w else None,
                    "is_marine": w.get("is_marine") if w else None,
                    "is_terrestrial": w.get("is_terrestrial") if w else None,
                    "is_freshwater": w.get("is_freshwater") if w else None,
                    "gbif_keys": [key],
                    "aliases": [g["name"]] if (w and w.get("valid_name") and
                                               w["valid_name"].lower() != g["name"].lower()) else [],
                    "phylum": g["phylum"],
                }
                if w:
                    node_of_aphia[w.get("valid_aphia_id") or w.get("aphia_id")] = nid
            else:
                nodes[nid]["gbif_keys"].append(key)
                if (w and w.get("valid_name") and
                        w["valid_name"].lower() != nodes[nid]["name"].lower()):
                    nodes[nid]["aliases"].append(g["name"])
            key2node[key] = nid
            if i % 500 == 0 and i:
                print(f"  WoRMS 回填 {i}/{len(sorted_keys)}", flush=True)
                save_cache(force=True)
    save_cache(force=True)

    # 4) 边重定向到合并后节点 (去重, 防自环)
    final_edges = set()
    for pk, ck in edges:
        p, c = key2node.get(pk), key2node.get(ck)
        if p and c and p != c:
            final_edges.add((p, c))
    # 5) WoRMS parent 递归补边: 沿 parent_aphia_id 向上, 缺失的中间节点
    #    (超科/下目等 GBIF 骨干缺位的) 以 WoRMS 记录建桥节点并入树,
    #    直到接上树内祖先; 查不到的保留断链, 下次重跑自动重试
    stats["worms_bridge_nodes"] = 0
    todo = [(nid, n.get("parent_aphia_id"), 0)
            for nid, n in nodes.items() if n.get("parent_aphia_id")]
    while todo:
        nid, wpid, dep = todo.pop()
        if wpid in node_of_aphia:
            if node_of_aphia[wpid] != nid:
                final_edges.add((node_of_aphia[wpid], nid))
            continue
        if dep >= 12:  # 防异常环: 到根 (Animalia) 理论不过 8 级
            continue
        r = worms_by_ids([wpid]).get(wpid)
        if not r or not r.get("valid_aphia_id"):
            continue  # 查不到: 断链保留, 下次重跑重试
        pid = f"aphia:{r['valid_aphia_id']}"
        if pid == nid:
            continue
        if pid not in nodes:
            nodes[pid] = {
                "node_id": pid,
                "name": r.get("valid_name") or r.get("name"),
                "rank": r.get("rank"),
                "worms_aphia_id": r.get("valid_aphia_id") or r.get("aphia_id"),
                "worms_status": r.get("status"),
                "parent_aphia_id": r.get("parent_aphia_id"),
                "is_marine": r.get("is_marine"), "is_terrestrial": r.get("is_terrestrial"),
                "is_freshwater": r.get("is_freshwater"),
                "gbif_keys": [], "aliases": [], "phylum": "(bridge)",
            }
            node_of_aphia[r["valid_aphia_id"]] = pid
            stats["worms_bridge_nodes"] += 1
            todo.append((pid, r.get("parent_aphia_id"), dep + 1))
        final_edges.add((pid, nid))

    # 6) 统计
    rank_cnt = defaultdict(int)
    for n in nodes.values():
        rank_cnt[n["rank"]] += 1
    phylum_cnt = defaultdict(lambda: defaultdict(int))
    for n in nodes.values():
        if n["rank"] in ("phylum", "kingdom"):
            continue
        phylum_cnt[n["phylum"]][n["rank"]] += 1
    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "GBIF backbone (structure) + WoRMS AphiaRecordsByName (concept)",
        "params": {"stop_ranks": sorted(STOP_RANKS), "max_depth": MAX_DEPTH,
                   "phyla": phyla},
        "nodes": nodes,
        "edges": sorted(final_edges),
        "statistics": {
            "nodes": len(nodes), "edges": len(final_edges),
            "ranks": dict(rank_cnt),
            "gbif_only_nodes": stats["gbif_only"],
            "worms_bridge_nodes": stats["worms_bridge_nodes"],
            "per_phylum": {p: dict(d) for p, d in phylum_cnt.items()},
        },
    }
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT_JSON)

    print("\n==== 骨干树完成 ====", flush=True)
    print(f"节点 {len(nodes)}  边 {len(final_edges)}  GBIF-only {stats['gbif_only']}", flush=True)
    for rk, n in sorted(rank_cnt.items(), key=lambda x: -x[1]):
        print(f"  {rk:14} {n}")
    for p, d in phylum_cnt.items():
        tot = sum(d.values())
        print(f"  {p:16} {tot} 节点  {json.dumps(d, ensure_ascii=False)}", flush=True)
    print(f"输出: {OUT_JSON}", flush=True)
    print(f"用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
