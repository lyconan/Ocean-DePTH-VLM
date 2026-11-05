#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FathomNet 深海形态种扩充: 本地 FathomNet 数据集 (MBARI 等 ROV 原位影像) +
官方 API 逐图深度, 把 depth>=500m 的类别扩充进形态种清单 (FNM_ 前缀)。

数据源:
  - 本地: $FATHOMNET_ROOT (环境变量, 默认指向源工程 FathomNet 数据目录)
      class_mapping.json   2,553 类 (双名种名 / 'sp.' 形态种 / 属·科级 / 器械)
      Train/*.png (132,944) + Val/*.png (~33,236), 文件名即 UUID
  - API: https://database.fathomnet.org/api
      GET /images/query/concept/<name> -> 逐图 depthMeters (98% 覆盖)

三阶段 (断点续传, 各自独立可重跑):
  python3 scripts/expand_fathomnet.py enumerate  # 类名解析+WoRMS校验(14门)
  python3 scripts/expand_fathomnet.py query      # API 逐图深度 -> manifest
  python3 scripts/expand_fathomnet.py merge      # 图片复制+入库 FNM_

过滤规则:
  - 类名分三类: 双名种名(kind=species) / 'sp.'形态种(kind=morpho, 按父阶元挂
    WoRMS 链, 与图集 'sp. inc.' 形态种同构) / 单名属·科·目级(kind=higher);
    器械/垃圾等非生物名不命中 WoRMS 自动排除
  - WoRMS 校验: AphiaRecordsByName (marine; 响应自带 kingdom..genus
    分类字段, 直接构链免二次查询), 门限定 14 门, higher/morpho 限定
    species..order 阶元
  - 深度: API depthMeters >= 500 (每类最多取 30 张, 优先本地已有图片)
  - 去重: 与现有形态种按 species 级学名精确匹配 (大小写不敏感)

输出:
  data/processed/fathomnet_expansion/
    fathomnet_candidates.json   (Phase1 候选)
    fathomnet_manifest.json     (Phase2 图片清单)
    species/<concept>/<uuid>.png (图片池, 本地复制+S3补漏)
    .worms_cache.json / .local_uuids.json
  并更新 data/processed/atlas_lookup/morphospecies_index.json (Phase3)
"""
import os
import sys
import re
import json
import shutil
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data/processed/fathomnet_expansion")
CAND_JSON = os.path.join(OUT_DIR, "fathomnet_candidates.json")
MANIFEST = os.path.join(OUT_DIR, "fathomnet_manifest.json")
POOL_ROOT = os.path.join(OUT_DIR, "species")
WORMS_CACHE = os.path.join(OUT_DIR, ".worms_cache.json")
LOCAL_IDX = os.path.join(OUT_DIR, ".local_uuids.json")
LOCK = os.path.join(OUT_DIR, ".lock")
MSP_JSON = os.path.join(
    BASE, "data/processed/atlas_lookup/morphospecies_index.json")
# 复用的既有缓存 (离线命中, 避免重打 WoRMS API)
DEEPSEA_WCACHE = os.path.join(
    BASE, "data/processed/deepsea_expansion/.worms_cache.json")
BACKBONE_CACHE = os.path.join(
    BASE, "data/processed/taxon_backbone/cache.json")

FN_ROOT = os.environ.get(
    "FATHOMNET_ROOT", "/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet")
FN_API = "https://database.fathomnet.org/api"
WORMS = "https://www.marinespecies.org/rest"
UA = {"User-Agent": "Ocean-DePTH-VLM/2.0 (fathomnet expansion; academic)"}

DEPTH_MIN = 500            # 深海阈值 (用户确认)
MAX_PER_SP = 30            # 每类最多入库图片 (与 GBIF 管线一致)
WORKERS = 8

# 14 门 (名称, 来自 taxon_backbone)
PHyla = {
    "Annelida", "Cnidaria", "Chordata", "Echinodermata", "Ctenophora",
    "Mollusca", "Bryozoa", "Arthropoda", "Chaetognatha", "Nemertea",
    "Hemichordata", "Porifera", "Platyhelminthes", "Brachiopoda"}

# higher/morpho 允许的阶元 (现有图集形态种主体为 genus/family/order 级)
RANK_KEEP = {"species", "genus", "subgenus", "family", "subfamily",
             "tribe", "order"}
RANK_CN = {"species": "种", "genus": "属", "subgenus": "亚属",
           "family": "科", "subfamily": "亚科", "tribe": "族",
           "order": "目", "class": "纲"}

# 'sp.' 形态种的标记词 (取首个标记前的部分为父阶元)
# 注意: 标记后必须是空格或结尾 (sp. 1 / sp. / cf. xxx / complex / type N)
MORPHO_MARK = re.compile(
    r'\s+(sp\.|spp\.|cf\.|aff\.|complex|type\s+\d+)(?=\s|$)', re.I)
BINOMIAL = re.compile(r'^[A-Z][a-z\-]+\s+[a-z\-]+$')
MONOMIAL = re.compile(r'^[A-Z][a-z\-]+$')

_last_call = {"worms": 0.0, "fn": 0.0}


def pace(host, interval):
    now = time.time()
    wait = _last_call[host] + interval - now
    if wait > 0:
        time.sleep(wait)
        now = time.time()
    _last_call[host] = now


def http_json(url, host="worms", interval=0.4, tries=4):
    for i in range(tries):
        pace(host, interval)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (404, 204):
                return {"_status": "not_found"}
            if i == tries - 1:
                return None
            time.sleep((2, 5, 10, 20)[min(i, 3)])
        except Exception:
            if i == tries - 1:
                return None
            time.sleep((2, 5, 10, 20)[min(i, 3)])
    return None


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


W_CACHE = {"name": {}, "chain": {}}


def preload_caches():
    """回读自身缓存 + 复用 GBIF 深海管线与骨干树的 WoRMS 缓存 (离线命中)"""
    own = load_json(WORMS_CACHE, None)
    if own:
        for k, v in (own.get("name") or {}).items():
            W_CACHE["name"][k] = v
        for k, v in (own.get("chain") or {}).items():
            W_CACHE["chain"][k] = v
    ds = load_json(DEEPSEA_WCACHE, None)
    if ds:
        for k, v in (ds.get("name") or {}).items():
            W_CACHE["name"].setdefault(k, v)
        for k, v in (ds.get("chain") or {}).items():
            W_CACHE["chain"].setdefault(k, v)
    bb = load_json(BACKBONE_CACHE, None)
    if bb:
        for k, v in (bb.get("worms_name") or {}).items():
            if v and v.get("is_marine") and k not in W_CACHE["name"]:
                W_CACHE["name"][k] = {
                    "aphia_id": v.get("aphia_id"),
                    "name": v.get("valid_name") or v.get("name"),
                    "status": v.get("status"),
                    "valid_aphia_id": v.get("valid_aphia_id"),
                    "valid_name": v.get("valid_name"),
                    "is_marine": v.get("is_marine")}


def save_worms_cache():
    atomic_json(WORMS_CACHE, W_CACHE)


def _chain_from_record(r):
    """从 WoRMS record 构建分类链 (AphiaRecordsByName 与
    AphiaRecordByAphiaID 的响应均含 kingdom..genus 分类字段,
    0 额外 API 调用; AphiaClassificationByID 端点 2026-09 起 404 下线)"""
    if not (isinstance(r, dict) and r.get("AphiaID")):
        return None
    chain = {}
    for rk in ("kingdom", "phylum", "class", "order", "family"):
        if r.get(rk):
            chain[rk] = {"name": r[rk], "aphia_id": None}
    if r.get("genus"):
        chain.setdefault("genus", {"name": r["genus"], "aphia_id": None})
    rank = (r.get("rank") or "").lower()
    if rank and rank not in chain:
        own = r.get("valid_name") or r.get("scientificname")
        if own:
            chain[rank] = {
                "name": own,
                "aphia_id": (r.get("valid_AphiaID") or r.get("AphiaID")
                             if r.get("valid_name") else r.get("AphiaID"))}
    return chain


def worms_lookup(name):
    """WoRMS 精确名称查询 (缓存): aphia_id/status/valid_name
    命中 API 时同步从响应分类字段构建 chain 缓存 (每类仅 1 次 API)"""
    cache = W_CACHE["name"]
    ck = name.strip().lower()
    if ck in cache:
        return cache[ck]
    d = http_json(
        f"{WORMS}/AphiaRecordsByName/{urllib.parse.quote(name)}"
        f"?marine_only=true", host="worms", interval=0.4)
    rec = None
    if isinstance(d, list) and d:
        first = next((x for x in d if x.get("status") == "accepted"), d[0])
        rec = {
            "aphia_id": first.get("AphiaID"),
            "name": first.get("scientificname"),
            "status": first.get("status"),
            "valid_aphia_id": first.get("valid_AphiaID"),
            "valid_name": first.get("valid_name"),
            "is_marine": first.get("isMarine"),
        }
        chain = _chain_from_record(first)
        if chain is not None:
            for key in {str(first.get("AphiaID")),
                        str(first.get("valid_AphiaID"))} - {"None"}:
                W_CACHE["chain"].setdefault(key, chain)
    elif isinstance(d, dict) and d.get("_status") == "not_found":
        rec = None
    elif d is None:
        return None  # 限流/网络失败: 不缓存
    cache[ck] = rec
    if len(cache) % 50 == 0:
        save_worms_cache()
    return rec


def worms_chain(aphia_id):
    """WoRMS 分类链 (缓存): {rank_lower: {name, aphia_id}}
    lookup 命中时已同步构链; 此处仅处理链缓存未命中的 aphia_id
    (如历史 name 缓存条目), 走 AphiaRecordByAphiaID 单次查询"""
    ck = str(aphia_id)
    if ck in W_CACHE["chain"]:
        return W_CACHE["chain"][ck]
    r = http_json(f"{WORMS}/AphiaRecordByAphiaID/{aphia_id}",
                  host="worms", interval=0.4)
    chain = _chain_from_record(r)
    if chain is not None:
        W_CACHE["chain"][ck] = chain
        if len(W_CACHE["chain"]) % 50 == 0:
            save_worms_cache()
    return chain


# ---------------------------------------------------------------- 工具
def classify_concept(name):
    """类名 -> (kind, parent, suffix); kind: species/morpho/higher/skip"""
    n = name.strip()
    if not n or not re.match(r'^[A-Za-z][A-Za-z0-9 .\-()]*$', n):
        return ("skip", None, None)
    m = MORPHO_MARK.search(n)
    if m:
        parent = n[:m.start()].strip()
        if not parent:
            return ("skip", None, None)
        return ("morpho", parent, n[m.start():].strip())
    if BINOMIAL.match(n):
        return ("species", n, "")
    if MONOMIAL.match(n):
        return ("higher", n, "")
    return ("skip", None, None)


def load_existing():
    """现有形态种: species 级学名集合 + 全部 display_name"""
    with open(MSP_JSON) as f:
        msp = json.load(f)["morphospecies"]
    sp, displays = set(), set()
    for m in msp:
        lat = (m.get("latin") or "").strip().lower()
        if m.get("rank") == "species" and lat:
            sp.add(lat)
        if m.get("display_name"):
            displays.add(m["display_name"].strip().lower())
    return sp, displays


def safe_dirname(concept):
    d = concept.replace("/", "_").replace("\\", "_").strip().rstrip(".")
    return d or "unnamed"


# ---------------------------------------------------------------- Phase 1
def phase_enumerate():
    preload_caches()
    existing_sp, existing_disp = load_existing()
    print(f"现有形态种 species 级学名 {len(existing_sp)} 个", flush=True)

    with open(os.path.join(FN_ROOT, "class_mapping.json")) as f:
        cm = json.load(f)
    names = sorted(k.strip() for k in cm.keys() if k.strip())
    print(f"FathomNet 类别: {len(names)}", flush=True)

    cand = load_json(CAND_JSON, None)
    concepts = dict(cand["concepts"]) if cand else {}
    todo = [n for n in names if n not in concepts]
    print(f"待校验 {len(todo)} (缓存命中跳过 {len(names)-len(todo)})",
          flush=True)

    from collections import Counter
    kinds = Counter(classify_concept(n)[0] for n in todo)
    print(f"类名分布: {dict(kinds)}", flush=True)

    t0 = time.time()
    n_pass = 0
    for i, name in enumerate(todo, 1):
        kind, parent, suffix = classify_concept(name)
        ent = {"concept": name, "kind": kind, "pass": False,
               "reason": "", "worms_checked": True}
        try:
            if kind == "skip":
                ent["reason"] = "非生物学名格式"
            elif kind == "species" and name.lower() in existing_sp:
                ent["reason"] = "与现有形态种重复"
            else:
                lookup = name if kind != "morpho" else parent
                rec = worms_lookup(lookup)
                if not rec or not rec.get("aphia_id"):
                    ent["reason"] = "WoRMS 未命中"
                else:
                    chain = worms_chain(rec["aphia_id"]) or {}
                    phylum = (chain.get("phylum") or {}).get("name")
                    ent.update({
                        "latin": parent if kind == "morpho" else name,
                        "suffix": suffix or "",
                        "worms": rec, "taxonomy_worms": {
                            rk: dict(v) for rk, v in chain.items()
                            if rk in ("kingdom", "phylum", "class", "order",
                                      "family", "subfamily", "genus",
                                      "subgenus", "species")},
                        "phylum": phylum,
                    })
                    # 自身阶元: 从链尾附近找 rec.name 对应阶元
                    own_rank = None
                    for rk in ("species", "subgenus", "genus", "tribe",
                               "subfamily", "family", "order", "class",
                               "phylum"):
                        if chain.get(rk, {}).get("name", "").lower() == \
                                (rec.get("valid_name") or rec.get("name")
                                 or "").lower():
                            own_rank = rk
                            break
                    if kind == "species":
                        own_rank = "species"
                    if not phylum:
                        ent["reason"] = "WoRMS 链无门"
                    elif phylum not in PHyla:
                        ent["reason"] = f"门 {phylum} 不在 14 门"
                    elif own_rank and own_rank not in RANK_KEEP:
                        ent["reason"] = f"阶元 {own_rank} 过粗"
                    else:
                        ent["pass"] = True
                        ent["rank"] = own_rank or "species"
                        n_pass += 1
        finally:
            concepts[name] = ent
        if i % 50 == 0:
            print(f"    WoRMS {i}/{len(todo)} 通过 {n_pass} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            atomic_json(CAND_JSON, {"concepts": concepts})
    atomic_json(CAND_JSON, {"concepts": concepts})
    save_worms_cache()

    # 汇总
    all_c = list(concepts.values())
    passed = [c for c in all_c if c.get("pass")]
    print(f"\nPhase1 完成: 通过 {len(passed)}/{len(all_c)}", flush=True)
    print("通过类 kind 分布:", dict(Counter(c["kind"] for c in passed)))
    print("通过类门分布:", dict(Counter(c.get("phylum") for c in passed)))
    print("未通过原因:", dict(Counter(c.get("reason") for c in all_c
                                      if not c.get("pass")).most_common(10)))
    print(f"输出: {CAND_JSON}", flush=True)


# ---------------------------------------------------------------- Phase 2
def build_local_index():
    """扫描本地 Train/Val png 文件名 (UUID) -> 集合"""
    cached = load_json(LOCAL_IDX, None)
    if isinstance(cached, list):
        return set(cached)
    s = set()
    for sub in ("Train", "Val"):
        d = os.path.join(FN_ROOT, sub)
        with os.scandir(d) as it:
            for e in it:
                if e.name.endswith(".png") and e.is_file():
                    s.add(e.name[:-4])
    atomic_json(LOCAL_IDX, sorted(s))
    print(f"本地图片索引: {len(s)} 个 UUID (Train+Val)", flush=True)
    return s


def phase_query():
    preload_caches()
    cand = load_json(CAND_JSON, None)
    if not cand:
        sys.exit("先运行 enumerate")
    passed = {k: v for k, v in cand["concepts"].items() if v.get("pass")}
    print(f"候选 {len(passed)} 个类别", flush=True)

    local = build_local_index()
    mani = load_json(MANIFEST, {"concepts": {}})
    done = mani["concepts"]

    t0 = time.time()
    n_img = 0
    for i, (name, c) in enumerate(sorted(passed.items()), 1):
        if name in done and done[name].get("queried"):
            n_img += done[name].get("kept", 0)
            continue
        d = http_json(f"{FN_API}/images/query/concept/"
                      f"{urllib.parse.quote(name)}",
                      host="fn", interval=0.3)
        recs = d if isinstance(d, list) else []
        deep = [r for r in recs
                if r.get("depthMeters") is not None
                and float(r["depthMeters"]) >= DEPTH_MIN]
        # 优先本地已有图片, 其余记 S3 url 待 merge 下载; uuid 排序保证确定性
        deep.sort(key=lambda r: (0 if r["uuid"] in local else 1, r["uuid"]))
        kept = deep[:MAX_PER_SP]
        images = []
        for r in kept:
            images.append({
                "uuid": r["uuid"], "depth": r.get("depthMeters"),
                "lat": r.get("latitude"), "lon": r.get("longitude"),
                "url": r.get("url"),
            })
        entry = dict(c)
        entry.update({
            "queried": True, "api_count": len(recs),
            "with_depth": sum(1 for r in recs
                              if r.get("depthMeters") is not None),
            "deep_count": len(deep), "kept": len(kept), "images": images,
        })
        done[name] = entry
        n_img += len(kept)
        if i % 25 == 0:
            print(f"    API {i}/{len(passed)} 深海图累计 {n_img} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            atomic_json(MANIFEST, mani)
    atomic_json(MANIFEST, mani)

    hit = [e for e in done.values() if e.get("kept", 0) > 0]
    n_local = sum(1 for e in hit for im in e["images"]
                  if im["uuid"] in local)
    print(f"\nPhase2 完成: 有深海图的类别 {len(hit)}/{len(done)}, "
          f"共 {sum(e['kept'] for e in hit)} 张 "
          f"(本地 {n_local} / S3 待下 {sum(e['kept'] for e in hit)-n_local})")
    print("深海图类别门分布:",
          dict(__import__("collections").Counter(
              e.get("phylum") for e in hit).most_common()))
    print(f"输出: {MANIFEST}", flush=True)


# ---------------------------------------------------------------- Phase 3
def copy_one(job):
    src_local, url, dst = job
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    tmp = dst + ".part"
    try:
        if src_local:
            shutil.copy2(src_local, dst)
        else:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(tmp, dst)
        return os.path.getsize(dst) > 0
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def phase_merge():
    mani = load_json(MANIFEST, None)
    if not mani:
        sys.exit("先运行 query")
    with open(MSP_JSON) as f:
        idx = json.load(f)
    msp = idx["morphospecies"]
    existing_sp, existing_disp = load_existing()
    have_fn = set()
    for m in msp:
        for e in m.get("entries", []):
            if e.get("fathomnet_concept"):
                have_fn.add(e["fathomnet_concept"])

    hits = sorted((e for e in mani["concepts"].values()
                   if e.get("kept", 0) > 0),
                  key=lambda e: e["concept"])
    print(f"有深海图的类别 {len(hits)} 个, 开始复制图片", flush=True)

    # 生成复制任务 (本地优先, 缺失走 S3)
    jobs, key_of = [], {}
    for e in hits:
        d = os.path.join(POOL_ROOT, safe_dirname(e["concept"]))
        os.makedirs(d, exist_ok=True)
        for im in e["images"]:
            dst = os.path.join(d, im["uuid"] + ".png")
            src = None
            for sub in ("Train", "Val"):
                p = os.path.join(FN_ROOT, sub, im["uuid"] + ".png")
                if os.path.exists(p):
                    src = p
                    break
            jobs.append((src, im.get("url"), dst))
            key_of[dst] = (e["concept"], im["uuid"])
    ok_dst = set()
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for job, ok in zip(jobs, ex.map(copy_one, jobs)):
            if ok:
                ok_dst.add(job[2])
    n_fail = len(jobs) - len(ok_dst)
    print(f"图片就绪 {len(ok_dst)} 张 (失败 {n_fail}, "
          f"{time.time()-t0:.0f}s)", flush=True)

    n_add = n_skip = 0
    next_id = 1 + max(
        (int(m["morphospecies_id"][4:]) for m in msp
         if m["morphospecies_id"].startswith("FNM_")), default=0)
    for e in hits:
        concept = e["concept"]
        if concept in have_fn:
            n_skip += 1
            continue
        lat = e["latin"]
        if e["kind"] == "species" and lat.lower() in existing_sp:
            n_skip += 1
            continue
        if concept.strip().lower() in existing_disp:
            n_skip += 1
            continue
        d = os.path.join(POOL_ROOT, safe_dirname(concept))
        rel = [os.path.relpath(os.path.join(d, im["uuid"] + ".png"), BASE)
               for im in e["images"]
               if os.path.join(d, im["uuid"] + ".png") in ok_dst]
        if not rel:
            continue
        depths = [im["depth"] for im in e["images"]
                  if os.path.join(POOL_ROOT, safe_dirname(concept),
                                  im["uuid"] + ".png") in ok_dst]
        chain = e.get("taxonomy_worms", {})
        lineage = {rk: chain[rk]["name"] for rk in
                   ("phylum", "class", "order", "family", "genus", "species")
                   if rk in chain}
        rank = e.get("rank") or "species"
        idx["morphospecies"].append({
            "morphospecies_id": f"FNM_{next_id:05d}",
            "rank": rank,
            "rank_cn": RANK_CN.get(rank, rank),
            "latin": lat,
            "display_name": concept,
            "name_cn": "",
            "suffix": e.get("suffix", ""),
            "lineage": lineage,
            "morphotype_ids": [],
            "entries": [{
                "source": "FathomNet 深海扩充 (ROV 原位影像, depth>=500m)",
                "fathomnet_concept": concept,
            }],
            "images": rel,
            "image_count": len(rel),
            "worms": e.get("worms"),
            "taxonomy_worms": {
                rk: {"name": v.get("name"),
                     "aphia_id": v.get("aphia_id"),
                     "gbif_key": None}
                for rk, v in chain.items()},
            "descriptions": [],
            "vernacular_names": {},
            "expansion": {
                "source": "FathomNet",
                "phylum": e.get("phylum"),
                "api_images": e.get("api_count"),
                "deep_images": e.get("deep_count"),
                "kept_images": len(rel),
                "depth_min": min(depths) if depths else None,
                "depth_max": max(depths) if depths else None,
                "imaging": "ROV",
            },
        })
        next_id += 1
        n_add += 1

    _update_stats(idx)
    atomic_json(MSP_JSON, idx)
    print(f"新增 {n_add} 个 FNM 形态种 (跳过重复 {n_skip}), "
          f"总数 {len(idx['morphospecies'])}")
    print(f"输出: {MSP_JSON}", flush=True)


def _update_stats(idx):
    from collections import Counter
    msp = idx["morphospecies"]
    by_rank = Counter(m["rank"] for m in msp)
    img_by_rank = Counter()
    for m in msp:
        img_by_rank[m["rank"]] += m.get("image_count", 0)
    st = idx.setdefault("statistics", {})
    st["total_morphospecies"] = len(msp)
    st["total_images"] = sum(m.get("image_count", 0) for m in msp)
    st["by_rank"] = dict(by_rank)
    st["images_by_rank"] = dict(img_by_rank)
    st["deepsea_expansion_species"] = sum(
        1 for m in msp if m["morphospecies_id"].startswith("WDS_"))
    st["fathomnet_expansion_species"] = sum(
        1 for m in msp if m["morphospecies_id"].startswith("FNM_"))
    if "FNM_" not in idx.get("description", ""):
        idx["description"] = idx.get("description", "") + (
            " FNM_ 前缀=FathomNet 深海扩充种 (MBARI 等 ROV 原位影像, "
            "depth>=500m, 见 data/processed/fathomnet_expansion/)。")


# ---------------------------------------------------------------- main
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in (
            "enumerate", "query", "merge"):
        print(__doc__)
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    phase = sys.argv[1]
    if phase in ("enumerate", "query"):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
        except FileExistsError:
            sys.exit(f"已有进程在运行 (锁: {LOCK}); 确认无进程可删该文件")
        try:
            if phase == "enumerate":
                phase_enumerate()
            else:
                phase_query()
        finally:
            try:
                os.remove(LOCK)
            except OSError:
                pass
    else:
        phase_merge()


if __name__ == "__main__":
    main()
