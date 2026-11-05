#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
形态种 (morphospecies) 映射构建

原则:
  - 未定种 (sp./indet./sp. inc./gen. indet. 等) 的标签打到能确定的最低阶元:
    可能为纲/目/科/属, 不一定是属或科 — 以 scientific_name 中 sp. 前缀名与
    条目自身分类阶元字段的匹配结果为准
  - 不改变原有图片目录, 用映射 + 符号链接新建形态种视图:
      data/processed/atlas_lookup/morphospecies_index.json   映射索引
      data/processed/atlas_lookup/形态种/<rank>/<形态种名>/   图片符号链接
  - APSMA 374 条已是专家整理的形态型 (morphotype_id), 1:1 映射保留其粒度;
    西太 + 海山条目按 (能确定的最低阶元, 拉丁名) 聚合 — 同属/同科/同目的
    未定种跨图集合并, 提升每类图片量

morphospecies 记录:
  morphospecies_id  APSMA 用原 morphotype_id; 其余 MSP_<rank>_<拉丁名>
  rank/rank_cn      能确定的最低阶元 (species..phylum)
  latin/display     规范拉丁名 / 原始学名 (含 sp. 等后缀)
  lineage           phylum..genus 完整谱系 (下游可按任意阶元聚合)
  entries/images    来源条目引用与图片相对路径
"""
import os
import re
import json
import time
import shutil
import difflib
import urllib.request
import urllib.error
import urllib.parse
from collections import Counter
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECIES_JSON = os.path.join(
    BASE, "data/processed/atlas_lookup/species_name_index.json")
MSP_JSON = os.path.join(
    BASE, "data/processed/atlas_lookup/morphospecies_index.json")
MSP_IMG_ROOT = os.path.join(BASE, "data/processed/atlas_lookup/形态种")

# 阶元顺序 (细 -> 粗); rank_cn 用于中文标注
RANK_ORDER = ["species", "genus", "subfamily", "family", "superfamily",
              "infraorder", "suborder", "order", "superorder", "infraclass",
              "subclass", "class", "superclass", "subphylum", "phylum"]
RANK_IDX = {r: i for i, r in enumerate(RANK_ORDER)}
RANK_CN = {
    "species": "种", "genus": "属", "subfamily": "亚科", "family": "科",
    "superfamily": "总科", "infraorder": "次目", "suborder": "亚目",
    "order": "目", "superorder": "总目", "infraclass": "次纲",
    "subclass": "亚纲", "class": "纲", "superclass": "总纲",
    "subphylum": "亚门", "phylum": "门",
}
LINEAGE_RANKS = ["phylum", "subphylum", "superclass", "class", "subclass",
                 "infraclass", "superorder", "order", "suborder",
                 "infraorder", "superfamily", "family", "subfamily", "genus"]

WORMS_TAXA_CACHE = os.path.join(
    BASE, "data/processed/atlas_lookup/.morphospecies_worms_taxa_cache.json")
GBIF_CHAIN_CACHE = os.path.join(
    BASE, "data/processed/atlas_lookup/.morphospecies_gbif_cache.json")

# WoRMS 印刷错字手动纠正 (fuzzy 端点偶发 200+HTML 空页, None 入缓存后不再
# 重试): 名 -> (纠正名, AphiaID, rank)。已实地核实为 WoRMS 接受名。
MANUAL_WORMS_FIX = {
    "Rosellidae": ("Rossellidae", 131694, "Family"),
    "Deltocyathiidae": ("Deltocyathidae", 708974, "Family"),
    "Hymenopeneaus": ("Hymenopenaeus", 106825, "Genus"),
}

# WoRMS 框架性空档人工补链: 查询名 -> {rank: (阶元名, AphiaID)}
# Polycladida: WoRMS 以亚门 Rhabditophora 替代旧纲级概念, 纲级节点为空
MANUAL_CHAIN_FIX = {
    "Polycladida": {"subphylum": ("Rhabditophora", 479175)},
}

# 图集旧分类与 WoRMS 现行冲突的纠链: 查询名 -> {rank: (WoRMS现行名, AphiaID)}
# lineage 保持书原文不动 (以书为准), 仅修正 taxonomy_worms 参考链;
# 与 MANUAL_CHAIN_FIX 的"只补缺"不同, 这里必须覆盖 (过时值已被
# lineage 回填写入 tw)
MANUAL_TAXONOMY_FIX = {
    "Lithodidae": {  # 石蟹科 = 致癌化的寄居蟹; 书用旧分类 Brachyura/Majoidea
        "infraorder": ("Anomura", 106671),
        "superfamily": ("Paguroidea", 106687),
    },
}

# 阶元链缺口说明 (chain_gaps 字段): (token, rank) -> 原因; 缺省为通用说明。
# 实证结论 (2026-09 WoRMS 实测): 勿硬补以下节点, 属权威现状
GAP_REASONS = {
    ("Sagittoidea", "order"):
        "Sagittoidea 纲下 WoRMS 有两个并列目 (Phragmophora 5951 / "
        "Aphragmophora 5950), 图集仅定到纲, 不强行二选一",
    ("Benthothuria", "family"):
        "WoRMS 属/种级记录 family 均未挂 (科级位置未定 incertae sedis); "
        "文献旧归属 Synallactidae(123185) 属 Synallactida 目, 与 WoRMS "
        "挂的 Persiculida 目(1036230) 矛盾, 不硬补",
    ("Hansenothuria", "family"):
        "WoRMS 属/种级记录 family 均未挂 (科级位置未定); 文献旧归属 "
        "Synallactidae 与 WoRMS 挂的 Persiculida 目矛盾, 不硬补",
    ("Polycladida", "class"):
        "WoRMS 以亚门 Rhabditophora(479175) 替代旧纲级概念, 纲级节点"
        "框架性为空 (本记录已补 subphylum 节点)",
}
GBIF_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus",
              "species"]

# 学名后缀解析: 'X gen. indet. 2' / 'X sp. indet.' / 'X sp. inc.' /
# 'X sp. 1' (编号未定种, 书中区分同高阶元下不同形态型) / 'X spp.'
# 限定词 (gen./fam./order./sp.) 表示"该级未定", token 的阶元即可确定级别
SUFFIX_RE = re.compile(
    r"^(?P<token>.+?)\s+(?P<qual>gen|fam|sp|order|species)\.\s*"
    r"(?P<tail>indet|inc)(?:\.?\s*(?P<num>\d+))?$"
    r"|^(?P<token2>.+?)\s+(?P<sfx>spp?|indet)(?:\.?\s*(?P<num2>\d+))?$",
    re.I)


def sanitize_name(name):
    s = re.sub(r'[\\/:*?"<>|]+', " ", name)
    return re.sub(r"\s+", " ", s).strip().rstrip(".")


def parse_scientific(sn):
    """解析学名 -> (token, display, suffix_desc, sp_number)。
    token = 未定种的阶元前缀名 (用于 rank 判定); 确定种返回 None。
    sp_number = 编号未定种的编号 ('Araeosoma sp. 1' -> '1')。"""
    n = re.sub(r"\s+", " ", sn or "").strip()
    if not n:
        return None, n, None, None
    core = n.rstrip(".").rstrip()
    m = SUFFIX_RE.match(core)
    if m:
        if m.group("token") is not None:
            return (re.sub(r"\s+", " ", m.group("token").strip()), n,
                    f"{m.group('qual')}. {m.group('tail')}.", m.group("num"))
        tok = re.sub(r"\s+(?:gen|fam|sp|order)\.$", "",
                     m.group("token2").strip(), flags=re.I)
        return (re.sub(r"\s+", " ", tok), n, m.group("sfx").lower(),
                m.group("num2"))
    # 无未定后缀: 二名法 (可含 cf./aff./亚属括号) = 确定种; 其余 (单词名、
    # 'X gen.'、'X/Y' 等) 按未定种处理, token = 去限定词后的名称
    probe = re.sub(r"\b(c f|cf|aff)\.?\b", "", core, flags=re.I)
    probe = re.sub(r"[()]", " ", probe)
    words = [w for w in re.sub(r"\s+", " ", probe).strip().split(" ") if w]
    qual_words = {"gen", "fam", "sp", "spp", "order", "species",
                  "indet", "inc", "nov"}
    if (len(words) == 2 and words[0][:1].isupper()
            and words[1][:1].islower() and words[1].isalpha()
            and words[1].lower() not in qual_words):
        return None, n, None, None  # 确定种
    tok = re.sub(r"\s+(?:gen|fam|sp|species|order)\.?$", "", core,
                 flags=re.I)
    return re.sub(r"\s+", " ", tok).strip(), n, None, None


def worms_taxa(name, cache, pending=None):
    """WoRMS 查询名称 -> 完整记录 (AphiaID/rank/status/接受名/classification
    祖先链); 未命中返回 None。
    pending 非空时为"收集模式": 只记录待查名不联网 (配合并行预取)。
    精确未命中时用 AphiaRecordsByMatchNames fuzzy 端点找拼写建议
    (ratio>0.85 才采纳, 如 Hymenopeneaus -> Hymenopenaeus)。"""
    if name in cache:
        return cache[name]
    if pending is not None:
        pending.add(name)
        return None
    rec = None

    def to_rec(d):
        if not isinstance(d, dict) or not d.get("AphiaID"):
            return None
        return {
            "aphia_id": d.get("AphiaID"),
            "scientificname": d.get("scientificname"),
            "authority": d.get("authority"),
            "rank": (d.get("rank") or "").lower() or None,
            "status": d.get("status"),
            "valid_name": d.get("valid_name"),
            "valid_aphia_id": d.get("valid_AphiaID"),
            "classification": d.get("classification") or [],
            # 新版 WoRMS 记录自带 kingdom..genus 直接字段 (补链用)
            "chain_direct": {rk: d.get(rk) for rk in
                             ("kingdom", "phylum", "class", "order",
                              "family", "genus") if d.get(rk)},
        }

    url = ("https://www.marinespecies.org/rest/AphiaRecordsByName/"
           + urllib.parse.quote(name) + "?marine_only=false")
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list) and data:
            rec = to_rec(data[0])
    except urllib.error.HTTPError as e:
        if e.code in (404, 204):
            rec = None  # 确定性未命中
        else:
            return None  # 服务端错误: 不写缓存, 下次重试
    except Exception:
        return None  # 超时/网络错误: 不写缓存 (避免污染, 下次重试)
    if rec is None:  # fuzzy 拼写纠正
        try:
            furl = ("https://www.marinespecies.org/rest/"
                    "AphiaRecordsByMatchNames?scientificnames%5B%5D="
                    + urllib.parse.quote(name) + "&marine_only=false")
            req = urllib.request.Request(
                furl, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                fdata = json.loads(resp.read().decode())
            if isinstance(fdata, list) and fdata and fdata[0]:
                cand = to_rec(fdata[0][0])
                if cand and cand.get("rank") in RANK_CN:
                    ratio = difflib.SequenceMatcher(
                        None, name.lower(),
                        (cand.get("scientificname") or "").lower()).ratio()
                    if ratio > 0.85:
                        cand["corrected_from"] = name
                        rec = cand
                        print(f"  fuzzy 纠正: {name} -> "
                              f"{cand['scientificname']} ({cand['rank']})")
        except Exception:
            pass
    cache[name] = rec
    time.sleep(0.05)
    return rec


def infer_rank(token, entry, style, cache, pending=None):
    """token 的真实阶元判定 (按可靠性排序):
    1. 条目自身阶元字段精确匹配; 2. 分类词尾规则 (-idae/-inae/-oidea);
    3. WoRMS 查询 rank (含 fuzzy 拼写纠正)。
    返回 (rank, WoRMS命中学名或None)。"""
    r = rank_of_token(token, entry, style)
    if r:
        return r, None
    t = token.lower()
    if t.endswith("idae"):
        return "family", None
    if t.endswith("inae"):
        return "subfamily", None
    if t.endswith("oidea"):
        return "superfamily", None
    rec = worms_taxa(token, cache, pending)
    if rec:
        return rec["rank"], rec["scientificname"]
    return None, None


def determine_msp(entry, style, worms_cache, pending=None):
    """判定条目的形态种: 返回 (rank, latin, display, suffix, lineage)。
    确定种 -> species; 未定种 -> 前缀名的真实阶元 (谱系匹配/词尾/WoRMS);
    编号未定种 ('X sp. 1') 的 latin 保留编号 (保持书中形态型区分)。"""
    sn = (entry.get("scientific_name") or "").strip()
    token, display, suffix, sp_num = parse_scientific(sn)
    lineage = build_lineage(entry, style)
    if token is None:  # 确定种
        if not display:
            # 无学名: 回退到谱系中最低非空阶元
            for r in RANK_ORDER:
                if r != "species" and lineage.get(r):
                    return r, lineage[r], display, "fallback", lineage
            return None, "", display, "empty", lineage
        return "species", display, display, None, lineage
    # 二名 token (如 'Axoniderma mexicana sp. inc.') -> 属 = 首词
    words = token.split()
    if len(words) >= 2 and "/" not in token:
        token = words[0]
    base = token
    if "/" in token:  # 双属斜杠 ('Paelopatides/Benthothuria sp. 3'):
        # 属级未定 (两属之间), 保留双属名作独立形态种
        rank = "genus"
    else:
        rank, wmatch = infer_rank(token, entry, style, worms_cache, pending)
        if wmatch:
            base = wmatch  # WoRMS fuzzy 纠正后的规范拼写
    if rank is None:
        # token 阶元无法确定 -> 谱系最低非空阶元兜底
        for r in RANK_ORDER:
            if r != "species" and lineage.get(r):
                print(f"  !! token 阶元未知, 谱系兜底: {display!r} -> {r}")
                return r, lineage[r], display, suffix, lineage
        print(f"  !! 无法确定阶元: {display!r}")
        return None, "", display, suffix, lineage
    latin = f"{base} sp. {sp_num}" if sp_num else base
    if not lineage.get(rank):
        lineage[rank] = base  # 谱系缺失该阶元时补全
    return rank, latin, display, suffix, lineage


def get_latin(entry, rank, style):
    """按来源风格取条目在指定阶元的拉丁名。"""
    if style == "seamount":
        v = (entry.get("taxonomy") or {}).get(rank)
        return (v or {}).get("latin") or ""
    return (entry.get(rank) or "").strip()


def rank_of_token(token, entry, style):
    """token 与条目自身阶元字段匹配 -> 最低匹配阶元; 失败返回 None。"""
    tok = token.lower()
    for r in RANK_ORDER:
        if r == "species":
            continue
        v = get_latin(entry, r, style)
        if v and v.lower() == tok:
            return r
    return None


def build_lineage(entry, style):
    lin = {}
    for r in LINEAGE_RANKS:
        v = get_latin(entry, r, style)
        if v:
            lin[r] = v
    return lin


def collect_images(entry, style):
    """返回条目的图片相对路径列表。"""
    if style == "westpac":
        p = (entry.get("path") or "").strip()
        return [p] if p else []
    return [im["path"] for im in (entry.get("images") or [])
            if (im.get("path") or "").strip()]


def msp_query_names(rec):
    """形态种 -> WoRMS 查询名列表 (依次尝试取首个命中)。
    编号未定种剥 'sp. N' 查前缀名; cf./aff. 剥离后查; 种未命中退化查属;
    双属斜杠拆开分别查询。"""
    latin = rec["latin"]
    if rec["rank"] == "species":
        n = re.sub(r"\s+(cf|aff)\.\s*", " ", latin, flags=re.I).strip()
        names = [n] if n else []
        g = n.split(" ")[0] if n else ""
        if g and g != n:
            names.append(g)
        return names
    n = re.sub(r"\s+sp\.\s*\d+$", "", latin).strip()
    if "/" in n:
        return [w.strip() for w in n.split("/") if w.strip()]
    return [n] if n else []


def gbif_chain(name, cache, pending=None):
    """GBIF species/match -> 全阶元链 {rank: {name, gbif_key}}
    (kingdom..genus/species, GBIF 海洋部分以 WoRMS 为 backbone,
    链名用于回查 AphiaID); 未命中返回 None。"""
    if name in cache:
        return cache[name]
    if pending is not None:
        pending.add(name)
        return None
    chain = None
    url = ("https://api.gbif.org/v1/species/match?name="
           + urllib.parse.quote(name))
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            m = json.loads(resp.read().decode())
        if isinstance(m, dict) and m.get("usageKey"):
            chain = {}
            for rk in GBIF_RANKS:
                if m.get(rk):
                    chain[rk] = {"name": m[rk], "gbif_key": m.get(rk + "Key")}
    except Exception:
        return None  # 网络错误不写缓存
    cache[name] = chain
    time.sleep(0.05)
    return chain


def build_taxonomy_worms(rec, gbif_cache, taxa_cache):
    """形态种 -> 全阶元链 {rank: {name, aphia_id, gbif_key}}。
    链名取自 GBIF match (海洋 backbone=WoS), 各级 AphiaID 回查 WoRMS;
    自身阶元以 WoRMS 命中记录覆盖 (含亚级阶元, GBIF 链只给主阶元)。"""
    tw = {}
    names = msp_query_names(rec)
    chain = gbif_cache.get(names[0]) if names else None
    if chain:
        for rk, node in chain.items():
            w = taxa_cache.get(node["name"])
            tw[rk] = {"name": node["name"],
                      "aphia_id": w.get("aphia_id") if w else None,
                      "gbif_key": node.get("gbif_key")}
    # GBIF 链缺失的阶元 (GBIF match 常缺 phylum/family/亚级节点)
    # -> 图集谱系回查补全 (不覆盖 GBIF 已有节点)
    for r in LINEAGE_RANKS:
        if r in tw:
            continue
        nm = rec.get("lineage", {}).get(r)
        if nm:
            w = taxa_cache.get(nm)
            tw[r] = {"name": nm,
                     "aphia_id": w.get("aphia_id") if w else None,
                     "gbif_key": None}
    # WoRMS 记录自带 kingdom..genus 直接字段 (补 GBIF/谱系都缺的阶元)
    blocks = rec["worms"].get("candidates") or [rec["worms"]]
    for b in blocks:
        for rk, nm in (b.get("chain_direct") or {}).items():
            if nm and rk not in tw:
                w = taxa_cache.get(nm)
                tw[rk] = {"name": nm,
                          "aphia_id": w.get("aphia_id") if w else None,
                          "gbif_key": None}
    # 框架性空档人工补链 (MANUAL_CHAIN_FIX)
    for nm_q in names:
        for rk, (nm, aphia) in MANUAL_CHAIN_FIX.get(nm_q, {}).items():
            if rk not in tw:
                tw[rk] = {"name": nm, "aphia_id": aphia, "gbif_key": None}
    # 旧分类纠链 (MANUAL_TAXONOMY_FIX): 覆盖 lineage 回填的过时节点
    for nm_q in names:
        for rk, (nm, aphia) in MANUAL_TAXONOMY_FIX.get(nm_q, {}).items():
            tw[rk] = {"name": nm, "aphia_id": aphia, "gbif_key": None}
    # 自身阶元: WoRMS 命中记录 (亚科/亚目等 GBIF 链缺失的阶元由此补全)
    for b in blocks:
        if b.get("aphia_id") and b.get("rank") == rec["rank"]:
            tw[rec["rank"]] = {"name": b.get("scientificname") or rec["latin"],
                               "aphia_id": b["aphia_id"],
                               "gbif_key": tw.get(rec["rank"], {}).get("gbif_key")}
            break
    return tw


def attach_worms(rec, taxa_cache, pending=None, gbif_cache=None):
    """为形态种挂 WoRMS 信息:
    worms          自身阶元名的 AphiaID/接受名/状态 (唯一 AphiaID 统一标识)
    taxonomy_worms kingdom..自身 全阶元链, 每级含 AphiaID
    双属斜杠时 worms.candidates 列出两属各自的 AphiaID。"""
    names = msp_query_names(rec)
    blocks, first_rec = [], None
    for qn in names:
        w = worms_taxa(qn, taxa_cache, pending)
        if w is None:
            blocks.append({"query_name": qn, "aphia_id": None})
            continue
        if first_rec is None:
            first_rec = w
        b = {
            "query_name": qn,
            "aphia_id": w["aphia_id"],
            "scientificname": w["scientificname"],
            "rank": w["rank"],
            "status": w["status"],
            "url": ("https://www.marinespecies.org/aphia.php?p=taxdetails&id="
                    + str(w["aphia_id"])),
        }
        if w.get("valid_name"):  # 未接受名 -> 附有效名
            b["valid_name"] = w["valid_name"]
            b["valid_aphia_id"] = w["valid_aphia_id"]
        if w.get("corrected_from"):
            b["corrected_from"] = w["corrected_from"]
        if w.get("chain_direct"):
            b["chain_direct"] = w["chain_direct"]
        blocks.append(b)
    if len(blocks) > 1:  # 双属: 两候选并列
        rec["worms"] = {"candidates": blocks}
    else:
        rec["worms"] = blocks[0] if blocks else {"query_name": None,
                                                 "aphia_id": None}
    if gbif_cache is not None:  # 收集模式 (gbif_cache=None) 时不建链
        rec["taxonomy_worms"] = build_taxonomy_worms(rec, gbif_cache,
                                                     taxa_cache)


def parallel_fetch(names, cache, fn=worms_taxa):
    """并行预取 (marinespecies.org 单请求延迟高, 串行不可行)。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    todo = sorted(n for n in names if n not in cache)
    if not todo:
        return
    print(f"并行预取 ({fn.__name__}): {len(todo)} 个名称 (12 线程)...")
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fn, n, cache): n for n in todo}
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
    n_hit = sum(1 for n in todo if cache.get(n))
    print(f"  预取完成: 命中 {n_hit}, 未命中 {len(todo) - n_hit}")


def build_all(sp, taxa_cache, pending=None):
    """两遍构建的第一/第二遍 (pending=None 时全缓存命中, 不联网)。"""
    msp = {}   # key -> morphospecies 记录
    order = []  # 保持首次出现顺序

    def add_msp(key, base):
        if key not in msp:
            msp[key] = base
            order.append(key)
        else:
            if base["morphotype_ids"]:
                msp[key]["morphotype_ids"].extend(base["morphotype_ids"])
        return msp[key]

    # --- 1. APSMA: 每个形态型独立 morphospecies (保留专家粒度) ---
    # 原书编号有两处印刷重号 (AST_012: p127/p128; HOL_108: p166/p169,
    # 2026-09 PDF 全书核验), 重复 id 加 _dupN 后缀保证形态型不互并
    ap_ids = set()
    for e in sp["apsma_entries"]:
        rank, latin, display, suffix, lineage = determine_msp(
            e, "apsma", taxa_cache, pending)
        if rank is None:
            continue
        mid = e.get("morphotype_id") or f"MSP_{rank}_{sanitize_name(latin)}"
        if mid in ap_ids:
            k = 2
            while f"{mid}_dup{k}" in ap_ids:
                k += 1
            mid = f"{mid}_dup{k}"
        ap_ids.add(mid)
        rec = add_msp(("AP", mid), {
            "morphospecies_id": mid, "rank": rank,
            "rank_cn": RANK_CN[rank], "latin": latin, "display_name": display,
            "name_cn": "", "suffix": suffix, "lineage": lineage,
            "morphotype_ids": [mid] if e.get("morphotype_id") else [],
            "entries": [], "images": [], "image_count": 0, "_src": "AP",
        })
        imgs = collect_images(e, "apsma")
        rec["entries"].append({"source": "北太平洋图集(APSMA完整版)",
                               "morphotype_id": e.get("morphotype_id")})
        rec["images"].extend(imgs)

    # --- 2. 西太 + 海山: 按 (rank, latin) 跨图集聚合 ---
    # 编号未定种 ('X sp. N') 同前缀名 + 同编号跨图集合并为同一形态种
    # (同归属 WoRMS 一致即按同种归类); 同书内不同编号仍各自独立
    SRC = [("westpac", "西太平洋图集", "westpac_entries", "seq"),
           ("seamount", "海山动物原色图谱", "seamount_entries", "entry_no")]
    for style, src_name, list_key, id_key in SRC:
        for e in sp[list_key]:
            rank, latin, display, suffix, lineage = determine_msp(
                e, style, taxa_cache, pending)
            if rank is None:
                continue
            mid = f"MSP_{rank}_{re.sub(r'[^A-Za-z0-9]+', '_', latin).strip('_')}"
            # 编号未定种 ('X sp. N'): 同前缀名 + 同编号跨图集合并为同一
            # 形态种 (同归属 WoRMS 一致即按同种归类, 不分书表示);
            # 编号格式差异 ('sp.1'/'sp. 1') 归一后合并
            tok, _, _, num = parse_scientific(latin)
            key = ((rank, (tok or latin).lower(), num) if num is not None
                   else (rank, latin.lower()))
            rec = add_msp(key, {
                "morphospecies_id": mid, "rank": rank,
                "rank_cn": RANK_CN[rank], "latin": latin,
                "display_name": display,
                "name_cn": (e.get("name_cn") or "").strip(),
                "suffix": suffix, "lineage": lineage,
                "morphotype_ids": [], "entries": [], "images": [],
                "image_count": 0, "_src": style,
            })
            if not rec["name_cn"] and e.get("name_cn"):
                rec["name_cn"] = (e.get("name_cn") or "").strip()
            imgs = collect_images(e, style)
            ref = {"source": src_name, id_key: e.get(id_key)}
            rec["entries"].append(ref)
            rec["images"].extend(imgs)
    return msp, order


def _rm_retry(func, path, exc):
    """NAS (NFS) 异步删除竞态: rmdir 报 Directory not empty 时重试。"""
    for _ in range(6):
        time.sleep(0.5)
        try:
            func(path)
            return
        except OSError:
            continue


def main():
    with open(SPECIES_JSON) as f:
        sp = json.load(f)
    # 重建前清空旧符号链接树 (只含链接, 不动原图目录)
    if os.path.exists(MSP_IMG_ROOT):
        shutil.rmtree(MSP_IMG_ROOT, onexc=_rm_retry)
    # WoRMS 分类单元缓存 (全记录: AphiaID/status/classification 链)
    taxa_cache = {}
    if os.path.exists(WORMS_TAXA_CACHE):
        with open(WORMS_TAXA_CACHE) as f:
            taxa_cache = json.load(f)
    for bad, (good, aphia, rank) in MANUAL_WORMS_FIX.items():
        if taxa_cache.get(bad) is None:
            taxa_cache[bad] = {
                "aphia_id": aphia, "scientificname": good,
                "rank": rank.lower(), "status": "accepted",
                "valid_name": good, "valid_aphia_id": aphia,
                "classification": [], "chain_direct": {},
                "corrected_from": bad}
            print(f"  手动纠正播种: {bad} -> {good} (AphiaID {aphia})")
    # 旧缓存条目无 chain_direct (WoRMS 记录新自带的 kingdom..genus 字段),
    # 删除后随本轮重取
    stale = [k for k, v in taxa_cache.items()
             if v is not None and "chain_direct" not in v]
    for k in stale:
        del taxa_cache[k]
    if stale:
        print(f"刷新旧缓存条目 (补 chain_direct): {len(stale)} 个")

    # Pass 1: 收集待查名 (阶元推断 miss + 各形态种查询名), 不联网
    pending = set()
    msp1, _ = build_all(sp, taxa_cache, pending)
    for rec in msp1.values():
        pending.update(msp_query_names(rec))
    print(f"待查名称: {len(pending)} 个")
    parallel_fetch(pending, taxa_cache, worms_taxa)
    with open(WORMS_TAXA_CACHE, "w") as f:
        json.dump(taxa_cache, f, ensure_ascii=False, indent=1)

    # Pass 1b: GBIF 全阶元链 + 链上各级名称回查 WoRMS AphiaID
    # (WoRMS REST 已不返回 classification 字段, 链名取 GBIF match,
    #  其海洋部分以 WoRMS 为 backbone)
    gbif_cache = {}
    if os.path.exists(GBIF_CHAIN_CACHE):
        with open(GBIF_CHAIN_CACHE) as f:
            gbif_cache = json.load(f)
    gbif_pending = set()
    for rec in msp1.values():
        names = msp_query_names(rec)
        if names:
            gbif_chain(names[0], gbif_cache, gbif_pending)
    parallel_fetch(gbif_pending, gbif_cache, gbif_chain)
    with open(GBIF_CHAIN_CACHE, "w") as f:
        json.dump(gbif_cache, f, ensure_ascii=False, indent=1)
    anc_pending = set()
    for chain in gbif_cache.values():
        for node in (chain or {}).values():
            if node.get("name") and node["name"] not in taxa_cache:
                anc_pending.add(node["name"])
    # 图集谱系名也需回查 AphiaID (补全 GBIF 链缺失的阶元);
    # 链条祖先名理应命中 WoRMS, 已缓存 None 的多为 HTML 空页 flaky, 删除重试
    for rec in msp1.values():
        for nm in (rec.get("lineage") or {}).values():
            if nm and nm not in taxa_cache:
                anc_pending.add(nm)
    # WoRMS 记录自带 kingdom..genus 直接字段的祖先名也回查 AphiaID
    for v in taxa_cache.values():
        for nm in ((v or {}).get("chain_direct") or {}).values():
            if nm and nm not in taxa_cache:
                anc_pending.add(nm)
    for n in sorted(anc_pending):
        if n in taxa_cache and taxa_cache[n] is None:
            del taxa_cache[n]
    parallel_fetch(anc_pending, taxa_cache, worms_taxa)
    with open(WORMS_TAXA_CACHE, "w") as f:
        json.dump(taxa_cache, f, ensure_ascii=False, indent=1)

    # 本地近似纠正: WoRMS/fuzzy 均未命中的名 (多为印刷错字), 与缓存中
    # 已验证名做相似度比对 (如 Rosellidae -> Rossellidae)
    for name, v in list(taxa_cache.items()):
        if v is not None:
            continue
        best, br = None, 0.0
        for k, w in taxa_cache.items():
            if w and w.get("rank") and abs(len(k) - len(name)) <= 2:
                ratio = difflib.SequenceMatcher(
                    None, name.lower(), k.lower()).ratio()
                if ratio > br:
                    br, best = ratio, k
        if best and br > 0.85:
            print(f"  本地近似纠正: {name} -> {best} (ratio={br:.2f})")
            taxa_cache[name] = dict(taxa_cache[best])
            taxa_cache[name]["corrected_from"] = name
    with open(WORMS_TAXA_CACHE, "w") as f:
        json.dump(taxa_cache, f, ensure_ascii=False, indent=1)

    # Pass 2: 全缓存命中构建 + 挂 WoRMS AphiaID/全阶元链 + 目录树
    msp, order = build_all(sp, taxa_cache)

    # --- 3. 挂 WoRMS AphiaID + 全阶元链 (统一全球公认标识) ---
    records = []
    n_link = 0
    used_labels = {}  # (rank, 目录名) -> morphospecies_id, 防跨图集同名混目录
    for i, key in enumerate(order):
        rec = msp[key]
        attach_worms(rec, taxa_cache, gbif_cache=gbif_cache)
        if (i + 1) % 100 == 0:
            print(f"  WoRMS 挂载 {i + 1}/{len(order)}")
        rec["image_count"] = len(rec["images"])
        # 阶元链缺口说明: 细于某主阶元却缺该级 -> 标注原因 (WoRMS 现状)
        token = parse_scientific(rec["latin"])[0] or rec["latin"]
        rec["chain_gaps"] = {
            rk: GAP_REASONS.get((token, rk), "WoRMS 数据源未挂该阶元")
            for rk in ("phylum", "class", "order", "family")
            if (RANK_IDX[rec["rank"]] < RANK_IDX[rk]
                and not rec["taxonomy_worms"].get(rk, {}).get("aphia_id"))}
        records.append(rec)
        label = sanitize_name(
            rec["display_name"] or rec["latin"] or rec["morphospecies_id"])
        if used_labels.get((rec["rank"], label),
                           rec["morphospecies_id"]) != rec["morphospecies_id"]:
            # 兜底: 跨图集合并后仍同名的情形按图集名分列, 仍冲突则用唯一 id
            label = f"{label}_{rec['_src']}"
            if used_labels.get((rec["rank"], label)) not in (
                    None, rec["morphospecies_id"]):
                label = rec["morphospecies_id"]
        used_labels[(rec["rank"], label)] = rec["morphospecies_id"]
        rank_dir = os.path.join(MSP_IMG_ROOT, rec["rank"])
        label_dir = os.path.join(rank_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        used = set()
        for img in rec["images"]:
            target = os.path.join(BASE, img)
            if not os.path.exists(target):
                print(f"  !! 图片不存在: {img}")
                continue
            link = os.path.join(label_dir, os.path.basename(img))
            if os.path.basename(img) in used:  # 同名冲突加序号
                stem, ext = os.path.splitext(os.path.basename(img))
                k = 2
                while os.path.exists(os.path.join(label_dir, f"{stem}_{k}{ext}")):
                    k += 1
                link = os.path.join(label_dir, f"{stem}_{k}{ext}")
            used.add(os.path.basename(link))
            if not os.path.lexists(link):
                os.symlink(os.path.relpath(target, label_dir), link)
                n_link += 1

    # --- 4. 统计与输出 ---
    by_rank = Counter(r["rank"] for r in records)
    img_by_rank = Counter()
    for r in records:
        img_by_rank[r["rank"]] += r["image_count"]
    n_merged = sum(1 for r in records if len(r["entries"]) > 1)
    total_img = sum(r["image_count"] for r in records)
    n_matched = sum(1 for r in records
                    if (r["worms"].get("aphia_id")
                        or any(c.get("aphia_id") for c in
                               r["worms"].get("candidates", []))))
    n_accepted = sum(
        1 for r in records
        if r["worms"].get("status") == "accepted"
        or any(c.get("status") == "accepted"
               for c in r["worms"].get("candidates", [])))
    n_full_chain = sum(
        1 for r in records
        if all(r["taxonomy_worms"].get(k, {}).get("aphia_id")
               for k in ("phylum", "class", "order", "family")))
    n_gapped = sum(1 for r in records if r["chain_gaps"])
    for r in records:  # 内部字段不入库
        r.pop("_src", None)
    out = {
        "version": "2.1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "形态种映射: 未定种标签打到能确定的最低阶元; APSMA 按形态型 1:1, "
            "西太+海山按 (rank, latin) 聚合; 图片经符号链接映射到 "
            "形态种/<rank>/<名>/, 原目录不变。"
            "lineage=图集原文阶元 (以书为准); worms=自身阶元名的 WoRMS "
            "AphiaID/接受名/状态 (全球统一标识, 编号未定种挂其前缀名的 "
            "AphiaID); taxonomy_worms=kingdom..genus/species 全阶元链 "
            "(链名取自 GBIF match, 其海洋部分以 WoRMS 为 backbone, "
            "每级含 WoRMS AphiaID + GBIF key; 亚级阶元由自身 WoRMS "
            "记录补全; 参考补全用, 不覆盖 lineage); "
            "chain_gaps=阶元链空缺及原因 (WoRMS 数据源未挂/并列目不可定, "
            "实证结论, 勿硬补); 编号未定种同前缀名+同编号跨图集合并 "
            "(同归属 WoRMS 一致即按同种归类, 不分书表示)"),
        "statistics": {
            "total_morphospecies": len(records),
            "total_images": total_img,
            "by_rank": {r: by_rank[r] for r in RANK_ORDER if r in by_rank},
            "images_by_rank": {r: img_by_rank[r]
                               for r in RANK_ORDER if r in img_by_rank},
            "merged_from_multiple_entries": n_merged,
            "apsma_morphotypes": sum(
                1 for r in records if r["morphotype_ids"]),
            "worms_matched": n_matched,
            "worms_accepted": n_accepted,
            "worms_full_chain_phylum_to_family": n_full_chain,
            "records_with_chain_gaps": n_gapped,
        },
        "morphospecies": records,
    }
    with open(MSP_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n形态种: {len(records)} 个, 图片 {total_img} 张 "
          f"(符号链接 {n_link} 个新建)")
    print("按阶元 (各阶元独立键/目录, 互不混并):")
    for r in RANK_ORDER:
        if r in by_rank:
            print(f"  {r:<11}({RANK_CN[r]}): {by_rank[r]:>4} 形态种, "
                  f"{img_by_rank[r]:>5} 图")
    print(f"跨条目合并的形态种: {n_merged}")
    print(f"WoRMS 命中: {n_matched}/{len(records)} "
          f"(接受名 {n_accepted}, phylum..family 全链 {n_full_chain})")
    print(f"输出: {MSP_JSON}\n      {MSP_IMG_ROOT}")


if __name__ == "__main__":
    main()
