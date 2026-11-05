#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海山动物原色图谱 (《西太平洋海沟洋脊交联区海山动物原色图谱》徐奎栋等) 结构化整理

版面结构 (已目检确认):
  - 正文每页 1-2 个条目: [编号. 中文名 拉丁名] + 分类学地位 + 采集地/水深/分布 + 照片
  - 照片排布在条目文字块下方, 溢出照片流到下一页顶部 (下一标题之前)
  - 图片归属规则 (目检验证): 归属于阅读顺序中它之前最近的条目标题
  - 本 PDF 无条带切片问题 (紧邻图片均为独立照片, p30/p71/p124/p145 已目检)
  - 6 个门类封面页 (p24/57/135/142/160/213) 有照片但无条目, 照片跳过
  - 页脚含印刷页码; 奇数印刷页页眉右上角有门名

输出:
  data/processed/atlas_lookup/海山图谱/<Phylum>/<拉丁名>/  图片
  data/processed/atlas_lookup/seamount_atlas_species_index.json  独立索引
  并合并到 index.json 和 species_name_index.json
"""
import os
import re
import json
import shutil
import time
import difflib
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime

import pymupdf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(
    BASE, "data/raw/生物图集",
    "《西太平洋海沟洋脊交联区海山动物原色图谱》_徐奎栋等 著.pdf")
OUT_ROOT = os.path.join(BASE, "data/processed/atlas_lookup/海山图谱")
SEAMOUNT_JSON = os.path.join(
    BASE, "data/processed/atlas_lookup/seamount_atlas_species_index.json")
INDEX_PATH = os.path.join(BASE, "data/processed/atlas_lookup/index.json")
SPECIES_JSON = os.path.join(
    BASE, "data/processed/atlas_lookup/species_name_index.json")

# 内容页范围 (PDF 页码, 1-based): p24 门封面起, p242 最后条目; p1-23 前言/目录, p252 封底
PAGE_START, PAGE_END = 24, 251
# 书尾索引: 中文名索引 p243-246, 拉丁名索引 p247-251 (格式: "名称 印刷页码")
INDEX_CN_PAGES = range(243, 247)
INDEX_LATIN_PAGES = range(247, 252)
WORMS_CACHE = os.path.join(BASE, "data/processed/atlas_lookup/.seamount_worms_cache.json")
# 门类封面页: PDF页 -> (拉丁门名, 中文门名); 同时界定各门的页码范围
PHYLUM_COVERS = {
    24: ("Porifera", "多孔动物门"),
    57: ("Cnidaria", "刺胞动物门"),
    135: ("Mollusca", "软体动物门"),
    142: ("Arthropoda", "节肢动物门"),
    160: ("Echinodermata", "棘皮动物门"),
    213: ("Chordata", "脊索动物门"),
}

TITLE_RE = re.compile(r"^(\d{1,3})\.\s*(\S.*)$")
RANK_RE = re.compile(
    r"^([\u4e00-\u9fff（）]+?)\s*[\u3000\s]*"
    r"(Phylum|Subphylum|Superclass|Subclass|Class|Suborder|Order|Infraorder|"
    r"Superfamily|Family|Subfamily|Genus)\s+"
    r"([A-Z][A-Za-z\-']+)")
FIELD_RE = re.compile(r"^(采集地|水深|分布|生态)\s*[\u3000\s]*(.+)$")
FIRST_LETTER_RE = re.compile(r"[A-Za-z]")

MIN_IMG_BYTES = 8000
MIN_IMG_PX = 80

# 文本层缺字且书尾索引同样缺字、无法自动恢复的中文名 -> 人工核实修正
# (#263 Lophiodes sp.: 正文与索引的'鮟鱇'二字均因字体缺字丢失)
MANUAL_CN_FIX = {
    "263": "拟鮟鱇属未定种",
}


def sanitize_name(name):
    """物种名转文件夹安全名。"""
    s = re.sub(r'[\\/:*?"<>|]+', " ", name)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s


def norm_text(t):
    """PDF 连字/私有区字符归一。"""
    return (t.replace("\ufb01", "fi").replace("\ufb02", "fl")
             .replace("\ufb00", "ff").replace("\ufb03", "ffi")
             .replace("\ufb04", "ffl").replace("\uf0b7", ""))


def split_cn_latin(text):
    """'中文名拉丁名' 混排拆分: 在首个 ASCII 字母处切开。"""
    m = FIRST_LETTER_RE.search(text)
    if not m:
        return text.strip(), ""
    return text[:m.start()].strip(), text[m.start():].strip()


# 命名人连接词/限定词 (清洗学名时保留或删除的边界词)
AUTHOR_WORDS = {"de", "von", "van", "der", "den", "du", "la", "le",
                "di", "da", "della", "dal", "&", "in", "ex", "et"}
STOP_WORDS = {"cf", "cf.", "aff", "aff.", "sp", "sp.", "spp", "spp.",
              "nov", "indet"}


def short_latin(latin):
    """去掉命名人/年份 (括号式或裸露式), 保留纯学名。
    'Metallogorgia melanotrichos (Wright & Studer, 1889)'
        -> 'Metallogorgia melanotrichos'
    'Bolosoma cf. pedunculatus Schulze, 1896' -> 'Bolosoma cf. pedunculatus'
    'Caenopedina hawaiiensis H. L. Clark, 1912' -> 'Caenopedina hawaiiensis'
    'Araeosoma sp. 4' / 'Lophiodes sp.' 保持不变。
    原始名 (含命名人) 由 name_raw 字段单独保留。"""
    t = re.sub(r"\(.*$", "", latin).strip() or latin
    toks = t.split()
    # 年份 token 是命名人群的锚点, 从那里向前清除人名 token 直至种加词
    yi = next((i for i, w in enumerate(toks)
               if re.match(r"^\(?[12]\d{3}", w)), None)
    if yi is not None:
        j = yi
        while j > 1:
            b = toks[j - 1].rstrip(",.")
            if b in STOP_WORDS or (re.match(r"^[a-z]{2,}$", b)
                                   and b not in AUTHOR_WORDS):
                break  # 种加词或 cf./sp. 等限定词, 停止
            j -= 1
        toks = toks[:j]
    elif len(toks) > 2 and re.match(r"^[A-Z]", toks[-1]) and re.match(
            r"^(?:[A-Z]|&|d[aeiou]|von|van|de|la|le|di|da)", toks[-2]):
        # 无年份的裸人名尾 (罕见, 如 'Genus species Smith'): 仅多词时截断
        toks = toks[:2]
    return " ".join(toks)


def parse_pages(doc):
    """解析所有内容页, 返回 (entries, images, section_headers)。

    entries: 条目列表 (含分类/字段/图片引用)
    images:  已归属到条目的图片记录
    """
    entries = []
    images = []
    headers = []  # (page, y, cn, rank, latin) 纲级章节头
    cur_phylum = ("", "")
    cur_class = ("", "")  # (中文, 拉丁)
    entry = None

    for pi in range(PAGE_START - 1, min(PAGE_END, len(doc))):
        page_no = pi + 1
        page = doc[pi]
        if page_no in PHYLUM_COVERS:
            cur_phylum = PHYLUM_COVERS[page_no]
            cur_class = ("", "")

        blocks = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            lines = []
            for ln in b["lines"]:
                txt = norm_text("".join(s["text"] for s in ln["spans"]).strip())
                if txt:
                    lines.append(txt)
            if lines:
                blocks.append((b["bbox"], lines))

        # 条目标题 (整块首行匹配)
        titles = []  # (y0, no, title_text)
        for bbox, lines in blocks:
            m = TITLE_RE.match(lines[0])
            if m and "...." not in lines[0]:
                titles.append((bbox[1], int(m.group(1)), m.group(2)))
        titles.sort()

        first_title_y = titles[0][0] if titles else 1e9

        # 印刷页码
        printed = None
        for bbox, lines in blocks:
            if bbox[1] > 755 and re.match(r"^\d+$", lines[-1]):
                printed = int(lines[-1])
                break

        # 逐事件 (标题/行/图片) 按 y 处理
        events = []  # (y, kind, payload)
        for bbox, lines in blocks:
            events.append((bbox[1], "text", (bbox, lines)))
        prev_is_taxo = False  # 上一文本事件是否为 '分类学地位' 行
        imgs = []
        if page_no not in PHYLUM_COVERS:
            for info in page.get_image_info(xrefs=True):
                x0, y0, x1, y1 = info["bbox"]
                if (info["width"] < MIN_IMG_PX or info["height"] < MIN_IMG_PX
                        or (x1 - x0) < 40):
                    continue
                try:
                    raw = doc.extract_image(info["xref"])
                except Exception:
                    continue
                if len(raw["image"]) < MIN_IMG_BYTES:
                    continue
                imgs.append({"xref": info["xref"], "bbox": info["bbox"],
                             "ext": raw["ext"], "bytes": raw["image"],
                             "px": (info["width"], info["height"])})
        for im in imgs:
            events.append((im["bbox"][1], "image", im))
        events.sort(key=lambda e: (e[0], 0 if e[1] == "text" else 1))

        for y, kind, payload in events:
            if kind == "image":
                if entry is not None:
                    entry["_imgs"].append(payload)
                continue
            bbox, lines = payload
            first = lines[0]
            # 条目标题
            tm = TITLE_RE.match(first)
            if tm and "...." not in first and y >= first_title_y - 1:
                cn, latin = split_cn_latin(tm.group(2))
                if not latin:
                    # 缺字标题的拉丁名可能拆在同块续行
                    # ('263. 拟' + '属未定种Lophiodes sp.')
                    for t in lines[1:]:
                        m2 = re.search(r"[A-Za-z][A-Za-z\-\.\s]*$", t)
                        if m2:
                            latin = m2.group(0).strip()
                            break
                entry = {
                    "entry_no": tm.group(1),
                    "name_cn": cn,
                    "name_latin": latin,
                    "phylum": cur_phylum[0],
                    "phylum_cn": cur_phylum[1],
                    "class": cur_class[1],
                    "class_cn": cur_class[0],
                    "taxonomy": {},
                    "locality": "", "depth": "", "distribution": "",
                    "remark": "",
                    "page": page_no, "printed_page": printed,
                    "_imgs": [], "_y": y,
                }
                entries.append(entry)
                prev_is_taxo = False
                continue
            # 章节头判定: Class/Phylum/Subphylum 行若前一文本事件不是
            # '分类学地位', 即为纲级章节头 — 可在页中任意位置 (照片之后),
            # 如 p29 '六放海绵纲'/p139 '头足纲'/p207 '海参纲' 夹在两条目
            # 之间; 若紧跟 '分类学地位' 则属条目自身分类行 (#5/#6/#139/
            # #140/#179), 由下方逐行匹配归入条目。
            # Superclass 行在页首标题之前为章节分隔 (p214 '鱼总纲'),
            # 不得归入上一条目 (#245 曾因此混入 Pisces)。
            rm = RANK_RE.match(first)
            if rm:
                rank = rm.group(2)
                if rank in ("Class", "Phylum", "Subphylum") and not prev_is_taxo:
                    cur_class = (rm.group(1), rm.group(3))
                    headers.append((page_no, y, rm.group(1), rank, rm.group(3)))
                    prev_is_taxo = False
                    continue
                if rank == "Superclass" and y < first_title_y:
                    prev_is_taxo = False
                    continue
            prev_is_taxo = first.startswith("分类学地位")
            if entry is None:
                continue
            # 条目内的分类行: 逐行匹配 (缺字标题行与分类行可能同块,
            # 如 '263. 拟' + '属 Genus Lophiodes Goode & Bean, 1896')
            rank_keys = {"Subphylum": "subphylum", "Superclass": "superclass",
                         "Subclass": "subclass",
                         "Class": "class", "Order": "order",
                         "Suborder": "suborder", "Infraorder": "infraorder",
                         "Superfamily": "superfamily", "Family": "family",
                         "Subfamily": "subfamily", "Genus": "genus"}
            had_rank = False
            for t in lines:
                rm2 = RANK_RE.match(t)
                if rm2:
                    entry["taxonomy"][rank_keys[rm2.group(2)]] = {
                        "cn": rm2.group(1), "latin": rm2.group(3)}
                    had_rank = True
            if had_rank:
                continue
            fm = FIELD_RE.match(first)
            if fm:
                val = " ".join([fm.group(2)] + lines[1:]).strip()
                key = {"采集地": "locality", "水深": "depth",
                       "分布": "distribution", "生态": "remark"}[fm.group(1)]
                entry[key] = val

        # 图片图注: 紧贴图片下方的小文本块
        for im in imgs:
            ix0, iy0, ix1, iy1 = im["bbox"]
            caps = []
            for bbox, lines in blocks:
                by0, by1 = bbox[1], bbox[3]
                if iy1 + 1 <= by0 <= iy1 + 22 and bbox[0] >= ix0 - 5 \
                        and bbox[2] <= ix1 + 5:
                    joined = " ".join(lines)
                    if (TITLE_RE.match(joined) or RANK_RE.match(joined)
                            or FIELD_RE.match(joined)):
                        continue
                    caps.append(joined)
            im["caption"] = " ".join(caps)

    return entries, headers


def _pair_column(names, numbers):
    """同栏内名字行与页码行配对 (全局最近邻, dist<9pt)。
    返回 [(页码, 名)...] — 同一页码可有多个名称 (如同页两条目),
    不能用页码作 dict 键 (会互相覆盖)。
    未配对的名字行若紧邻某已配对行上方 (<14pt) 视为换行延续, 前置拼接。"""
    cands = []
    for i, (ny, _) in enumerate(names):
        for j, (py, _) in enumerate(numbers):
            d = abs(ny - py)
            if d < 9:
                cands.append((d, i, j))
    cands.sort()
    used_n, used_p, pairs = set(), set(), []
    for d, i, j in cands:
        if i in used_n or j in used_p:
            continue
        used_n.add(i)
        used_p.add(j)
        pairs.append((i, j))
    for i, j in pairs:
        # 向上收集换行延续行 (紧邻上方、未配对的行, 逐行前置拼接)
        cur = i
        while True:
            cand = [(k, ny, nm) for k, (ny, nm) in enumerate(names)
                    if k not in used_n and k < cur
                    and 0 < names[cur][0] - ny < 14]
            if not cand:
                break
            k, ny, nm = max(cand, key=lambda x: x[1])
            names[i] = (names[i][0], nm + " " + names[i][1])
            used_n.add(k)
            cur = k
    return [(numbers[j][1], names[i][1]) for i, j in pairs]


def _cluster_x(items, gap=60):
    """按 x0 升序把行聚类分栏 (相邻 x0 差 > gap 即新栏), 各栏内保持原序。"""
    cols = []
    for it in sorted(items, key=lambda it: it[0]):
        if cols and it[0] - cols[-1][-1][0] <= gap:
            cols[-1].append(it)
        else:
            cols.append([it])
    return cols


def parse_name_indexes(doc):
    """解析书尾索引: 中文名索引(p243-246) / 拉丁名索引(p247-251)。
    索引为多子栏排版, 每子栏 = 名称行 + 其右侧 ~128pt 处的页码行
    (中文名索引 3 子栏: 名称x≈65/221/377, 页码x≈193/349/505;
    拉丁名索引 2 子栏)。按名称/页码行的 x0 聚类分栏后逐栏配对;
    长名换行时拆成多行, 靠延续行拼接还原。
    返回 ({印刷页: [名...]}, {印刷页: [拉丁名...]})。"""
    cn_idx, latin_idx = {}, {}
    for pi in list(INDEX_CN_PAGES) + list(INDEX_LATIN_PAGES):
        target = cn_idx if pi in INDEX_CN_PAGES else latin_idx
        page = doc[pi - 1]
        names, nums = [], []  # (x0, y0, 值)
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for ln in b["lines"]:
                t = norm_text("".join(s["text"] for s in ln["spans"])).strip()
                if (not t or t in ("中文名索引", "拉丁名索引")
                        or "海山动物原色图谱" in t
                        or (len(t) == 1 and t.isalpha())):
                    continue
                x0, y0 = ln["bbox"][:2]
                if re.match(r"^\d+$", t):
                    if y0 <= 750:  # 排除页脚印刷页码
                        nums.append((x0, y0, int(t)))
                else:
                    names.append((x0, y0, re.sub(r"\s+", " ", t).strip()))
        ncols, pcols = _cluster_x(names), _cluster_x(nums)
        used = set()
        for col in ncols:
            # 页码栏在名称栏右侧: 取 x0 >= 名称栏起点的最近未用页码栏
            cand = [(pc[0][0], k) for k, pc in enumerate(pcols)
                    if k not in used and pc[0][0] >= col[0][0] - 5]
            pcol = []
            if cand:
                k = min(cand)[1]
                used.add(k)
                pcol = pcols[k]
            pairs = _pair_column([(y, t) for _, y, t in sorted(col, key=lambda i: i[1])],
                                 [(y, n) for _, y, n in sorted(pcol, key=lambda i: i[1])])
            for pageno, name in pairs:
                target.setdefault(pageno, []).append(name)
    return cn_idx, latin_idx


def _score(parsed, cand):
    """parsed 片段与候选名的匹配分 (0-1): 前缀覆盖优先, 其次相似度。"""
    if not parsed:
        return 0.5  # 空片段: 中性分
    p, c = parsed.lower(), cand.lower()
    if p and c.startswith(p):
        return 1.0 + len(p) / max(len(c), 1)
    return difflib.SequenceMatcher(None, p, c).ratio()


def correct_entries_with_index(entries, cn_idx, latin_idx):
    """用书尾索引按印刷页纠正中文名/拉丁名 (索引为权威来源)。"""
    fixed, flags = [], []
    for e in entries:
        e["name_raw"] = e["name_latin"]  # 原始解析名 (可能带命名人/缺字)
    by_page = {}
    for e in entries:
        by_page.setdefault(e["printed_page"], []).append(e)
    for pg, es in by_page.items():
        cns = list(cn_idx.get(pg, []))
        lats = list(latin_idx.get(pg, []))
        if not cns and not lats:
            flags.append((es[0]["entry_no"], pg, "索引无此页"))
            continue
        # 贪心两轮匹配: 先高分唯一者, 再剩余任意未占用候选
        used_c, used_l = set(), set()
        plans = []
        for e in es:
            cs = sorted(range(len(cns)), key=lambda i: -_score(e["name_cn"], cns[i]))
            ls = sorted(range(len(lats)), key=lambda i: -_score(e["name_latin"], lats[i]))
            plans.append((e, cs, ls))
        for rnd in range(2):
            for e, cs, ls in plans:
                if e.get("_ok"):
                    continue
                ci = next((i for i in cs if i not in used_c), None) if cns else None
                li = next((i for i in ls if i not in used_l), None) if lats else None
                if ci is None and li is None:
                    continue
                if rnd == 0 and cns and lats:
                    sc = _score(e["name_cn"], cns[ci]) if ci is not None else 0
                    sl = _score(e["name_latin"], lats[li]) if li is not None else 0
                    if len(es) > 1 and (sc < 0.45 or sl < 0.45):
                        continue  # 多条目页: 低分留待二轮
                if ci is not None:
                    used_c.add(ci)
                    # 索引行同样可能缺字 (字体缺字): 索引名与正文互为子串时
                    # 取信息量大的超串, 否则索引为准
                    if cns[ci] in e["name_cn"] or e["name_cn"] in cns[ci]:
                        e["name_cn"] = max(cns[ci], e["name_cn"], key=len)
                    else:
                        e["name_cn"] = cns[ci]
                if li is not None:
                    used_l.add(li)
                    e["name_latin"] = lats[li]
                e["_ok"] = True
        for e, _, _ in plans:
            e.pop("_ok", None)
        if len(es) == 1 or all(len(e["name_cn"]) > 1 for e in es):
            fixed.append(pg)
    return fixed, flags


def split_latin_name(latin):
    """拉丁名 -> (查询名, 推断阶元)。'Genus sp. 2'->('Genus','Genus');
    'Genus species'->(原名,'Species'); 'Gastropoda'->('Gastropoda',None)。
    先经 short_latin 清洗命名人/年份, 并剔除 cf./aff. 限定词
    ('Scytalium cf. sarsi'->'Scytalium sarsi')。"""
    t = short_latin(latin)
    toks = [w for w in t.split()
            if w.rstrip(".").lower() not in {"cf", "aff"}]
    if not toks:
        return "", None
    bad = {"sp", "spp", "indet", "nov"}
    while len(toks) > 1 and (
            toks[-1].lower().rstrip(".") in bad
            or re.match(r"^\d{1,2}$", toks[-1].rstrip("."))):
        toks = toks[:-1]
        if toks and toks[-1].lower().rstrip(".") in {"sp", "spp"}:
            toks = toks[:-1]
            break
    if len(toks) == 1:
        return toks[0], "Genus"
    return " ".join(toks), "Species"


def _worms_query(name):
    """单次 WoRMS 查询 (含限流重试) -> (records, err)。
    404/204/空响应/非JSON响应 = 未命中 (None, None) — WoRMS 对查不到的
    名字常返回 200 + HTML 空页而非 404, 此处统一按未命中处理。
    仅网络/HTTP 异常重试后仍失败时返回 (None, err)。"""
    url = ("https://www.marinespecies.org/rest/AphiaRecordsByName/"
           + urllib.parse.quote(name) + "?marine_only=false")
    err = None
    for _ in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "atlas-index/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
            try:
                return json.loads(body), None
            except ValueError:
                return None, None  # HTML/空 body = 未命中
        except urllib.error.HTTPError as ex:
            if ex.code in (204, 404):
                return None, None
            err = str(ex)
        except Exception as ex:
            err = str(ex)
        time.sleep(2.0)  # 限流/瞬时失败后重试
    return None, err


def fuzzy_suggest(name):
    """WoRMS 模糊匹配建议 (AphiaRecordsByMatchNames 自带 fuzzy, 可修
    OCR 错字如 Enallopsmmia->Enallopsammia); 失败时退回 GBIF match。"""
    url = ("https://www.marinespecies.org/rest/AphiaRecordsByMatchNames"
           "?scientificnames%5B%5D=" + urllib.parse.quote(name)
           + "&marine_only=false")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "atlas-index/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            recs = json.loads(r.read())
        if recs and recs[0]:
            return recs[0][0].get("scientificname")
    except Exception:
        pass
    return gbif_suggest(name)


def gbif_suggest(name):
    """GBIF 模糊匹配建议 (WoRMS 未命中时的拼写纠错参考)。"""
    url = ("https://api.gbif.org/v1/species/match?verbose=true&name="
           + urllib.parse.quote(name))
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "atlas-index/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
    except Exception:
        return None
    if d.get("scientificname") and d.get("matchType") not in (None, "NONE"):
        return d["scientificname"]
    for a in d.get("alternatives") or []:
        if a.get("matchType") not in (None, "NONE") and a.get("scientificname"):
            return a["scientificname"]
    return None


def worms_validate(entries):
    """WoRMS 逐一验证 (AphiaRecordsByName, marine_only=false), 带磁盘缓存。
    上次查询出错的条目视为未缓存并重试; 未命中时用 GBIF 模糊建议做
    拼写纠错 (与查询名差异极小 ratio>0.85 才自动采用, 如 OCR 错字)。"""
    cache = {}
    if os.path.exists(WORMS_CACHE):
        with open(WORMS_CACHE) as f:
            cache = json.load(f)
    n_retry = sum(1 for v in cache.values() if v.get("error"))
    cache = {q: v for q, v in cache.items() if not v.get("error")}
    queries = {}
    for e in entries:
        q, rank = split_latin_name(e["name_latin"])
        if q:
            queries.setdefault(q, rank)
    n_new = 0
    for q, rank in queries.items():
        if q in cache:
            continue
        recs, err = _worms_query(q)
        if err:
            cache[q] = {"error": err}
        elif not recs:
            sug = fuzzy_suggest(q)
            toks = q.split()
            if not sug and len(toks) == 2:
                # 双名未命中时属名可能本身错字: 属名单独 fuzzy 再组合
                gs = fuzzy_suggest(toks[0])
                if gs and gs.lower() != toks[0].lower():
                    sug = f"{gs} {toks[1]}"
            if sug and sug != q and difflib.SequenceMatcher(
                    None, q.lower(), sug.lower()).ratio() > 0.85:
                recs2, err2 = _worms_query(sug)
                if recs2:
                    r0 = recs2[0]
                    cache[q] = {
                        "match": r0.get("scientificname"),
                        "aphia_id": r0.get("AphiaID"),
                        "status": r0.get("status"),
                        "rank": r0.get("rank"),
                        "authority": r0.get("authority"),
                        "valid_name": r0.get("valid_name"),
                        "valid_authority": r0.get("valid_authority"),
                        "corrected_from": q, "gbif_suggest": sug,
                    }
                    print(f"  拼写纠正: {q} -> {sug} (GBIF 模糊建议)")
                else:
                    cache[q] = {"match": None}
            else:
                cache[q] = {"match": None}
        else:
            r0 = recs[0]
            cache[q] = {
                "match": r0.get("scientificname"),
                "aphia_id": r0.get("AphiaID"),
                "status": r0.get("status"),
                "rank": r0.get("rank"),
                "authority": r0.get("authority"),
                "valid_name": r0.get("valid_name"),
                "valid_authority": r0.get("valid_authority"),
            }
            if len(recs) > 1:
                cache[q]["alternates"] = [
                    {"match": x.get("scientificname"), "aphia_id": x.get("AphiaID"),
                     "status": x.get("status"), "rank": x.get("rank")}
                    for x in recs[1:4]]
        n_new += 1
        time.sleep(0.4)
    with open(WORMS_CACHE, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    print(f"WoRMS 查询: {len(queries)} 名 (新增 {n_new}, 缓存命中 {len(queries)-n_new})")
    hit = miss = err = 0
    for e in entries:
        q, rank = split_latin_name(e["name_latin"])
        r = cache.get(q, {})
        if r.get("error"):
            e["worms"] = {"query": q, "error": r["error"]}
            err += 1
        elif r.get("match"):
            e["worms"] = {"query": q, **r}
            hit += 1
            # GBIF 拼写纠正写回学名 (OCR 错字, 如 Enallopsmmia->Enallopsammia)
            if r.get("corrected_from") and r["match"] != q:
                e["name_latin"] = re.sub(
                    re.escape(q), r["match"], e["name_latin"], count=1)
        else:
            e["worms"] = {"query": q, "match": None}
            miss += 1
        # authority: 优先 WoRMS
        w = e.get("worms", {})
        if w.get("status") == "accepted":
            e["authority"] = w.get("authority") or ""
        elif w.get("valid_name"):
            e["authority"] = w.get("valid_authority") or ""
        else:
            e["authority"] = ""
    print(f"WoRMS 命中 {hit} / 未命中 {miss} / 错误 {err}")
    return cache


# taxonomy 键 -> 阶元中文后缀 / WoRMS rank 名
RANK_INFO = {
    "phylum": ("门", "Phylum"), "subphylum": ("亚门", "Subphylum"),
    "superclass": ("总纲", "Superclass"),
    "class": ("纲", "Class"), "subclass": ("亚纲", "Subclass"),
    "order": ("目", "Order"), "suborder": ("亚目", "Suborder"),
    "infraorder": ("次目", "Infraorder"), "superfamily": ("总科", "Superfamily"),
    "family": ("科", "Family"), "subfamily": ("亚科", "Subfamily"),
    "genus": ("属", "Genus"),
}


def validate_taxonomy(entries):
    """分类学地位检查 (在条目名纠正之后调用):
    1. 阶元中文名缺字补全 — 条目中文名 '<X>属未定种'/'<X>科未定种' 的
       前缀即阶元中文, 若阶元行中文是其残缺前缀 (字体缺字如 '拟') 则补全;
    2. 阶元拉丁名 WoRMS 验证 + fuzzy 拼写纠正 (OCR 错字写回);
    3. WoRMS 返回阶元与行标阶元不符时警告。"""
    cache = {}
    if os.path.exists(WORMS_CACHE):
        with open(WORMS_CACHE) as f:
            cache = json.load(f)
    # --- 1. 阶元中文名缺字补全 ---
    n_cn = 0
    for e in entries:
        for key, (suffix, _) in RANK_INFO.items():
            r = e["taxonomy"].get(key)
            if not r or not r.get("cn") or not e["name_cn"]:
                continue
            m = re.search(r"([\u4e00-\u9fff]+?" + suffix + ")未定种",
                          e["name_cn"])
            if m and m.group(1) != r["cn"] \
                    and r["cn"] in m.group(1):  # 子串=阶元行中文缺字残缺
                old = r["cn"]
                r["cn"] = m.group(1)
                r.setdefault("fixes", []).append(f"cn: {old} -> {r['cn']}")
                print(f"  阶元中文名补全: #{e['entry_no']} {key} "
                      f"{old} -> {r['cn']}")
                n_cn += 1
    # --- 2. 阶元拉丁名 WoRMS 验证 ---
    queries = {}
    for e in entries:
        for key, r in e["taxonomy"].items():
            if r.get("latin"):
                queries.setdefault(r["latin"], key)
    for lat in queries:
        if lat in cache:
            continue
        recs, err = _worms_query(lat)
        if err:
            cache[lat] = {"error": err}
        elif not recs:
            sug = fuzzy_suggest(lat)
            if sug and sug != lat and difflib.SequenceMatcher(
                    None, lat.lower(), sug.lower()).ratio() > 0.85:
                recs2, _ = _worms_query(sug)
                r0 = recs2[0] if recs2 else None
                # fuzzy 建议阶元必须与行标阶元一致, 否则拒绝
                # (如海葵总科 Actinoidea 被误配到属 Actinioides)
                exp_rank = RANK_INFO.get(queries[lat], (None, None))[1]
                if r0 and exp_rank and r0.get("rank") != exp_rank:
                    cache[lat] = {"match": None, "fuzzy_rejected": sug}
                    print(f"  fuzzy 建议拒绝: {lat} -> {sug} "
                          f"(rank={r0.get('rank')} != {exp_rank})")
                elif r0:
                    cache[lat] = {
                        "match": r0.get("scientificname"),
                        "aphia_id": r0.get("AphiaID"),
                        "status": r0.get("status"), "rank": r0.get("rank"),
                        "corrected_from": lat, "gbif_suggest": sug,
                    }
                    print(f"  阶元拼写纠正: {lat} -> {sug} (模糊建议)")
                else:
                    cache[lat] = {"match": None}
            else:
                cache[lat] = {"match": None}
        else:
            r0 = recs[0]
            cache[lat] = {
                "match": r0.get("scientificname"),
                "aphia_id": r0.get("AphiaID"), "status": r0.get("status"),
                "rank": r0.get("rank"), "authority": r0.get("authority"),
            }
        time.sleep(0.4)
    # 书中印刷错误 (如 'Pices'/'Psices' -> 'Pisces'): fuzzy 端点无建议时,
    # 与同阶元已验证名做近似匹配纠正
    rank_hits = {}
    for lat0, v0 in cache.items():
        if v0.get("match") and v0.get("rank"):
            rank_hits.setdefault(v0["rank"], []).append(lat0)
    for lat, key in queries.items():
        v = cache.get(lat)
        if not v or v.get("match") or v.get("error"):
            continue
        exp_rank = RANK_INFO.get(key, (None, None))[1]
        cands = [(difflib.SequenceMatcher(None, lat.lower(), h.lower()).ratio(), h)
                 for h in rank_hits.get(exp_rank, [])]
        if not cands:
            continue
        r_best, h_best = max(cands)
        if r_best > 0.8:
            cache[lat] = {**cache[h_best], "corrected_from": lat,
                          "gbif_suggest": h_best}
            print(f"  阶元拼写纠正: {lat} -> {h_best} (同阶元近似匹配)")
    with open(WORMS_CACHE, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    # --- 写回条目 ---
    n_fix = n_hit = n_miss = 0
    for e in entries:
        for key, r in e["taxonomy"].items():
            lat = r.get("latin")
            if not lat:
                continue
            w = cache.get(lat, {})
            if w.get("error"):
                r["worms"] = {"query": lat, "error": w["error"]}
                continue
            if w.get("match"):
                r["worms"] = {"query": lat, "match": w["match"],
                              "aphia_id": w.get("aphia_id"),
                              "status": w.get("status"),
                              "rank": w.get("rank")}
                n_hit += 1
                exp_rank = RANK_INFO[key][1]
                if w.get("rank") and w["rank"] != exp_rank:
                    print(f"  !! 阶元不符: #{e['entry_no']} {key} {lat} "
                          f"WoRMS rank={w['rank']} (期望 {exp_rank})")
                if w.get("corrected_from") and w["match"] != lat:
                    old = r["latin"]
                    r["latin"] = w["match"]
                    r.setdefault("fixes", []).append(
                        f"latin: {old} -> {w['match']}")
                    n_fix += 1
            else:
                r["worms"] = {"query": lat, "match": None}
                n_miss += 1
    print(f"分类学地位: 阶元中文名补全 {n_cn}, 拉丁名验证命中 {n_hit} / "
          f"未命中 {n_miss}, 拼写纠正 {n_fix}")


def dedup_and_save_images(doc, entries):
    """同名条目加序号, 提取图片到 <Phylum>/<名>/ 文件夹。"""
    name_count = {}
    total_saved = 0
    no_img = []
    for e in entries:
        latin = e["name_latin"] or e["name_cn"] or "Unnamed"
        base = sanitize_name(short_latin(latin) or latin)
        key = (e["phylum"], base.lower())
        name_count[key] = name_count.get(key, 0) + 1
        cnt = name_count[key]
        folder_name = base if cnt == 1 else f"{base} {cnt}"
        e["display_name"] = folder_name
        folder = os.path.join(OUT_ROOT, e["phylum"] or "Other", folder_name)
        os.makedirs(folder, exist_ok=True)
        e["images"] = []
        for k, im in enumerate(e.pop("_imgs"), 1):
            fname = f"{folder_name}_{k:03d}.{im['ext']}"
            path = os.path.join(folder, fname)
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(im["bytes"])
            total_saved += 1
            e["images"].append({
                "filename": fname,
                "path": os.path.relpath(path, BASE),
                "page": e["page"],
                "px": f"{im['px'][0]}x{im['px'][1]}",
                "caption": im.get("caption", ""),
            })
        if not e["images"]:
            no_img.append(e)
    return total_saved, no_img


def build_seamount_json(entries, total_images):
    by_phylum = {}
    for e in entries:
        by_phylum[e["phylum"]] = by_phylum.get(e["phylum"], 0) + 1
    return {
        "version": "1.0",
        "description": "西太平洋海沟洋脊交联区海山动物原色图谱 (徐奎栋等, 科学出版社) 结构化索引",
        "source": {
            "pdf": os.path.basename(PDF_PATH),
            "title": "西太平洋海沟洋脊交联区海山动物原色图谱",
            "authors": "徐奎栋 等 著",
            "press": "科学出版社",
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "total_entries": len(entries),
            "total_images": total_images,
            "image_assignment_rule": "图片归属阅读顺序中其之前最近的条目标题 (已目检验证)",
        },
        "statistics": dict(
            total_entries=len(entries),
            total_images=total_images,
            by_phylum=by_phylum,
        ),
        "entries": [
            {
                "entry_no": e["entry_no"],
                "name_cn": e["name_cn"],
                "scientific_name": e["name_latin"],
                "authority": e.get("authority", ""),
                "name_raw": e.get("name_raw", ""),
                "display_name": e["display_name"],
                "morphotype_id": f"SM_{int(e['entry_no']):03d}",
                "phylum": e["phylum"], "phylum_cn": e["phylum_cn"],
                "class": (e["taxonomy"].get("class", {})
                          or {"latin": e["class"]}).get("latin", e["class"]),
                "taxonomy": e["taxonomy"],
                "class_context": e["class"],
                "worms": {k: v for k, v in e.get("worms", {}).items()},
                "locality": e["locality"], "depth": e["depth"],
                "distribution": e["distribution"], "remark": e["remark"],
                "pdf_page": e["page"], "printed_page": e["printed_page"],
                "images": e["images"],
            } for e in entries
        ],
    }


def backup(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)


def merge_into_index(entries):
    """合并到 index.json: entries 追加, by_phylum/by_class/statistics 重建。"""
    backup(INDEX_PATH)
    with open(INDEX_PATH) as f:
        ix = json.load(f)
    ix["entries"] = [e for e in ix["entries"]
                     if e.get("source") != "海山动物原色图谱"]
    for e in entries:
        imgs = e["images"]
        ix["entries"].append({
            "morphotype_id": f"SM_{int(e['entry_no']):03d}",
            "scientific_name": e["name_latin"],
            "name_cn": e["name_cn"],
            "phylum": e["phylum"], "phylum_cn": e["phylum_cn"],
            "class": e["class"] or (e["taxonomy"].get("class", {}) or {})
            .get("latin", ""),
            "locality": e["locality"], "depth": e["depth"],
            "distribution": e["distribution"],
            "source": "海山动物原色图谱",
            "image_count": len(imgs),
            "images": [{"filename": i["filename"], "path": i["path"]}
                       for i in imgs],
            "directory": os.path.dirname(imgs[0]["path"]) if imgs else
            os.path.relpath(os.path.join(
                OUT_ROOT, e["phylum"] or "Other", e["display_name"]), BASE),
            "pages": sorted({i["page"] for i in imgs}) or [e["page"]],
        })
    by_phylum, by_class = {}, {}
    for e in ix["entries"]:
        by_phylum.setdefault(e["phylum"], {"entry_count": 0, "image_count": 0})
        by_phylum[e["phylum"]]["entry_count"] += 1
        by_phylum[e["phylum"]]["image_count"] += e.get("image_count", 0)
        cls = e.get("class") or "未定"
        by_class.setdefault(cls, {"entry_count": 0, "image_count": 0})
        by_class[cls]["entry_count"] += 1
        by_class[cls]["image_count"] += e.get("image_count", 0)
    ix["by_phylum"] = by_phylum
    ix["by_class"] = by_class
    total_imgs = sum(e.get("image_count", 0) for e in ix["entries"])
    ix["statistics"] = {
        "total_entries": len(ix["entries"]),
        "total_images": total_imgs,
        "total_morphotypes": len(ix.get("by_morphotype", {})),
        "phyla": sorted(by_phylum.keys()),
        "sources": ["北太平洋图集(APSMA完整版)", "西太平洋图集", "海山动物原色图谱"],
    }
    ix["updated"] = datetime.now().isoformat(timespec="seconds")
    with open(INDEX_PATH, "w") as f:
        json.dump(ix, f, ensure_ascii=False, indent=2)


def merge_into_species_index(sm_json):
    """合并到 species_name_index.json: 新增 seamount_entries 键。"""
    backup(SPECIES_JSON)
    with open(SPECIES_JSON) as f:
        sp = json.load(f)
    sp["seamount_entries"] = sm_json["entries"]
    n_img = sm_json["statistics"]["total_images"]
    n_ent = sm_json["statistics"]["total_entries"]
    sp["sources"]["SeamountAtlas"] = {
        "pdf": sm_json["source"]["pdf"],
        "title": sm_json["source"]["title"],
        "authors": sm_json["source"]["authors"],
        "entries": n_ent,
        "total_images": n_img,
        "images_with_species": n_img,
        "images_without_species": 0,
    }
    np_img = sp["sources"].get("APSMA", {}).get("total_images", 0)
    wp_img = sp["sources"].get("WestPacific", {}).get("total_images", 0)
    # 物种级统计: westpac_entries 是图片级记录 (每图一条), 须按唯一物种名去重;
    # 不用 sources 元数据 (species_extracted 可能是旧值)
    n_apsma = len(sp.get("apsma_entries") or [])
    wp_list = sp.get("westpac_entries") or []
    n_wp = len({e.get("scientific_name") or e.get("name_cn") for e in wp_list})
    np_m = n_apsma
    sp["statistics"] = {
        "total_entries": n_apsma + n_wp + len(sp["seamount_entries"]),
        "total_morphotypes": np_m,
        "total_images": np_img + wp_img + n_img,
        "scientific_name_coverage": (
            f"{np_m}/{np_m} APSMA + {wp_img}/{wp_img} WestPacific "
            f"+ {n_ent}/{n_ent} SeamountAtlas"),
    }
    with open(SPECIES_JSON, "w") as f:
        json.dump(sp, f, ensure_ascii=False, indent=2)


def main():
    doc = pymupdf.open(PDF_PATH)
    print(f"PDF: {len(doc)} 页, 处理 p{PAGE_START}-p{PAGE_END}")
    entries, headers = parse_pages(doc)
    print(f"解析条目: {len(entries)} 个; 纲级章节头: {len(headers)} 个")
    nums = sorted(int(e["entry_no"]) for e in entries)
    gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in set(nums)]
    dup = [n for n in set(nums) if nums.count(n) > 1]
    print(f"编号范围 {nums[0]}..{nums[-1]}, 缺号: {gaps}, 重号: {dup}")
    # 书尾索引纠正 (索引为权威来源, 修复文本层缺字/错误)
    cn_idx, latin_idx = parse_name_indexes(doc)
    print(f"书尾索引: 中文名 {sum(len(v) for v in cn_idx.values())} 条, "
          f"拉丁名 {sum(len(v) for v in latin_idx.values())} 条")
    corrected, flags = correct_entries_with_index(entries, cn_idx, latin_idx)
    print(f"索引纠正: {len(corrected)} 页命中; 异常: {flags if flags else '无'}")
    # 人工核实修正 (文本层与索引均缺字、无法自动恢复)
    for e in entries:
        fix = MANUAL_CN_FIX.get(str(e["entry_no"]))
        if fix:
            e["name_cn"] = fix
            print(f"人工核实修正: #{e['entry_no']} -> {fix}")
    bad = [(e["entry_no"], e["name_cn"], e["name_latin"]) for e in entries
           if len(e["name_cn"]) <= 1 or not e["name_latin"]]
    if bad:
        print("!! 纠正后仍有残缺名字:", bad)
    # 统一清洗学名: 去命名人/年份 (索引名一般已干净, 正文名可能带)
    for e in entries:
        if e["name_latin"]:
            e["name_latin"] = short_latin(e["name_latin"])
    # WoRMS 验证 + authority 提取
    worms_validate(entries)
    # 分类学地位检查: 阶元拉丁名验证/拼写纠正, 阶元中文名缺字补全
    validate_taxonomy(entries)
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    total_saved, no_img = dedup_and_save_images(doc, entries)
    print(f"提取图片: {total_saved} 张 -> {OUT_ROOT}")
    if no_img:
        print("无图片条目:", [(e["entry_no"], e["name_cn"]) for e in no_img])
    for h in headers:
        print("  章节头:", h)
    sm = build_seamount_json(entries, total_saved)
    with open(SEAMOUNT_JSON, "w") as f:
        json.dump(sm, f, ensure_ascii=False, indent=2)
    print(f"独立索引: {SEAMOUNT_JSON}")
    merge_into_index(entries)
    merge_into_species_index(sm)
    print("已合并到 index.json 和 species_name_index.json (原文件备份为 .bak)")


if __name__ == "__main__":
    main()
