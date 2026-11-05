#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北太平洋图集 (Benthic Megafauna from the North Pacific Abyss) 结构化整理 v2

修复问题:
  1. 图片-物种错配: 旧脚本用整页首行匹配形态型ID, 把整页图片全归给第一个条目。
     新方案: 块级阅读顺序分割, 每页每个条目一个纵向区域, 图片按区域归属。
  2. 不再使用 DES_002 等缩写作为物种标识/文件夹名, 改用实际物种名(同名加序号)。
  3. 描述保留 Morphology/Notes, 但剥离 DOI/文献引用等不需要的内容。

版面结构 (每页通常上下两个条目):
  [ID 名称] / [图片组] / [Morphology: ...] / [Notes: ...] / 下一个条目...
名称可能与ID同块, 也可能在紧随其后的独立文本块中 (ACT/ALC/PEN等章节)。
分类上下文由章节分隔页 (Phylum/Class/Order 行) 跟踪。
"""
import os
import re
import sys
import json
import shutil

import pymupdf  # fitz

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE, "data/raw/生物图集",
                        "Benthic Megafauna from the North Pacific Abyss 2023-.pdf")
OUT_ROOT = os.path.join(BASE, "data/processed/atlas_lookup/北太平洋图集")
INDEX_PATH = os.path.join(BASE, "data/processed/atlas_lookup/north_pacific_species_index.json")

TITLE_RE = re.compile(r"^([A-Z]{3}_\d{3})\s*(.*)$")
DESC_RE = re.compile(r"^(Morphology|Notes|Note|Velum)\s*:\s*(.*)$", re.S)
HIER_RE = re.compile(r"^(Phylum|SubPhylum|SubPylum|Class|Order)\s*:?\s*([A-Za-z\-]+)\s*$",
                     re.IGNORECASE)
DOI_RE = re.compile(r"doi\.org/\S+|doi:\S+")

MIN_IMG_BYTES = 5000
MIN_IMG_W = 100
MIN_IMG_H = 100


def merge_block_lines(block):
    """把块内文本行合并为单行文本, 处理行尾连字符断词。"""
    lines = []
    for ln in block["lines"]:
        lines.append("".join(s["text"] for s in ln["spans"]).strip())
    out = ""
    for ln in lines:
        if not ln:
            continue
        if out.endswith("-"):
            out = out[:-1] + ln  # 断词拼接: "de-" + "tail" -> "detail"
        elif out:
            out = out + " " + ln
        else:
            out = ln
    return out.strip()


def clean_desc(t, dropped=None):
    """剥离文献引用(DOI/More detail), 保留形态学描述本体。

    dropped: 可选列表, 记录被删内容供人工核查。
    """
    # 1) DOI token 内部的换行空格归一 ("10.1016 /j.dsr2..." -> "10.1016/j.dsr2...")
    t = re.sub(r"doi\s*:\s*", "doi:", t)
    t = re.sub(r"doi\.org\s*/\s*", "doi.org/", t)
    t = re.sub(r"(10\.\d{3,5})\s*/\s*", r"\1/", t)
    t = re.sub(r"(?<=\d)\s+/\s*(?=[A-Za-z0-9])", "/", t)

    # 2) "see [also] [:] [e.g.] <doi片段>" 引用尾巴 (不碰 "see ACT_093" 等图集内部互引)
    m2 = re.search(r"[,;]?\s*\bsee\b(?:\s+also)?\s*:?\s*(?:e\.g\.\s*)?"
                   r"(?:doi\.org/[\w.\-]+|doi:[\w.\-]+|/[\w.\-]+|10\.\d{4,5}/[\w.\-]+)", t)
    if m2 and dropped is not None:
        dropped.append(m2.group(0))
    t = re.sub(r"[,;]?\s*\bsee\b(?:\s+also)?\s*:?\s*(?:e\.g\.\s*)?"
               r"(?:doi\.org/[\w.\-]+|doi:[\w.\-]+|/[\w.\-]+|10\.\d{4,5}/[\w.\-]+)",
               "", t, flags=re.I)

    # 3) 含 DOI/More detail 的括号引用整体删除 ("(e.g. /j.dsr2...; time-lapse)" 等)
    def _drop_cite_paren(m):
        inner = m.group(0)
        if (re.search(r"doi|10\.\d{4,5}|[Mm]ore\s+detail", inner)
                or ("/" in inner and re.search(r"\d", inner))):
            if dropped is not None:
                dropped.append(inner)
            return ""
        return inner
    t = re.sub(r"\([^()]*\)", _drop_cite_paren, t)

    # 4) 括号式引用: (More detail[s]: …)
    t = re.sub(r"\(\s*[Mm]ore\s+details?\s*:\s*[^()]*\)", "", t)
    # 5) 非括号式: More detail[s]: doi… / 10…
    m5 = re.search(r"\.?\s*[Mm]ore\s+details?\s*:\s*(?:doi\.org/|doi:)?[\w.\-/]*", t)
    if m5 and dropped is not None:
        dropped.append(m5.group(0))
    t = re.sub(r"\.?\s*[Mm]ore\s+details?\s*:\s*(?:doi\.org/|doi:)?[\w.\-/]*", "", t)
    # 6) 独立 DOI token + 断行残留的 DOI 片段 ("/s12526-015-0340-x" 等)
    t = re.sub(r"doi\.org/[\w.\-]+|doi:[\w.\-]+|10\.\d{4,5}/[\w.\-]+", "", t)
    t = re.sub(r"[,;]?\s*/[A-Za-z][\w]*(?:-[\w.]+)+", "", t)
    t = re.sub(r"\(\s*e\.g\.\s*;\s*", "(e.g. ", t)
    t = re.sub(r"\(\s*see\s*\)|\(\s*e\.g\.\s*\)|\(\s*\)", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([.,;])", r"\1", t)
    t = re.sub(r"\.{2,}", ".", t)
    return t.strip().rstrip(" ;").strip()


def sanitize_name(name):
    """物种名转文件夹安全名。"""
    s = re.sub(r'[\\/:*?"<>|]+', " ", name)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s


def update_context_from_lines(lines, ctx):
    """从分隔页文本行更新分类上下文。返回是否发生(protected)修改。"""
    pending_phylum = False
    for ln in lines:
        ln = ln.strip().rstrip(":").strip()
        if not ln:
            continue
        if pending_phylum:  # "Phylum" 与值分处两行 (p237: Phylum / Chaetognatha)
            if re.match(r"^[A-Za-z\-]+$", ln):
                ctx["phylum"] = ln
                ctx["subphylum"] = ""
                ctx["class"] = ""
                ctx["order"] = ""
            pending_phylum = False
            continue
        m = HIER_RE.match(ln)
        if m:
            rank, val = m.group(1).lower().replace("plylum", "phylum"), m.group(2)
            if rank == "phylum":
                pending_phylum = ln.rstrip(":").strip().lower().endswith("phylum")
                if not pending_phylum:
                    ctx["phylum"] = val
                    ctx["subphylum"] = ""
                    ctx["class"] = ""
                    ctx["order"] = ""
            elif rank in ("subphylum",):
                ctx["subphylum"] = val
                ctx["class"] = ""
                ctx["order"] = ""
            elif rank == "class":
                ctx["class"] = val
                ctx["order"] = ""
            elif rank == "order":
                ctx["order"] = val


def parse_pdf():
    doc = pymupdf.open(PDF_PATH)
    ctx = {"phylum": "", "subphylum": "", "class": "", "order": ""}
    entries = []

    for pi in range(len(doc)):
        page = doc[pi]
        d = page.get_text("dict")
        page_h = page.rect.height

        text_blocks = []
        for b in d["blocks"]:
            if b["type"] == 0:
                txt = merge_block_lines(b)
                if txt:
                    text_blocks.append({"bbox": b["bbox"], "text": txt})

        # 图片对象用 get_image_info (dict 会漏报/合并重叠图片)
        img_boxes = []
        seen_xrefs = set()
        for info in page.get_image_info(xrefs=True):
            x0, y0, x1, y1 = info["bbox"]
            if (x1 - x0) < 40 or (y1 - y0) < 40:
                continue  # 装饰性小元素
            if info["width"] < 80 or info["height"] < 80:
                continue
            if info["xref"] in seen_xrefs:
                continue
            seen_xrefs.add(info["xref"])
            img_boxes.append({"bbox": info["bbox"], "xref": info["xref"],
                              "px_w": info["width"]})

        titles = []
        for tb in text_blocks:
            m = TITLE_RE.match(tb["text"])
            if m and len(tb["text"]) < 200:
                titles.append({"bbox": tb["bbox"], "id": m.group(1),
                               "inline_name": m.group(2).strip()})
        titles.sort(key=lambda t: t["bbox"][1])

        # ---- 上下文更新: 非标题/非描述/非名称候选的层级行 ----
        if titles:
            # 同页分隔+条目 (如 p237): 只取第一个标题上方的层级行
            first_title_y = titles[0]["bbox"][1]
            hier_lines = []
            for tb in text_blocks:
                if DESC_RE.match(tb["text"]) or TITLE_RE.match(tb["text"]):
                    continue
                if tb["bbox"][1] < first_title_y:
                    hier_lines.extend(tb["text"].split("\n"))
        else:
            if any(DESC_RE.match(tb["text"]) for tb in text_blocks):
                hier_lines = []  # 条目页但标题漏检, 不动上下文
            else:
                hier_lines = []
                for tb in text_blocks:
                    hier_lines.extend(tb["text"].split("\n"))
        if hier_lines:
            if not titles:
                before = dict(ctx)
                update_context_from_lines(hier_lines, ctx)
                # 卷首封面页(同时列出多个门)不动上下文: 层级行>3时回滚
                n_hits = sum(1 for ln in hier_lines if HIER_RE.match(ln.strip().rstrip(":")))
                if n_hits > 3:
                    ctx.update(before)
                if "(non Decapoda)" in " ".join(hier_lines) or \
                   re.search(r"\(non\s+Decapoda\)", " ".join(hier_lines)):
                    ctx["order"] = ""
            else:
                # 条目页顶部的分隔行 (Phylum/Chaetognatha 分行等)
                update_context_from_lines(hier_lines, ctx)

        if not titles:
            continue

        # ---- 描述块归属: 最近的上方标题 ----
        for tb in text_blocks:
            m = DESC_RE.match(tb["text"])
            if not m:
                continue
            y0 = tb["bbox"][1]
            owner = None
            for t in titles:
                if t["bbox"][1] <= y0:
                    owner = t
            if owner is None:
                owner = titles[0]
            owner.setdefault("descs", []).append((m.group(1), m.group(2), tb["bbox"]))

        # ---- 名称解析 ----
        for k, t in enumerate(titles):
            name = t["inline_name"]
            if not name:
                # 名称在紧随其后的独立文本块: 第一个非描述/非标题/非纯数字的短块
                next_y = titles[k + 1]["bbox"][1] if k + 1 < len(titles) else page_h
                cands = []
                for tb in text_blocks:
                    y0 = tb["bbox"][1]
                    if t["bbox"][3] <= y0 < next_y and tb is not t:
                        if DESC_RE.match(tb["text"]) or TITLE_RE.match(tb["text"]):
                            continue
                        if re.match(r"^\d+$", tb["text"]):
                            continue
                        if len(tb["text"]) < 150:
                            cands.append((y0, tb["text"]))
                if cands:
                    name = min(cands)[1]
            t["name"] = name.strip()
            t["page"] = pi + 1

        # ---- 区域边界与图片归属 ----
        n = len(titles)
        bounds = [0.0]
        for k in range(n - 1):
            desc_bottom = titles[k]["bbox"][3]
            for _, _, bb in titles[k].get("descs", []):
                desc_bottom = max(desc_bottom, bb[3])
            bounds.append((desc_bottom + titles[k + 1]["bbox"][1]) / 2.0)
        bounds.append(page_h + 1)

        assigned = [[] for _ in range(n)]
        for ib in img_boxes:
            # 出血图顶边可能为负(如p41/p107/p252棋盘版面), 夹紧后按区域归属,
            # 否则负值图片落入兜底分支被最后一个条目整页吞掉
            y0 = min(max(ib["bbox"][1], 0.0), float(page_h))
            # 用图片顶边归属: 条目照片总在其标题之下、描述之上,
            # 可延伸越过下一条目标题(如p230右下大图), 顶边判定比中心点稳
            for k in range(n):
                if bounds[k] <= y0 < bounds[k + 1]:
                    assigned[k].append(ib)
                    break
            else:
                assigned[n - 1].append(ib)

        for k, t in enumerate(titles):
            t["images"] = assigned[k]
            t["ctx_phylum"] = ctx["phylum"]
            t["ctx_subphylum"] = ctx["subphylum"]
            t["ctx_class"] = ctx["class"]
            t["ctx_order"] = ctx["order"]
            entries.append(t)

    doc.close()
    return entries


def extract_entry_images(doc, page_idx, img_boxes, out_prefix):
    """按条目提取图片, 处理切片合并。返回保存文件列表。"""
    page = doc[page_idx]
    items = sorted(img_boxes, key=lambda s: (s["bbox"][1], s["bbox"][0]))
    groups = []
    i = 0
    while i < len(items):
        g = [items[i]]
        j = i + 1
        while j < len(items):
            a, b = g[-1]["bbox"], items[j]["bbox"]
            same_x = min(a[2], b[2]) - max(a[0], b[0]) > 0.9 * min(a[2] - a[0], b[2] - b[0])
            if same_x and -1 < b[1] - a[3] < 3 and abs((a[2] - a[0]) - (b[2] - b[0])) < 3:
                g.append(items[j])
                j += 1
            else:
                break
        groups.append(g)
        i = j

    os.makedirs(os.path.dirname(out_prefix + "_001.jpeg"), exist_ok=True)
    saved = []
    for g in groups:
        idx = len(saved) + 1
        if len(g) == 1:
            info = g[0]
            fn_base = f"{out_prefix}_{idx:03d}"
            ok = False
            try:
                raw = doc.extract_image(info["xref"])
                if raw["width"] >= MIN_IMG_W and raw["height"] >= MIN_IMG_H:
                    ext = raw["ext"] if raw["ext"] in ("jpeg", "jpg", "png") else "jpeg"
                    fn = fn_base + "." + ext
                    with open(fn, "wb") as f:
                        f.write(raw["image"])
                    saved.append(fn)
                    ok = True
            except Exception:
                pass
            if not ok:
                x0, y0, x1, y1 = info["bbox"]
                scale = max(600 / max(x1 - x0, 1), 300 / max(y1 - y0, 1))
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                      clip=pymupdf.Rect(x0, y0, x1, y1))
                fn = fn_base + ".jpeg"
                pix.save(fn, jpg_quality=92)
                saved.append(fn)
        else:
            # 切片组: 联合区域重渲染
            ux0 = min(b["bbox"][0] for b in g); uy0 = min(b["bbox"][1] for b in g)
            ux1 = max(b["bbox"][2] for b in g); uy1 = max(b["bbox"][3] for b in g)
            ref_px = max(b.get("px_w", 0) for b in g)
            scale = ref_px / (ux1 - ux0) if ref_px else 600 / max(ux1 - ux0, 1)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                  clip=pymupdf.Rect(ux0, uy0, ux1, uy1))
            fn = f"{out_prefix}_{idx:03d}.jpeg"
            pix.save(fn, jpg_quality=92)
            saved.append(fn)
    return saved


def main():
    if os.path.isdir(OUT_ROOT):
        print(f"[CLEAN] 清空旧目录 {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    os.makedirs(OUT_ROOT, exist_ok=True)

    doc = pymupdf.open(PDF_PATH)
    entries = parse_pdf()
    print(f"[OK] 解析条目: {len(entries)}")

    # 同名条目按文档顺序编号 (键用清洗后的名字, 避免"双空格"变体共用文件夹)
    name_count = {}
    for e in entries:
        base = e["name"] or "Unnamed"
        key = sanitize_name(base)
        name_count[key] = name_count.get(key, 0) + 1
        cnt = name_count[key]
        e["display_name"] = base if cnt == 1 else f"{base} {cnt}"

    index_entries = []
    ctx_by_page = {}  # 重新扫描一次保存每个条目页的上下文? 简化: parse时已并入entry

    # parse_pdf 中未返回 ctx 快照, 这里按条目存储的 ctx 字段使用
    problems = []
    dropped_all = []
    for e in entries:
        phyl = e.get("ctx_phylum", "")
        folder_name = sanitize_name(e["display_name"])
        rel_dir = os.path.join(phyl or "Unknown", folder_name)
        abs_dir = os.path.join(OUT_ROOT, rel_dir)
        prefix = os.path.join(abs_dir, folder_name)
        saved = extract_entry_images(doc, e["page"] - 1, e["images"], prefix)
        if not saved:
            problems.append(f"无图片: p{e['page']} {e['id']} {e['display_name']}")

        morphology, notes = [], []
        for kind, body, _ in e.get("descs", []):
            cleaned = clean_desc(body, dropped_all)
            if not cleaned:
                continue
            if kind in ("Morphology", "Velum"):
                morphology.append(cleaned)
            else:
                notes.append(cleaned)

        index_entries.append({
            "morphotype_id": e["id"],
            "name": e["display_name"],
            "source_name": e["name"],
            "folder": rel_dir,
            "page": e["page"],
            "phylum": e.get("ctx_phylum", ""),
            "subphylum": e.get("ctx_subphylum", ""),
            "class": e.get("ctx_class", ""),
            "order": e.get("ctx_order", ""),
            "morphology": " ".join(morphology),
            "notes": " ".join(notes),
            "images": [{
                "filename": os.path.basename(p),
                "path": os.path.join("data/processed/atlas_lookup/北太平洋图集",
                                     rel_dir, os.path.basename(p)),
                "page": e["page"],
            } for p in saved],
        })

    doc.close()

    # 统计
    from collections import Counter
    phyl_stat = Counter(x["phylum"] for x in index_entries)
    n_img = sum(len(x["images"]) for x in index_entries)
    no_name = [x for x in index_entries if not x["source_name"]]
    no_desc = [x for x in index_entries if not x["morphology"] and not x["notes"]]

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "version": "2.0",
            "description": "北太平洋深渊底栖生物图集结构化索引 (区域分割版, 物种名为文件夹)",
            "source_pdf": os.path.relpath(PDF_PATH, BASE),
            "statistics": {
                "entries": len(index_entries),
                "images": n_img,
                "phylums": dict(phyl_stat),
            },
            "entries": index_entries,
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] 条目 {len(index_entries)}, 图片 {n_img}")
    print(f"[OK] 门类分布: {dict(phyl_stat)}")
    print(f"[WARN] 无名称条目: {len(no_name)}")
    for x in no_name[:10]:
        print(f"   - p{x['page']} {x['morphotype_id']}")
    print(f"[WARN] 无描述条目: {len(no_desc)}")
    for x in no_desc[:10]:
        print(f"   - p{x['page']} {x['morphotype_id']} {x['name']}")
    for p in problems[:10]:
        print(f"[WARN] {p}")
    print(f"[OK] 描述中删除的引用片段 ({len(dropped_all)} 处):")
    for x in dropped_all[:30]:
        print(f"   - {x!r}")
    print(f"[OK] 索引: {INDEX_PATH}")


if __name__ == "__main__":
    main()
