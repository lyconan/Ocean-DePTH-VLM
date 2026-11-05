#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把新版北太平洋图集索引 (north_pacific_species_index.json, 物种名版) 合并进:
  1. species_name_index.json  — 替换旧的 apsma_entries (186 条 ID 版)
  2. index.json               — 替换 source='北太平洋图集(APSMA完整版)' 的旧条目,
                                并重建 by_phylum / by_morphotype / by_class / statistics

西太平洋内容 (westpac_entries / westpac_species_list / 西太平洋图集条目) 保持不变。
覆盖前生成 .bak 备份。
"""
import json
import os
import re
import shutil
from collections import Counter, OrderedDict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOKUP = os.path.join(BASE, "data/processed/atlas_lookup")
NP_INDEX = os.path.join(LOOKUP, "north_pacific_species_index.json")
SPECIES_INDEX = os.path.join(LOOKUP, "species_name_index.json")
FULL_INDEX = os.path.join(LOOKUP, "index.json")

NP_SOURCE = "北太平洋图集(APSMA完整版)"
IMG_ROOT = "data/processed/atlas_lookup/北太平洋图集"

PHYLUM_CN = {
    "Porifera": "多孔动物门",
    "Cnidaria": "刺胞动物门",
    "Echinodermata": "棘皮动物门",
    "Arthropoda": "节肢动物门",
    "Ctenophora": "栉水母动物门",
    "Bryozoa": "苔藓动物门",
    "Annelida": "环节动物",
    "Nemertea": "纽形动物",
    "Chaetognatha": "毛颚动物",
    "Platyhelminthes": "扁形动物",
    "Mollusca": "软体动物",
    "Brachiopoda": "腕足动物",
    "Hemichordata": "半索动物",
    "Chordata": "脊索动物",
}

# 从学名推测属名: "Axoniderma mexicana sp. inc." -> Axoniderma
# 属级以上阶元名开头 (Rhopalonematidae gen. indet. / Stolidobranchia fam. indet.) -> ""
NON_GENUS = {"indet", "cf", "aff", "order", "fam", "gen", "subsp", "inc",
             "superf", "subclass", "class"}


def guess_genus(name):
    toks = name.split()
    if len(toks) < 2:
        return ""
    t0, t1 = toks[0], re.sub(r"[.,]", "", toks[1])
    if not re.match(r"^[A-Z][a-z\-]+$", t0):
        return ""
    if t1 in NON_GENUS:
        return ""
    if re.match(r"^[a-z]{2,}$", t1) or t1 == "sp":
        return t0
    return ""


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def backup(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"[BAK] {bak}")


def main():
    np = load_json(NP_INDEX)
    np_entries = np["entries"]
    n_img = sum(len(e["images"]) for e in np_entries)
    # 图集原文存在 ID 复用 (AST_012/HOL_108 各出现2次, 不同物种), 统计重复
    dup_ids = sorted(k for k, v in Counter(e["morphotype_id"] for e in np_entries).items() if v > 1)
    print(f"[OK] 北太平洋新索引: {len(np_entries)} 条目, {n_img} 图片, 复用ID: {dup_ids}")

    # ---------- 1. species_name_index.json ----------
    backup(SPECIES_INDEX)
    sp = load_json(SPECIES_INDEX)
    new_apsma = []
    for e in np_entries:
        morphology = e.get("morphology", "")
        notes = e.get("notes", "")
        full_desc = " ".join(x for x in (morphology, notes) if x)
        new_apsma.append({
            "morphotype_id": e["morphotype_id"],
            "scientific_name": e["name"],
            "phylum": e["phylum"],
            "phylum_cn": PHYLUM_CN.get(e["phylum"], ""),
            "subphylum": e.get("subphylum", ""),
            "class": e["class"],
            "order": e["order"],
            "family": "",
            "genus": guess_genus(e["name"]),
            "description": morphology,
            "notes": notes,
            "description_from_pdf": full_desc,
            "image_count": len(e["images"]),
            "images": [{"filename": i["filename"], "path": i["path"]}
                       for i in e["images"]],
            "page": e["page"],
            "folder": e["folder"],
            "source": NP_SOURCE,
        })
    sp["apsma_entries"] = new_apsma

    # westpac_entries 每条即为一张图 (含 filename 字段)
    n_wp_img = len(sp["westpac_entries"])
    sp["sources"]["APSMA"] = {
        "pdf": "Benthic Megafauna from the North Pacific Abyss 2023-.pdf",
        "note": "v2 物种名版: 条目按物种命名并同名加序号, 图片按区域归属",
        "morphotypes": len(new_apsma),
        "with_scientific_name": sum(1 for e in new_apsma if e["scientific_name"]),
        "without_scientific_name": sum(1 for e in new_apsma if not e["scientific_name"]),
        "total_images": n_img,
    }
    # sources.WestPacific 校正为当前实际值 (旧值368为条带合并重提取前数量)
    sp["sources"]["WestPacific"]["total_images"] = n_wp_img
    sp["sources"]["WestPacific"]["images_with_species"] = n_wp_img
    sp["sources"]["WestPacific"]["images_without_species"] = 0
    sp["statistics"] = {
        "total_entries": len(new_apsma) + len(sp["westpac_entries"]),
        "total_morphotypes": len(new_apsma),
        "total_images": n_img + n_wp_img,
        "scientific_name_coverage":
            f"{sum(1 for e in new_apsma if e['scientific_name'])}/{len(new_apsma)} APSMA"
            f" + {n_wp_img}/{n_wp_img} WestPacific",
    }
    with open(SPECIES_INDEX, "w", encoding="utf-8") as f:
        json.dump(sp, f, ensure_ascii=False, indent=2)
    print(f"[OK] species_name_index.json: apsma_entries -> {len(new_apsma)}, "
          f"westpac_entries -> {len(sp['westpac_entries'])} (不变)")

    # ---------- 2. index.json ----------
    backup(FULL_INDEX)
    ix = load_json(FULL_INDEX)

    # 西太部分: index.json 里是条带合并重提取前的旧序号 (44 路径已失效),
    # 用 species_name_index.json 已校对的 westpac_entries (与磁盘337文件一致) 重建
    wp_groups = OrderedDict()
    for e in sp["westpac_entries"]:
        key = (e["phylum"], e["scientific_name"])
        if key not in wp_groups:
            wp_groups[key] = {
                "images": [],
                "name_cn": e.get("name_cn", ""),
                "class": e.get("class", ""),
                "pages": [],
            }
        wp_groups[key]["images"].append(
            {"filename": e["filename"], "path": e["path"]})
        if e.get("page"):
            wp_groups[key]["pages"].append(e["page"])

    new_wp = []
    for (ph, sci), g in wp_groups.items():
        img_dir = os.path.dirname(g["images"][0]["path"])
        entry = {
            "morphotype_id": "",
            "scientific_name": sci,
            "name_cn": g["name_cn"],
            "phylum": ph,
            "phylum_cn": PHYLUM_CN.get(ph, ""),
            "class": g["class"],
            "source": "西太平洋图集",
            "image_count": len(g["images"]),
            "images": g["images"],
            "directory": img_dir,
        }
        if g["pages"]:
            entry["pages"] = sorted(set(g["pages"]))
        new_wp.append(entry)
    print(f"[OK] 西太重建: {len(new_wp)} 物种组, "
          f"{sum(len(e['images']) for e in new_wp)} 图片")

    new_np = []
    for e in np_entries:
        folder_dir = os.path.join(IMG_ROOT, e["folder"])
        desc = {"morphology": e.get("morphology", "")}
        if e.get("notes"):
            desc["notes"] = e["notes"]
        new_np.append({
            "morphotype_id": e["morphotype_id"],
            "scientific_name": e["name"],
            "phylum": e["phylum"],
            "phylum_cn": PHYLUM_CN.get(e["phylum"], ""),
            "class": e["class"] or "unclassified",
            "source": NP_SOURCE,
            "description": desc,
            "image_count": len(e["images"]),
            "images": [{"filename": i["filename"], "path": i["path"]}
                       for i in e["images"]],
            "directory": folder_dir,
        })
    ix["entries"] = new_np + new_wp

    # 重建三维索引
    by_phylum = OrderedDict()
    for e in ix["entries"]:
        ph = e.get("phylum") or "Unknown"
        if ph not in by_phylum:
            by_phylum[ph] = {"morphotype_count": 0, "image_count": 0,
                             "morphotypes": [],
                             "phylum_cn": e.get("phylum_cn", "")}
        by_phylum[ph]["image_count"] += e.get("image_count", 0)
        if e.get("morphotype_id"):
            by_phylum[ph]["morphotype_count"] += 1
            by_phylum[ph]["morphotypes"].append(e["morphotype_id"])

    by_morphotype = OrderedDict()
    id_seen = Counter()  # 复用ID第2次出现起加 #N 后缀保证键唯一
    for e in ix["entries"]:
        if not e.get("morphotype_id"):
            continue
        mid = e["morphotype_id"]
        id_seen[mid] += 1
        key = mid if id_seen[mid] == 1 else f"{mid}#{id_seen[mid]}"
        by_morphotype[key] = {
            "phylum": e.get("phylum", ""),
            "phylum_cn": e.get("phylum_cn", ""),
            "class": e.get("class", ""),
            "source": e.get("source", ""),
            "description": e.get("description", {}),
            "image_count": e.get("image_count", 0),
            "directory": e.get("directory", ""),
            "images": e.get("images", []),
        }

    by_class = OrderedDict()
    for e in ix["entries"]:
        if not e.get("morphotype_id"):
            continue
        cls = e.get("class") or "unclassified"
        if cls not in by_class:
            by_class[cls] = {"phylum": e.get("phylum", ""), "morphotypes": []}
        by_class[cls]["morphotypes"].append(e["morphotype_id"])

    ix["by_phylum"] = by_phylum
    ix["by_morphotype"] = by_morphotype
    ix["by_class"] = by_class
    total_img = sum(e.get("image_count", 0) for e in ix["entries"])
    ix["statistics"] = {
        "total_entries": len(ix["entries"]),
        "total_images": total_img,
        "total_morphotypes": len(by_morphotype),
        "phyla": list(by_phylum.keys()),
        "sources": sorted({e.get("source", "") for e in ix["entries"]}),
    }
    ix["updated"] = "2026-08-28"
    with open(FULL_INDEX, "w", encoding="utf-8") as f:
        json.dump(ix, f, ensure_ascii=False, indent=2)
    print(f"[OK] index.json: entries {len(ix['entries'])} "
          f"(北太平洋 {len(new_np)} + 西太物种组 {len(new_wp)}), "
          f"morphotypes {len(by_morphotype)}, images {total_img}")


if __name__ == "__main__":
    main()
