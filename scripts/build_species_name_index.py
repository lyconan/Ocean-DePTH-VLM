"""构建完整的「图片-描述-科学名」对应关系索引。

从两本图集 PDF 中提取科学学名，与 atlas_lookup/index.json 中的图片路径和形态描述
关联，输出结构化索引文件。

数据源:
  1. APSMA PDF (北太平洋深渊图集):
     - 每页结构: 形态型代码\n科学学名\nMorphology: 描述文本
     - 186 个形态型, 875 张图片
  2. 西太平洋图集 PDF:
     - 目录页: 序号. 中文名 | 拉丁名
     - 286 种底栖生物, 413 张图片 (按页码命名)
  3. atlas_lookup/index.json:
     - 现有图片路径 + 形态描述 (但缺少科学学名)

输出:
  - data/processed/atlas_lookup/species_name_index.json  (完整索引)
  - data/processed/apsma_parsed/apsma_scientific_names.json  (APSMA学名映射)
  - data/processed/apsma_parsed/westpac_species_names.json  (西太平洋物种名)
"""
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APSMA_PDF = PROJECT_ROOT / "data/raw/生物图集/Benthic Megafauna from the North Pacific Abyss 2023-.pdf"
WESTPAC_PDF = PROJECT_ROOT / "data/raw/生物图集/西太平洋深渊及临近区底栖生物合集.pdf"
INDEX_JSON = PROJECT_ROOT / "data/processed/atlas_lookup/index.json"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/apsma_parsed"
INDEX_OUTPUT = PROJECT_ROOT / "data/processed/atlas_lookup/species_name_index.json"


# ─── 1. 从 APSMA PDF 提取形态型→学名映射 ───────────────────────────

def extract_apsma_scientific_names(pdf_path: Path) -> Dict[str, Dict]:
    """从 APSMA PDF 逐页提取形态型代码和科学学名。

    PDF 每页结构:
        DES_002                          ← 形态型代码 (正则: ^[A-Z]{3}_\\d{3}$)
        Axoniderma mexicana sp. inc.    ← 科学学名 (下一行)
        Morphology: Parasol-shaped...   ← 形态描述

    Returns:
        {morphotype_id: {"scientific_name": str, "description": str, "page": int}}
    """
    doc = fitz.open(str(pdf_path))
    morpho_map: Dict[str, Dict] = {}
    code_pattern = re.compile(r"^([A-Z]{3}_\d{3})$")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for i, line in enumerate(lines):
            m = code_pattern.match(line)
            if m and i + 1 < len(lines):
                morpho_id = m.group(1)
                sci_name = lines[i + 1].strip()

                # 跳过非学名行（如 "Morphology:" 或纯数字）
                if sci_name.startswith("Morphology") or re.match(r"^\d+$", sci_name):
                    continue

                # 提取 Morphology 描述
                description = ""
                for j in range(i + 2, min(i + 10, len(lines))):
                    if lines[j].startswith("Morphology:"):
                        description = lines[j].replace("Morphology:", "").strip()
                        break

                # 首次出现优先（保留最早页码）
                if morpho_id not in morpho_map:
                    morpho_map[morpho_id] = {
                        "scientific_name": sci_name,
                        "description": description,
                        "page": page_num + 1,
                    }

    doc.close()
    print(f"APSMA: 提取到 {len(morpho_map)} 个形态型-学名映射")
    return morpho_map


def _infer_taxa_from_sci_name(sci_name: str) -> Tuple[str, str]:
    """从 scientific_name 推断 genus 和 family。

    模式:
      "Axoniderma mexicana sp. inc." → genus="Axoniderma", family=""
      "Acrocirridae gen. indet."     → genus="", family="Acrocirridae"
      "Peniagone sp. indet."         → genus="Peniagone", family=""
      "HOL_001" or ""                → genus="", family=""
    """
    if not sci_name:
        return "", ""
    parts = sci_name.strip().split()
    if not parts:
        return "", ""
    first = parts[0]
    # "Xxxdae gen. indet." → family
    if first.endswith("dae") and "gen." in sci_name:
        return "", first
    # "Xxxinae gen. indet." → subfamily (treated as family-level)
    if first.endswith("inae") and "gen." in sci_name:
        return "", first
    # "Xxxidae fam. indet." → family
    if first.endswith("idae") and "fam." in sci_name:
        return "", first
    # "GenusName speciesname ..." → genus (first word is capitalized = genus)
    if first[0].isupper() and not first.endswith("dae") and not first.endswith("idae"):
        return first, ""
    return "", ""


def _extract_apsma_taxonomy(pdf_path: Path) -> Dict[str, Dict]:
    """从 APSMA PDF 逐页提取每个形态型的完整分类学信息。

    APSMA PDF 每页通常在形态型代码和学名之后包含:
        Class XXX
        Order XXX
        Family XXX
        (Genus 隐含在学名中)

    Returns:
        {morphotype_id: {"class": str, "order": str, "family": str, "genus": str}}
    """
    doc = fitz.open(str(pdf_path))
    taxonomy_map: Dict[str, Dict] = {}
    code_pattern = re.compile(r"^([A-Z]{3}_\d{3})$")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        morpho_id = None
        for i, line in enumerate(lines):
            m = code_pattern.match(line)
            if m:
                morpho_id = m.group(1)
                continue

            if morpho_id and morpho_id not in taxonomy_map:
                # Parse taxonomy lines after the morphotype ID
                taxonomy: Dict[str, str] = {}
                for kw, key in [("Class", "class"), ("Order", "order"),
                                ("Family", "family"), ("Genus", "genus"),
                                ("Subclass", "subclass"), ("Suborder", "suborder")]:
                    kw_match = re.search(kw + r"\s+([A-Z][a-zA-Z]+)", line)
                    if kw_match:
                        taxonomy[key] = kw_match.group(1)

                if taxonomy:
                    taxonomy_map[morpho_id] = taxonomy

        # Also try to extract genus from scientific name
        # e.g. "Axoniderma mexicana sp. inc." → genus = "Axoniderma"
        # e.g. "Acrocirridae gen. indet." → family = "Acrocirridae"

    doc.close()

    # Post-process: extract genus/family from scientific_name for entries missing them
    # This will be done in build_complete_index using the scientific_name
    print(f"APSMA: 从PDF提取到 {len(taxonomy_map)} 个形态型的分类学信息")
    return taxonomy_map


# ─── 2. 从西太平洋图集 PDF 提取物种名 ───────────────────────────

# 西太平洋图集 OCR 噪声替换映射（中文名修正）
_CN_OCR_FIXES = {
    "珊珊": "珊瑚",
    "甲胃": "甲胄",
    "来定": "未定",
    "眉来定种": "属未定种",
    "自须": "白须",
    "臼须": "白须",
    "晰鱼": "蜥鱼",
    "锺": "鳗",
    "鲤鱼曼": "鳃鳗",
    "鲤鱼量": "鳃鳗",
    "梓": "种",
    "要总科": "总科",
    "画集": "图集",
    "圈集": "图集",
    "参回": "参目",
    "海要": "海葵",
    "叶班参": "叶斑参",
    "叶疵参": "叶斑参",
    "叶统参": "叶斑参",
    "渊螃参": "渊蝠参",
    "东参科": "东参科",
    "辛那参回": "辛那参目",
    "辛那参目;": "辛那参目",
    "烟灰蜡": "烟灰蜡",
    "钱足虾": "钳足虾",
    "丑柳珊珊": "丑柳珊瑚",
    "虹柳珊珊": "虹柳珊瑚",
    "金柳珊珊": "金柳珊瑚",
    "角柳珊珊": "角柳珊瑚",
    "拟柳珊蝴": "拟柳珊瑚",
    # 鳗鲡目相关
    "鳗细目": "鳗鲡目",
    "鳗鲡目": "鳗鲡目",
    "鳗鳞目": "鳗鲡目",
    # 鼬鳚科相关
    "舶细科": "鼬鳚科",
    "蹦蹦科": "鼬鳚科",
    "翩蹦科": "鼬鳚科",
    "鼬鰚科": "鼬鳚科",
    "鼬鲫科": "鼬鳚科",
    "拟柳珊蝴": "拟柳珊瑚",
    "短花海百合": "短花海百合",
}

# 西太平洋图集 OCR 噪声替换映射（拉丁名修正）
_LATIN_OCR_FIXES = {
    # ─── 分类学关键词 OCR 变体 ───
    # Order 变体
    "臼der": "Order",
    "位der": "Order",
    "Ord町": "Order",
    "order": "Order",
    "Qrder": "Order",
    "0rder": "Order",
    "0阳der": "Order",
    # Family 变体
    "Fami1y": "Family",
    "Fami ly": "Family",
    "F缸咀ly": "Family",
    "F缸回ly": "Family",
    "Fan咀ly": "Family",
    "F晒ily": "Family",
    "Fam均": "Family",
    "Fum且y": "Family",
    "Fa皿ly": "Family",
    # Class 变体
    "C1ass": "Class",
    "Cl副s": "Class",
    "Clωs": "Class",
    "Cl叩s": "Class",
    "Cla1s": "Class",
    # Subclass 变体
    "Subc1ass": "Subclass",
    "Subc1a ss": "Subclass",
    "Subcla剑": "Subclass",
    # Subfamily 变体
    "Subfi臼回ly": "Subfamily",
    "Subfami1y": "Subfamily",
    # Suborder 变体
    "Subo川er": "Suborder",
    "Sub0rder": "Suborder",
    "Subord町": "Suborder",
    "suborder": "Suborder",
    # Infraorder 变体
    "In企aorder": "Infraorder",
    "Infl'aorder": "Infraorder",
    "Infra0rder": "Infraorder",
    # Superfamily 变体
    "Sup,erfamily": "Superfamily",
    "Superfami1y": "Superfamily",
    # Genus 变体
    "Genu5": "Genus",
    # ─── 物种名 OCR 变体 ───
    # Anguilliformes OCR 变体
    "Anguilli岛rmes": "Anguilliformes",
    "Angui11iformes": "Anguilliformes",
    "An伊illiformes": "Anguilliformes",
    "Anguillifo即es": "Anguilliformes",
    "Anilliformes": "Anguilliformes",
    # Ophidiiformes OCR 变体
    "Ophidiif01mes": "Ophidiiformes",
    "Ophidiif0rmes": "Ophidiiformes",
    # Synaphobranchidae OCR 变体
    "Synaphob旧nchidae": "Synaphobranchidae",
    # Heterocarpus OCR 变体
    "Heterocαω": "Heterocarpus",
    # Aldrovandia OCR 变体
    "t{'nis": "tornis",
    # Phyllodocida OCR 变体
    "Phyl1odocida": "Phyllodocida",
    "Ph归lodocida": "Phyllodocida",
    # Poecilosclerida OCR 变体
    "Poeci1osclerida": "Poecilosclerida",
    # Decapoda OCR 变体
    "D日capoda": "Decapoda",
    "Sempe1'ella": "Semperella",
    "Sempe1\"ella": "Semperella",
    "Sempe1ella": "Semperella",
    "Ijimallus,hus": "Ijimaiellus",
    "Ijimall吨，hus": "Ijimaiellus",
    "Ijimallus,hus": "Ijimaiellus",
    "Cau~吨Jhacus": "Caulophacus",
    "Caul,哩,hacus": "Caulophacus",
    "Caulus,hacus": "Caulophacus",
    "C倒lop.归cus": "Caulophacus",
    "Dictyaul.ω": "Dictyaulus",
    "Dictyaul叫5": "Dictyaulus",
    "均，alonema": "Hyalonema",
    # 注意: 不添加 "alonema": "Hyalonema"，因为它会匹配已修复的 "Hyalonema" 内部
    "Cerian也ria": "Ceriantharia",
    "Dictyaulus k，臼ueae": "Dictyaulus kitaueae",
    "Ac伽h巾": "Actiniaria",
    "Actinωcyph": "Actinoscyphia",
    "Olln硝us": "Olindias",
    "Fungillcyathus": "Fungiacyathus",
    "Fungicyathus": "Fungiacyathus",
    "Wa1terla": "Walteria",
    "Walтerla": "Walteria",
    # Subclass name OCR 变体
    "Hex臼阳ophora": "Hexasterophora",
    "Hex扣sterophora": "Hexasterophora",
    "Hex邵阳ophora": "Hexasterophora",
    "Hexasterophora": "Hexasterophora",  # already correct, for safety
    # Subfamily name OCR 变体
    "Lanuginel1inae": "Lanuginellinae",
    "Lanugine11inae": "Lanuginellinae",
    "Corbitel1inae": "Corbitellinae",
    "Corbi伽1inae": "Corbitellinae",
    # Ophiurida OCR 变体
    "句hi回da": "Ophiurida",
    "句hi田ida": "Ophiurida",
    "句hi回dasp": "Ophiurida sp",
    # Synallactida/Synallactidae OCR 变体
    "8归allactida": "Synallactida",
    "S归allactida": "Synallactida",
    "S归d剧目也": "Synallactidae",
    "S归allacti也": "Synallactidae",
    "严allactidae": "Synallactidae",
    "S归allac世也": "Synallactidae",
    # Elpidiidae OCR 变体
    "E1pidiidae": "Elpidiidae",
    "E1pidii也": "Elpidiidae",
    "E毕由dae": "Elpidiidae",
    "E1pi，也由": "Elpidiidae",
    # Iridogorgia OCR 变体
    "llridogo.喀臼": "Iridogorgia",
    "/1币dogo喀ia": "Iridogorgia",
    "lridogorgia": "Iridogorgia",
    # Peniagone OCR 变体
    "P，四iagone": "Peniagone",
    # Macrouridae OCR 变体
    "蜘crouri也e": "Macrouridae",
    "蜘crouri也esp": "Macrouridae sp",
    # Liparidae OCR 变体
    "L坦aridae": "Liparidae",
    # Echinosoma/Phrissoma OCR 变体
    "牛erosoma": "Echinosoma",
    "邱'grosoma": "Echinosoma",
    # Elpadiodemas OCR 变体
    "胁lpadiodemas": "Elpadiodemas",
    # Chrysogorgiidae OCR 变体
    "阶也moi也e": "Chrysogorgiidae",
    "阶也moi也esp": "Chrysogorgiidae sp",
    # Echinothuriidae OCR 变体
    "臼1沪nni也e": "Echinothuriidae",
    "臼lymni由e": "Echinothuriidae",
    "ωymni": "Echinothuriidae",
    # Keratoisididae OCR 变体
    "K.eratoisididae": "Keratoisididae",
    # Hyocidae → Hyocrinidae
    "Hyocidae": "Hyocrinidae",
    # 作者名修正
    "Sc恤idt": "Schmidt",
    "Cárdenas": "Cardenas",
    "Boury-Esnau扰": "Boury-Esnault",
    "Boury-Esnau1t": "Boury-Esnault",
    # seq 219: "Fa皿ly" OCR 变体 + Elpidiidae OCR 变体
    "Fa皿ly": "Family",
    "E1pidüdae": "Elpidiidae",
    "E1pidii也": "Elpidiidae",
    "Blpidü也": "Elpidiidae",
    "Blpidüdae": "Elpidiidae",
    "Elpadiodemas": "Elpadiodemas",
    # Corbitellinae OCR 变体 (seq 17)
    "Co油iteUinae": "Corbitellinae",
    "Corbitettinae": "Corbitellinae",
    "Corbi也Uinae": "Corbitellinae",
    # Psychropotes/Psychropotidae OCR 变体 (seq 206)
    "Psychropotidae": "Psychropotidae",
    "Psychropo也dae": "Psychropotidae",
    "Psychropo也id也": "Psychropotidae",
}

# 手动修正: 3 个 OCR 噪声过于严重无法自动提取的物种
_MANUAL_SPECIES_FIXES = {
    17: {"name_cn": "舟体海绵亚科未定种", "name_latin": "Corbitellinae sp."},
    206: {"name_cn": "蝶参属未定种", "name_latin": "Psychropotes sp."},
    219: {"name_cn": "东参科未定种3", "name_latin": "Elpidiidae sp. 3"},
}


def _fix_ocr(text: str, fixes: Dict[str, str]) -> str:
    """应用 OCR 修正映射。"""
    for old, new in fixes.items():
        text = text.replace(old, new)
    return text


def extract_westpac_species(pdf_path: Path) -> List[Dict]:
    """从西太平洋图集 PDF 提取物种列表（改进版）。

    策略:
      1. 从正文页标题行提取序号和中文名
      2. 从同一页"分类学地位"部分提取拉丁名
         - Genus 行 → 属级学名（最干净）
         - Family 行 → 科级学名
         - Order 行 → 目级学名
         - Class 行 → 纲级学名
      3. 根据中文名判断鉴定级别，选择对应分类行
      4. 如分类行不可用，回退到标题行拉丁名
    """
    doc = fitz.open(str(pdf_path))
    entries: Dict[int, Dict] = {}

    seq_pattern = re.compile(r"^(\d{1,3})\.\s*(.+)")

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for i, line in enumerate(lines):
            m = seq_pattern.match(line)
            if not m:
                continue
            seq = int(m.group(1))
            # 过滤假阳性: seq=1 和 seq=2 是章节标题，不是物种
            if not (3 <= seq <= 286):
                continue

            rest = m.group(2)

            # 去除目录页的点引导符和尾部页码
            if "…" in rest:
                rest = re.sub(r"….*$", "", rest).strip()
            rest = re.sub(r"\.{2,}.*$", "", rest).strip()
            rest = re.sub(r"~.*$", "", rest).strip()

            # 分离中文名和拉丁名（从标题行）
            name_cn, latin_header = _split_cn_latin(rest, lines, i)

            # 修正中文名 OCR 噪声
            name_cn = _fix_ocr(name_cn, _CN_OCR_FIXES)
            # 清理中文名尾部残留的非中文字符
            name_cn = re.sub(r"[^\u4e00-\u9fff\d]+$", "", name_cn).strip()
            name_cn = _fix_ocr(name_cn, _CN_OCR_FIXES)

            # 从分类学部分提取干净的拉丁名
            taxonomy = _extract_taxonomy(lines, i)
            latin_name = _resolve_latin_name(name_cn, latin_header, taxonomy)

            # 仅清理格式（不再重复应用 OCR 修正，避免破坏已清洁的学名）
            latin_name = _sanitize_latin(latin_name)

            if seq not in entries or (page_num >= 18 and entries[seq]["page"] < 18):
                # 正文页优先于目录页
                entries[seq] = {
                    "seq": seq,
                    "name_cn": name_cn,
                    "name_latin": latin_name,
                    "page": page_num + 1,
                    "taxonomy": taxonomy,
                }

    doc.close()

    # 使用分类学规则补全缺失的 class 信息
    for seq, entry in entries.items():
        tax = entry.get("taxonomy", {})
        if not tax.get("class"):
            inferred = _infer_class_from_taxonomy(entry.get("name_cn", ""), tax)
            if inferred:
                tax["class"] = inferred
                entry["taxonomy"] = tax

    # 应用手动修正（3 个 OCR 噪声过于严重的物种）
    for seq, fix in _MANUAL_SPECIES_FIXES.items():
        if seq in entries:
            entries[seq]["name_cn"] = fix["name_cn"]
            entries[seq]["name_latin"] = fix["name_latin"]
        else:
            # 如果 seq 未被提取到，手动添加
            entries[seq] = {
                "seq": seq,
                "name_cn": fix["name_cn"],
                "name_latin": fix["name_latin"],
                "page": 0,
            }

    species_list = sorted(entries.values(), key=lambda x: x["seq"])
    print(f"西太平洋: 提取到 {len(species_list)} 个物种")
    return species_list


def _infer_class_from_taxonomy(name_cn: str, taxonomy: Dict[str, str]) -> str:
    """根据已有的 order/subclass/family 信息推断 class。

    西太平洋图集中很多物种的分类学部分没有 Class 行，
    但有 Order/Subclass/Family 行，可以据此推断 Class。

    Returns:
        class 名（空字符串表示无法推断）
    """
    order = taxonomy.get("order", "")
    subclass = taxonomy.get("subclass", "")
    family = taxonomy.get("family", "")

    # Order → Class 映射（基于标准海洋无脊椎动物分类学）
    _ORDER_CLASS = {
        # Porifera - Hexactinellida
        "Amphidiscosida": "Hexactinellida",
        "Lyssacinosida": "Hexactinellida",
        "Sceptrulophora": "Hexactinellida",
        # Porifera - Demospongiae
        "Poecilosclerida": "Demospongiae",
        "Tetractinellida": "Demospongiae",
        "Hadromerida": "Demospongiae",
        "Axinellida": "Demospongiae",
        "Agelasida": "Demospongiae",
        "Clionaida": "Demospongiae",
        "Bubarida": "Demospongiae",
        "Merliida": "Demospongiae",
        "Suberitida": "Demospongiae",
        "Tethyida": "Demospongiae",
        "Chondrillida": "Demospongiae",
        "Chondrosiida": "Demospongiae",
        "Verongiida": "Demospongiae",
        "Dendroceratida": "Demospongiae",
        "Dictyoceratida": "Demospongiae",
        "Haplosclerida": "Demospongiae",
        # Cnidaria - Anthozoa
        "Alcyonacea": "Anthozoa",
        "Actiniaria": "Anthozoa",
        "Antipatharia": "Anthozoa",
        "Scleractinia": "Anthozoa",
        "Zoantharia": "Anthozoa",
        "Corallimorpharia": "Anthozoa",
        "Pennatulacea": "Anthozoa",
        # Cnidaria - Hydrozoa
        "Anthoathecata": "Hydrozoa",
        "Leptothecata": "Hydrozoa",
        "Siphonophorae": "Hydrozoa",
        "Trachylina": "Hydrozoa",
        # Annelida
        "Phyllodocida": "Polychaeta",
        "Eunicida": "Polychaeta",
        "Sabellida": "Polychaeta",
        "Terebellida": "Polychaeta",
        # Mollusca
        "Neogastropoda": "Gastropoda",
        "Nudibranchia": "Gastropoda",
        "Octopoda": "Cephalopoda",
        "Pectinida": "Bivalvia",
        "Mytilida": "Bivalvia",
        "Venerida": "Bivalvia",
        # Arthropoda
        "Decapoda": "Malacostraca",
        "Isopoda": "Malacostraca",
        "Amphipoda": "Malacostraca",
        "Mysida": "Malacostraca",
        "Cumacea": "Malacostraca",
        "Tanaidacea": "Malacostraca",
        # Echinodermata
        "Elasipodida": "Holothuroidea",
        "Holothuriida": "Holothuroidea",
        "Synallactida": "Holothuroidea",
        "Persiculida": "Holothuroidea",
        "Molpadida": "Holothuroidea",
        "Dendrochirotida": "Holothuroidea",
        "Comatulida": "Crinoidea",
        "Hyocrinida": "Crinoidea",
        "Isocrinida": "Crinoidea",
        "Ophiurida": "Ophiuroidea",
        "Ophiacanthida": "Ophiuroidea",
        "Euryalida": "Ophiuroidea",
        "Paxillosida": "Asteroidea",
        "Velatida": "Asteroidea",
        "Spinulosida": "Asteroidea",
        "Valvatida": "Asteroidea",
        "Forcipulatida": "Asteroidea",
        "Brisingida": "Asteroidea",
        "Echinothurioida": "Echinoidea",
        "Holasteroida": "Echinoidea",
        "Spatangoida": "Echinoidea",
        "Arbacioida": "Echinoidea",
        "Diadematoida": "Echinoidea",
        # Chordata
        "Anguilliformes": "Actinopterygii",
        "Ophidiiformes": "Actinopterygii",
        "Gadiformes": "Actinopterygii",
        "Aulopiformes": "Actinopterygii",
        "Notacanthiformes": "Actinopterygii",
        "Scorpaeniformes": "Actinopterygii",
        "Perciformes": "Actinopterygii",
        "Myxiniformes": "Myxini",
    }

    # Subclass → Class 映射
    _SUBCLASS_CLASS = {
        "Hexasterophora": "Hexactinellida",
        "Amphidiscophora": "Hexactinellida",
        "Heteroscleromorpha": "Demospongiae",
        "Ceractinomorpha": "Demospongiae",
        "Tetractinomorpha": "Demospongiae",
        "Octocorallia": "Anthozoa",
        "Hexacorallia": "Anthozoa",
        "Hydroidolina": "Hydrozoa",
        "Trachylinae": "Hydrozoa",
        "Coleoidea": "Cephalopoda",
    }

    # 1. Subclass 优先（最精确）
    if subclass:
        for key, cls in _SUBCLASS_CLASS.items():
            if subclass.startswith(key[:5]):  # 模糊匹配前5字符（处理OCR截断）
                return cls

    # 2. Order 次之
    if order:
        for key, cls in _ORDER_CLASS.items():
            if order.startswith(key[:5]) or key.startswith(order[:5]):
                return cls

    # 3. 中文名关键词匹配
    if "六放海绵" in name_cn:
        return "Hexactinellida"
    if "寻常海绵" in name_cn:
        return "Demospongiae"
    if "珊瑚" in name_cn:
        return "Anthozoa"
    if "海葵" in name_cn:
        return "Anthozoa"
    if "海鳃" in name_cn:
        return "Anthozoa"
    if "水母" in name_cn:
        return "Hydrozoa"
    if "水螅" in name_cn:
        return "Hydrozoa"
    if "参" in name_cn and "海" in name_cn:
        return "Holothuroidea"
    if "海百合" in name_cn or "海羊齿" in name_cn:
        return "Crinoidea"
    if "蛇尾" in name_cn:
        return "Ophiuroidea"
    if "海星" in name_cn:
        return "Asteroidea"
    if "海胆" in name_cn:
        return "Echinoidea"
    if "虾" in name_cn or "蟹" in name_cn:
        return "Malacostraca"
    if "鳗" in name_cn:
        return "Actinopterygii"
    if "鱼" in name_cn:
        return "Actinopterygii"

    return ""


def _sanitize_taxonomy_value(name: str) -> str:
    """清理分类学值中的 Author 年份和 OCR 残留。

    分类名（目/科/属/纲等）都是单个拉丁词，首字母大写、其余小写。
    OCR噪声模式：中文混入拉丁名（"Hexac邵阳orallia"）、Author年份粘连。

    例如:
      "Hexac邵阳orallia" → "Hexacorallia"
      "Anguilli岛rmesB吨，1943" → "Anguilliformes"
      "Hexasterophora Schulze" → "Hexasterophora"
    """
    if not name:
        return ""
    # 1. 先去除所有非ASCII字符（中文OCR噪声），将断裂的拉丁名重新连接
    name = re.sub(r"[^\x00-\x7f]+", "", name).strip()
    # 2. 去除 Author 年份和粘连的Author名
    # "Name, 1886" → "Name"
    name = re.sub(r",\s*\d{4}.*$", "", name).strip()
    # "NameSchulze，1886" → "Name"（Author名首字母粘连）
    name = re.sub(r"[a-z][A-Z][a-zA-Z]*[，,]\s*\d{4}.*$", "", name).strip()
    # "Name Schulze" → "Name"（空格分隔的Author名）
    # 但要区分：如果第二个词也是大写开头，可能是Author名
    # 只保留第一个词
    name = name.split()[0] if name.split() else name
    # 3. 清理尾部标点
    name = re.sub(r"[.,;:]+$", "", name).strip()
    # 4. 清理多余空格
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _extract_taxonomy(lines: List[str], header_idx: int) -> Dict[str, str]:
    """从标题行之后的行中提取分类学信息。

    PDF 分类学部分结构:
        分类学地位
        [中文分类名]
        Subclass/Class/Order/Family/Genus LatinName Author, Year
        ...

    Returns:
        {"genus": str, "family": str, "order": str, "class": str, ...}
    """
    taxonomy: Dict[str, str] = {}
    # 关键词 → 键名映射（按优先级：Genus > Family > Order > Class）
    keywords = [
        ("Genus", "genus"),
        ("Family", "family"),
        ("Order", "order"),
        ("Class", "class"),
        ("Subclass", "subclass"),
        ("Suborder", "suborder"),
        ("Subfamily", "subfamily"),
        ("Infraorder", "infraorder"),
        ("Superfamily", "superfamily"),
    ]

    for j in range(header_idx + 1, min(header_idx + 35, len(lines))):
        line = lines[j]
        # 遇到下一个物种标题则停止
        if re.match(r"^\d{1,3}\.\s", line):
            break
        if line.startswith("采集地") or line.startswith("深度") or line.startswith("来集地"):
            break

        for kw, key in keywords:
            # 先修复 OCR 噪声（如 "Fa皿ly" → "Family", "臼der" → "Order"）
            fixed_line = _fix_ocr(line, _LATIN_OCR_FIXES)
            if kw not in fixed_line:
                continue
            # 提取关键词后的拉丁名
            # 策略: 匹配关键词后到逗号(作者年份)或行尾之间的拉丁名部分
            # 允许数字（OCR噪声如 "Lanugine11inae"）
            # 也要处理关键词与拉丁名粘连的情况（如 "OrderLyssacinosida"）
            kw_pos = fixed_line.index(kw)
            segment = fixed_line[kw_pos + len(kw):].strip()
            # 截断到逗号（作者年份开始）
            comma_pos = segment.find(",")
            if comma_pos > 0:
                segment = segment[:comma_pos].strip()
            # 截断到数字年份 (如 "1886")
            year_match = re.search(r"\s\d{4}", segment)
            if year_match:
                segment = segment[:year_match.start()].strip()
            # 提取拉丁名
            # 策略: 匹配关键词后第一个大写开头的拉丁名
            # 处理名字和Author粘连（如 "DictyaulusSchulze" → "Dictyaulus"）
            # 方法: 找到第一个大写字母开头后跟小写的位置，分割名字和Author
            # 但名字本身可能有内部大写（如 "Nematocarcinoidea"），不能过早截断
            # 所以: 只在 "小写字母+大写字母+小写字母" 模式处分割
            split_pos = re.search(r"[a-z][A-Z][a-z]", segment)
            if split_pos:
                name = segment[:split_pos.start() + 1]
            else:
                name = segment
            # 只保留第一个词（空格分隔）
            name = name.split()[0] if name.split() else name
            # 确保以大写字母开头
            if name and name[0].isupper() and key not in taxonomy:
                taxonomy[key] = name
            # 清理 OCR 数字
            if key in taxonomy:
                taxonomy[key] = _fix_ocr(taxonomy[key], _LATIN_OCR_FIXES)
                # 清理 taxonomy 值中的 Author 年份和 OCR 残留
                taxonomy[key] = _sanitize_taxonomy_value(taxonomy[key])
            break

    return taxonomy


def _resolve_latin_name(
    name_cn: str,
    latin_header: str,
    taxonomy: Dict[str, str],
) -> str:
    """根据中文名判断鉴定级别，从分类学信息获取拉丁名。

    中文名模式 → 鉴定级别:
      "X属未定种" → 属级 → Genus name + " sp."
      "X科未定种" → 科级 → Family name + " sp."
      "X目未定种" → 目级 → Order name + " sp."
      "X纲未定种" → 纲级 → Class name + " sp."
      "X亚科未定种" → 亚科级
      "X亚目未定种" → 亚目级
      "X亚纲未定种" → 亚纲级
      "X总科未定种" → 总科级
      其他 → 种级 → 使用标题行拉丁名
    """
    # 清理标题行拉丁名（仅对标题行应用 OCR 修正）
    if latin_header:
        latin_header = _fix_ocr(latin_header, _LATIN_OCR_FIXES)
        latin_header = _sanitize_latin(latin_header)

    # 从中文名提取编号（如 "未定种3" → "3"）
    num_match = re.search(r"未定种(\d+)$", name_cn)
    sp_num = num_match.group(1) if num_match else ""
    sp_suffix = f" sp. {sp_num}" if sp_num else " sp."

    # 属级未定种
    if "属未定种" in name_cn or "眉来定种" in name_cn:
        if "genus" in taxonomy:
            return taxonomy["genus"] + sp_suffix
        return latin_header

    # 科级未定种
    if "科未定种" in name_cn:
        if "family" in taxonomy:
            return taxonomy["family"] + sp_suffix
        return latin_header

    # 目级未定种
    if "目未定种" in name_cn:
        if "order" in taxonomy:
            return taxonomy["order"] + sp_suffix
        return latin_header

    # 纲级未定种
    if "纲未定种" in name_cn:
        if "class" in taxonomy:
            return taxonomy["class"] + sp_suffix
        return latin_header

    # 亚科级未定种
    if "亚科未定种" in name_cn:
        if "subfamily" in taxonomy:
            return taxonomy["subfamily"] + sp_suffix
        return latin_header

    # 亚目级未定种
    if "亚目未定种" in name_cn:
        if "suborder" in taxonomy:
            return taxonomy["suborder"] + sp_suffix
        return latin_header

    # 亚纲级未定种
    if "亚纲未定种" in name_cn:
        if "subclass" in taxonomy:
            return taxonomy["subclass"] + sp_suffix
        return latin_header

    # 总科级未定种
    if "总科未定种" in name_cn:
        if "superfamily" in taxonomy:
            return taxonomy["superfamily"] + sp_suffix
        # 总科在分类中通常没有单独的行，回退到 family 或 order
        if "family" in taxonomy:
            return taxonomy["family"] + sp_suffix
        return latin_header

    # 下目级未定种
    if "下目未定种" in name_cn:
        if "infraorder" in taxonomy:
            return taxonomy["infraorder"] + sp_suffix
        return latin_header

    # 种级（有完整种名）
    if latin_header and len(latin_header) >= 4:
        return latin_header
    # 回退: 用 genus 名 + sp.
    if "genus" in taxonomy:
        return taxonomy["genus"] + " sp."
    if "family" in taxonomy:
        return taxonomy["family"] + " sp."
    return latin_header


def _split_cn_latin(rest: str, lines: List[str], j: int) -> Tuple[str, str]:
    """将 "中文名 拉丁名" 或 "中文名拉丁名" 分离为 (name_cn, name_latin)。

    PDF OCR 噪声导致中文名和拉丁名可能:
      1. 有空格分隔: "多鳞虫科未定种1 Polynoidae sp. 1"
      2. 无空格连接: "囊霉海绵属未定种Saccocalyx sp."
      3. 拉丁名在下一行
      4. 拉丁名混入中文字符: "均，alonema sp."
      5. 拉丁名因OCR噪声不以大写字母开头: "长茎海绵属未定种3C倒lop.归cus sp. 3"
    """
    # 策略0: 使用中文分类标记作为分割边界
    # 模式: "X属未定种N" 或 "X科未定种N" 或 "X目未定种N" 等
    cn_boundary = re.match(
        r'^(.+?(?:属|科|目|纲|亚科|亚目|亚纲|总科)未定种\d*)(.*)', rest
    )
    if cn_boundary:
        cn_part = cn_boundary.group(1).strip()
        latin_part = cn_boundary.group(2).strip()
        if latin_part:
            return cn_part, latin_part

    # 策略1: 尝试在中文和拉丁交界处切分
    # 拉丁名以大写字母开头
    latin_start = re.search(r"[A-Z][a-z]", rest)
    if latin_start:
        pos = latin_start.start()
        cn_part = rest[:pos].strip()
        latin_part = rest[pos:].strip()
        # 去除中文名尾部的非中文字符
        cn_part = re.sub(r"[^\u4e00-\u9fff\d]+$", "", cn_part).strip()
        return cn_part, latin_part

    # 策略2: 检查是否有小写字母混在中文名中（如 "均，alonema"）
    embedded = re.search(r"[，,]([a-z][a-z]+)", rest)
    if embedded:
        pos = embedded.start()
        cn_part = rest[:pos].strip()
        cn_part = re.sub(r"[^\u4e00-\u9fff\d]+$", "", cn_part).strip()
        latin_part = rest[embedded.start() + 1:].strip()
        return cn_part, latin_part

    # 策略3: 拉丁名在下一行
    if j + 1 < len(lines):
        next_line = lines[j + 1]
        if re.match(r"^[A-Z][a-z]", next_line):
            return rest.strip(), next_line.strip()

    # 策略4: 纯中文名，无拉丁名
    return rest.strip(), ""


def _sanitize_latin(name: str) -> str:
    """清理拉丁名中的 OCR 噪声和格式问题。"""
    if not name:
        return ""
    # 去除所有类型的尾部省略号、点、波浪号及跟随的页码
    # 但保留 "sp. N" 格式（如 "sp. 2"）
    if not re.search(r"sp\.\s*\d+$", name):
        name = re.sub(r"[.…~·]+\s*\d*\s*$", "", name).strip()
        name = re.sub(r"\.{2,}.*$", "", name).strip()
        name = re.sub(r"….*$", "", name).strip()
        name = re.sub(r"~.*$", "", name).strip()
    # 去除尾部孤立的数字（序号残留）但保留 sp. N 格式
    if not re.search(r"sp\.\s*\d+$", name):
        name = re.sub(r"\s+\d+$", "", name).strip()
    # 去除尾部的句点和逗号（但保留 "sp." 中的点）
    if not name.rstrip().endswith("sp."):
        name = re.sub(r"[.,]+$", "", name).strip()
    # 合并多余空格
    name = re.sub(r"\s+", " ", name).strip()
    # 移除混入的中文字符
    name = re.sub(r"[\u4e00-\u9fff]+", "", name).strip()
    # 再次合并空格和清理尾部（保留 "sp." 中的点）
    name = re.sub(r"\s+", " ", name).strip()
    if not name.rstrip().endswith("sp."):
        name = re.sub(r"[.,]+$", "", name).strip()
    return name


def clean_latin_name(name: str) -> str:
    """清洗拉丁名中的 OCR 噪声（用于后续处理步骤）。"""
    if not name:
        return ""
    name = _fix_ocr(name, _LATIN_OCR_FIXES)
    name = _sanitize_latin(name)
    return name


# ─── 3. 将西太平洋图集图片按页码映射到物种 ───────────────────────────

def map_westpac_images_to_species(
    species_list: List[Dict],
    index_data: Dict,
) -> List[Dict]:
    """将西太平洋图集图片按页码映射到对应物种。

    图片命名: Echinodermata_p100_453.jpeg → page 100
    需要建立 page → species 映射。
    """
    # 建立 page → species 映射
    page_to_species: Dict[int, Dict] = {}
    for sp in species_list:
        page_to_species[sp["page"]] = sp

    # 从 index.json 获取西太平洋图集的图片列表
    westpac_entries = []
    entries = index_data.get("entries", [])
    for entry in entries:
        if entry.get("source") == "西太平洋图集":
            westpac_entries.append(entry)

    # 也检查 by_morphotype 中 source 为西太平洋的
    by_morphotype = index_data.get("by_morphotype", {})
    for mt_id, info in by_morphotype.items():
        if info.get("source") == "西太平洋图集":
            westpac_entries.append(info)

    # 按页码分组图片并匹配物种
    result = []
    for entry in westpac_entries:
        images = entry.get("images", [])
        phylum = entry.get("phylum", "")

        for img in images:
            filename = img.get("filename", "")
            # 从文件名提取页码: Echinodermata_p100_453.jpeg → 100
            page_match = re.search(r"_p(\d+)_", filename)
            if page_match:
                page_num = int(page_match.group(1))
                # 查找该页对应的物种（允许±1页误差）
                species = page_to_species.get(page_num)
                if not species:
                    # 尝试相邻页
                    for offset in [-1, 1, -2, 2]:
                        species = page_to_species.get(page_num + offset)
                        if species:
                            break

                if species:
                    result.append({
                        "filename": filename,
                        "path": img.get("path", ""),
                        "phylum": phylum,
                        "species_seq": species["seq"],
                        "name_cn": species["name_cn"],
                        "name_latin": clean_latin_name(species["name_latin"]),
                        "page": page_num,
                        "taxonomy": species.get("taxonomy", {}),
                    })
                else:
                    result.append({
                        "filename": filename,
                        "path": img.get("path", ""),
                        "phylum": phylum,
                        "species_seq": None,
                        "name_cn": "",
                        "name_latin": "",
                        "page": page_num,
                        "taxonomy": {},
                    })

    print(f"西太平洋: {len(result)}/{sum(len(e.get('images',[])) for e in westpac_entries)} 张图片映射到物种")
    return result


# ─── 4. 构建完整索引 ───────────────────────────

def build_complete_index(
    index_data: Dict,
    apsma_names: Dict[str, Dict],
    westpac_species: List[Dict],
    westpac_image_map: List[Dict],
) -> Dict:
    """构建完整的「图片-描述-科学名」对应关系索引。"""
    by_morphotype = index_data.get("by_morphotype", {})

    # ── 从 APSMA PDF 提取完整分类学信息 ──
    apsma_taxonomy = _extract_apsma_taxonomy(APSMA_PDF)

    # ── APSMA 部分 ──
    apsma_entries = []
    matched = 0
    unmatched = 0

    for mt_id, info in by_morphotype.items():
        if info.get("source") != "北太平洋图集(APSMA完整版)":
            continue

        sci_name_data = apsma_names.get(mt_id, {})
        scientific_name = sci_name_data.get("scientific_name", "")
        pdf_description = sci_name_data.get("description", "")

        if scientific_name:
            matched += 1
        else:
            unmatched += 1

        # 合并分类学信息: index.json 的 class + PDF 提取的 order/family/genus
        tax = apsma_taxonomy.get(mt_id, {})
        entry_class = info.get("class", "") or tax.get("class", "")

        # 从 scientific_name 推断缺失的 genus/family
        inferred_genus, inferred_family = _infer_taxa_from_sci_name(scientific_name)

        images = info.get("images", [])
        apsma_entries.append({
            "morphotype_id": mt_id,
            "scientific_name": scientific_name,
            "phylum": info.get("phylum", ""),
            "phylum_cn": info.get("phylum_cn", ""),
            "class": entry_class,
            "order": tax.get("order", ""),
            "family": tax.get("family", "") or inferred_family,
            "genus": tax.get("genus", "") or inferred_genus,
            "description": info.get("description", {}).get("morphology", "") if isinstance(info.get("description"), dict) else str(info.get("description", "")),
            "description_from_pdf": pdf_description,
            "image_count": len(images),
            "images": [{"filename": img["filename"], "path": img["path"]} for img in images],
            "source": "北太平洋图集(APSMA完整版)",
        })

    # ── 西太平洋部分 ──
    westpac_entries = []
    for item in westpac_image_map:
        tax = item.get("taxonomy", {})
        westpac_entries.append({
            "filename": item["filename"],
            "path": item["path"],
            "phylum": item["phylum"],
            "scientific_name": item["name_latin"],
            "name_cn": item["name_cn"],
            "species_seq": item["species_seq"],
            "page": item["page"],
            "class": tax.get("class", ""),
            "subclass": tax.get("subclass", ""),
            "order": tax.get("order", ""),
            "suborder": tax.get("suborder", ""),
            "infraorder": tax.get("infraorder", ""),
            "superfamily": tax.get("superfamily", ""),
            "family": tax.get("family", ""),
            "subfamily": tax.get("subfamily", ""),
            "genus": tax.get("genus", ""),
            "source": "西太平洋图集",
        })

    # ── 汇总统计 ──
    total_images_apsma = sum(e["image_count"] for e in apsma_entries)
    total_images_westpac = len(westpac_entries)
    matched_westpac = sum(1 for e in westpac_entries if e["scientific_name"])

    complete_index = {
        "version": "2.0",
        "description": "深渊底栖生物「图片-描述-科学名」完整对应关系索引",
        "sources": {
            "APSMA": {
                "pdf": "Benthic Megafauna from the North Pacific Abyss 2023-.pdf",
                "morphotypes": len(apsma_entries),
                "with_scientific_name": matched,
                "without_scientific_name": unmatched,
                "total_images": total_images_apsma,
            },
            "WestPacific": {
                "pdf": "西太平洋深渊及临近区底栖生物合集.pdf",
                "species_extracted": len(westpac_species),
                "total_images": total_images_westpac,
                "images_with_species": matched_westpac,
                "images_without_species": total_images_westpac - matched_westpac,
            },
        },
        "statistics": {
            "total_entries": len(apsma_entries) + len(westpac_entries),
            "total_morphotypes": len(apsma_entries),
            "total_images": total_images_apsma + total_images_westpac,
            "scientific_name_coverage": f"{matched}/{len(apsma_entries)} APSMA + {matched_westpac}/{total_images_westpac} WestPacific",
        },
        "apsma_entries": apsma_entries,
        "westpac_entries": westpac_entries,
        "westpac_species_list": westpac_species,
    }

    return complete_index


# ─── 5. 更新 apsma_knowledge_validated.jsonl ───────────────────────────

def update_apsma_jsonl(
    jsonl_path: Path,
    apsma_names: Dict[str, Dict],
    output_path: Path,
):
    """为 apsma_knowledge_validated.jsonl 每条记录补充 scientific_name 字段。"""
    updated = 0
    records = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            orig_id = rec.get("original_morphotype_id", "")
            name_data = apsma_names.get(orig_id, {})
            if name_data:
                rec["scientific_name"] = name_data.get("scientific_name", "")
                rec["taxonomy"]["scientific_name"] = name_data.get("scientific_name", "")
                updated += 1
            else:
                rec["scientific_name"] = ""
            records.append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"更新 apsma_knowledge_validated.jsonl: {updated}/{len(records)} 条记录补充了学名")


# ─── 主函数 ───────────────────────────

def main():
    print("=" * 60)
    print("构建「图片-描述-科学名」完整对应关系索引")
    print("=" * 60)

    # 1. 从 APSMA PDF 提取学名
    print("\n[1/5] 从 APSMA PDF 提取形态型-学名映射...")
    apsma_names = extract_apsma_scientific_names(APSMA_PDF)

    # 保存学名映射
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apsma_names_path = OUTPUT_DIR / "apsma_scientific_names.json"
    with open(apsma_names_path, "w", encoding="utf-8") as f:
        json.dump(apsma_names, f, ensure_ascii=False, indent=2)
    print(f"  → 保存至 {apsma_names_path}")

    # 2. 从西太平洋 PDF 提取物种
    print("\n[2/5] 从西太平洋图集提取物种列表...")
    westpac_species = extract_westpac_species(WESTPAC_PDF)

    # 清洗拉丁名
    for sp in westpac_species:
        sp["name_latin"] = clean_latin_name(sp["name_latin"])

    westpac_names_path = OUTPUT_DIR / "westpac_species_names.json"
    with open(westpac_names_path, "w", encoding="utf-8") as f:
        json.dump(westpac_species, f, ensure_ascii=False, indent=2)
    print(f"  → 保存至 {westpac_names_path}")

    # 3. 加载现有 index.json
    print("\n[3/5] 加载 atlas_lookup/index.json...")
    with open(INDEX_JSON, encoding="utf-8") as f:
        index_data = json.load(f)
    print(f"  → {len(index_data.get('by_morphotype', {}))} 个形态型")

    # 4. 映射西太平洋图片到物种
    print("\n[4/5] 映射西太平洋图片到物种...")
    westpac_image_map = map_westpac_images_to_species(westpac_species, index_data)

    # 5. 构建完整索引
    print("\n[5/5] 构建完整索引...")
    complete_index = build_complete_index(index_data, apsma_names, westpac_species, westpac_image_map)

    with open(INDEX_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(complete_index, f, ensure_ascii=False, indent=2)
    print(f"  → 保存至 {INDEX_OUTPUT}")

    # 6. 更新 apsma_knowledge_validated.jsonl
    jsonl_input = OUTPUT_DIR / "apsma_knowledge_validated.jsonl"
    jsonl_output = OUTPUT_DIR / "apsma_knowledge_validated_v2.jsonl"
    if jsonl_input.exists():
        print("\n[额外] 更新 apsma_knowledge_validated.jsonl...")
        update_apsma_jsonl(jsonl_input, apsma_names, jsonl_output)

    # 打印摘要
    print("\n" + "=" * 60)
    print("摘要")
    print("=" * 60)
    stats = complete_index["statistics"]
    print(f"  总条目: {stats['total_entries']}")
    print(f"  形态型(APSMA): {stats['total_morphotypes']}")
    print(f"  总图片: {stats['total_images']}")
    print(f"  学名覆盖: {stats['scientific_name_coverage']}")

    # 打印前10个 APSMA 学名示例
    print("\n  APSMA 学名示例:")
    for entry in complete_index["apsma_entries"][:10]:
        print(f"    {entry['morphotype_id']:10s} → {entry['scientific_name']}")

    # 打印前10个西太平洋物种示例
    print("\n  西太平洋物种示例:")
    for sp in westpac_species[:10]:
        print(f"    {sp['seq']:3d}. {sp['name_cn']} | {sp['name_latin']}")

    print(f"\n输出文件:")
    print(f"  1. {INDEX_OUTPUT}")
    print(f"  2. {apsma_names_path}")
    print(f"  3. {westpac_names_path}")
    if jsonl_input.exists():
        print(f"  4. {jsonl_output}")


if __name__ == "__main__":
    main()
