"""FathomNet概念 → APSMA对齐YOLO类别映射"""
import csv
import json
from pathlib import Path
from typing import Optional

# APSMA门级粗分类映射
APSMA_COARSE_CLASS_MAP = {
    "Porifera": 0,
    "Echinodermata": 1,
    "Cnidaria": 2,
    "Annelida": 3,
    "Arthropoda": 4,
    "Xenophyophorea": 5,
    "Mollusca": 6,
    "Brachiopoda": 7,
    "Foraminifera": 8,
}

# FathomNet概念 → 门级映射 (基础映射)
_CONCEPT_TO_PHYLUM_BASE = {
    # Porifera
    "Hexactinellida": "Porifera", "Demospongiae": "Porifera", "Porifera": "Porifera",
    # Echinodermata
    "Ophiuroidea": "Echinodermata", "Asteroidea": "Echinodermata",
    "Holothuroidea": "Echinodermata", "Echinoidea": "Echinodermata",
    "Crinoidea": "Echinodermata",
    # Cnidaria
    "Ceriantharia": "Cnidaria", "Actiniaria": "Cnidaria",
    "Hydrozoa": "Cnidaria", "Alcyonacea": "Cnidaria", "Antipatharia": "Cnidaria",
    # Annelida
    "Serpulidae": "Annelida", "Sabellidae": "Annelida", "Polychaeta": "Annelida",
    # Arthropoda
    "Caridea": "Arthropoda", "Anomura": "Arthropoda",
    "Brachyura": "Arthropoda", "Amphipoda": "Arthropoda",
    # Xenophyophorea
    "Xenophyophorea": "Xenophyophorea",
    # Mollusca
    "Gastropoda": "Mollusca", "Bivalvia": "Mollusca", "Octopoda": "Mollusca",
    # Brachiopoda
    "Brachiopoda": "Brachiopoda",
    # Foraminifera
    "Foraminifera": "Foraminifera",
}

# 扩展映射: FathomNet 类别的属/种/科级 → 门级
# 通过 WoRMS API 和分类学知识构建
_CONCEPT_TO_PHYLUM_EXTENDED = {
    # Echinodermata - 属级
    "Strongylocentrotus": "Echinodermata", "Scotoplanes": "Echinodermata",
    "Psolus": "Echinodermata", "Hippasteria": "Echinodermata",
    "Rathbunaster": "Echinodermata", "Pycnopodia": "Echinodermata",
    "Crossaster": "Echinodermata", "Solaster": "Echinodermata",
    "Zoroaster": "Echinodermata", "Luidia": "Echinodermata",
    "Ceramaster": "Echinodermata", "Pteraster": "Echinodermata",
    "Henricia": "Echinodermata", "Brisingidae": "Echinodermata",
    "Gorgonocephalus": "Echinodermata", "Asteronyx": "Echinodermata",
    "Peniagone": "Echinodermata", "Molpadia": "Echinodermata",
    "Oneirophanta": "Echinodermata", "Psychropotes": "Echinodermata",
    "Benthodytes": "Echinodermata", "Mesothuria": "Echinodermata",
    "Pannychia": "Echinodermata", "Synallactes": "Echinodermata",
    "Aporocidaris": "Echinodermata", "Fariometra": "Echinodermata",
    "Lophaster": "Echinodermata", "Dipsacaster": "Echinodermata",
    "Stomphia": "Echinodermata", "Pterasteridae": "Echinodermata",
    "Antedonoidea": "Echinodermata", "Thenea": "Porifera",
    # Porifera - 属级
    "Heterochone": "Porifera", "Farrea": "Porifera",
    "Staurocalyptus": "Porifera", "Acanthascinae": "Porifera",
    "Chonelasma": "Porifera", "Walteria": "Porifera",
    "Caulophacus": "Porifera", "Sericolophus": "Porifera",
    "Hyalonema": "Porifera", "Aulosaccus": "Porifera",
    "Bathydorus": "Porifera", "Rhabdocalyptus": "Porifera",
    "Aphrocallistes": "Porifera", "Euplectella": "Porifera",
    "Saccocalyx": "Porifera", "Cladorhiza": "Porifera",
    "Abyssocladia": "Porifera", "Psamminidae": "Porifera",
    "Corbitellinae": "Porifera",
    # Cnidaria - 属级
    "Paragorgia": "Cnidaria", "Keratoisis": "Cnidaria",
    "Swiftia": "Cnidaria", "Chrysogorgia": "Cnidaria",
    "Primnoa": "Cnidaria", "Bolocera": "Cnidaria",
    "Actinostola": "Cnidaria", "Hormathia": "Cnidaria",
    "Metridium": "Cnidaria", "Urticina": "Cnidaria",
    "Liponema": "Cnidaria", "Anthomastus": "Cnidaria",
    "Heteropolypus": "Cnidaria", "Funiculina": "Cnidaria",
    "Pennatula": "Cnidaria", "Umbellula": "Cnidaria",
    "Acanthoptilum": "Cnidaria", "Ptychogastria": "Cnidaria",
    "Solmaris": "Cnidaria", "Pantachogon": "Cnidaria",
    "Mitrocoma": "Cnidaria", "Eutonina": "Cnidaria",
    "Aurelia": "Cnidaria", "Atolla": "Cnidaria",
    "Periphylla": "Cnidaria", "Stygiomedusa": "Cnidaria",
    "Poralia": "Cnidaria", "Desmophyes": "Cnidaria",
    "Apolemia": "Cnidaria", "Erenna": "Cnidaria",
    "Vogtia": "Cnidaria", "Sicyonis": "Cnidaria",
    "Amphianthus": "Cnidaria", "Metallogorgia": "Cnidaria",
    "Iridogorgia": "Cnidaria", "Jasonisis": "Cnidaria",
    "Sibogagorgia": "Cnidaria", "Leptogorgia": "Cnidaria",
    "Candidella": "Cnidaria", "Medusae": "Cnidaria",
    "Tomopteris": "Annelida", "Modeeria": "Cnidaria",
    "Halicreas": "Cnidaria", "Stellamedusa": "Cnidaria",
    "Lipkea": "Cnidaria", "Epizoanthus": "Cnidaria",
    "Ctenophora": "Ctenophora",
    # Annelida
    "Poeobius": "Annelida", "Hirudinea": "Annelida",
    "Lamellibrachia": "Annelida", "Riftia": "Annelida",
    "Osedax": "Annelida", "Echiura": "Annelida",
    # Arthropoda
    "Chionoecetes": "Arthropoda", "Munidopsis": "Arthropoda",
    "Galatheoidea": "Arthropoda", "Culeolus": "Arthropoda",
    "Uroptychus": "Arthropoda", "Pandalopsis": "Arthropoda",
    "Plesionika": "Arthropoda", "Glyphocrangon": "Arthropoda",
    "Acanthephyra": "Arthropoda", "Paralomis": "Arthropoda",
    "Lithodes": "Arthropoda", "Chorilia": "Arthropoda",
    "Acanthamunnopsis": "Arthropoda", "Hyperiidae": "Arthropoda",
    "Neoloricata": "Arthropoda", "Mysidae": "Arthropoda",
    "Euphausia": "Arthropoda", "Boreomysis": "Arthropoda",
    "Pycnogonida": "Arthropoda", "Isopoda": "Arthropoda",
    "Copepoda": "Arthropoda",
    # Mollusca
    "Octopodidae": "Mollusca", "Muusoctopus": "Mollusca",
    "Gonatus": "Mollusca", "Onykia": "Mollusca",
    "Galiteuthis": "Mollusca", "Leachia": "Mollusca",
    "Planctoteuthis": "Mollusca", "Abraliopsis": "Mollusca",
    "Gonatopsis": "Mollusca", "Bathyteuthis": "Mollusca",
    "Doryteuthis": "Mollusca", "Acesta": "Mollusca",
    "Calyptogena": "Mollusca", "Calliostoma": "Mollusca",
    "Bathybembix": "Mollusca", "Myxoderma": "Mollusca",
    "Pleurobranchaea": "Mollusca", "Vitreosalpa": "Chordata",
    "Ptychogena": "Chordata", "Salpa": "Chordata",
    "Procymbulia": "Chordata",
    # Chordata (鱼类/被囊类)
    "Sebastes": "Chordata", "Sebastolobus": "Chordata",
    "Sebastomus": "Chordata", "Sebastidae": "Chordata",
    "Coryphaenoides": "Chordata", "Bathyraja": "Chordata",
    "Ophiodon": "Chordata", "Physiculus": "Chordata",
    "Macropinna": "Chordata", "Melanocetus": "Chordata",
    "Bathysaurus": "Chordata", "Cherublemma": "Chordata",
    "Spectrunculus": "Chordata", "Lycodapus": "Chordata",
    "Lycenchelys": "Chordata", "Eretmichthys": "Chordata",
    "Anoplogaster": "Chordata", "Icichthys": "Chordata",
    "Psychrolutes": "Chordata", "Hydrolagus": "Chordata",
    "Rajiformes": "Chordata", "Ophidiidae": "Chordata",
    "Coelorinchus": "Chordata", "Bathypterois": "Chordata",
    "Merluccius": "Chordata", "Tunicata": "Chordata",
    "Ascidiacea": "Chordata", "Megalodicopia": "Chordata",
    "Octacnemidae": "Chordata",
    # Foraminifera
    "Bathysiphon": "Foraminifera",
    # 其他
    "Cydippida": "Ctenophora",
}

# 34 纲 YOLO 类别 -> 门 (yolo11m_class34, 2026-09)
_CLASS_TO_PHYLUM_34 = {
    "Octocorallia": "Cnidaria", "Hexacorallia": "Cnidaria",
    "Hydrozoa": "Cnidaria", "Scyphozoa": "Cnidaria",
    "Malacostraca": "Arthropoda", "Thecostraca": "Arthropoda",
    "Pycnogonida": "Arthropoda",
    "Holothuroidea": "Echinodermata", "Echinoidea": "Echinodermata",
    "Asteroidea": "Echinodermata", "Crinoidea": "Echinodermata",
    "Ophiuroidea": "Echinodermata",
    "Hexactinellida": "Porifera", "Demospongiae": "Porifera",
    "Actinopterygii": "Chordata", "Elasmobranchii": "Chordata",
    "Holocephali": "Chordata", "Myxini": "Chordata",
    "Thaliacea": "Chordata", "Appendicularia": "Chordata",
    "Ascidiacea": "Chordata",
    "Polychaeta": "Annelida", "Clitellata": "Annelida",
    "Cephalopoda": "Mollusca", "Bivalvia": "Mollusca",
    "Gastropoda": "Mollusca", "Polyplacophora": "Mollusca",
    "Tentaculata": "Ctenophora", "Nuda": "Ctenophora",
    "Rhynchonellata": "Brachiopoda",
    "Enteropneusta": "Hemichordata",
    "Gymnolaemata": "Bryozoa",
}

# 合并映射
CONCEPT_TO_PHYLUM = {**_CLASS_TO_PHYLUM_34, **_CONCEPT_TO_PHYLUM_BASE, **_CONCEPT_TO_PHYLUM_EXTENDED}

# 尝试加载外部 WoRMS 映射文件 (如果存在)
_WORMS_MAPPING_PATH = Path(__file__).parent.parent.parent / "data/processed/fathomnet_phylum_mapping.json"
if _WORMS_MAPPING_PATH.exists():
    try:
        with open(_WORMS_MAPPING_PATH) as _f:
            _worms_mapping = json.load(_f)
        for _name, _phylum in _worms_mapping.items():
            if _phylum not in ("unknown", "error", "non-marine"):
                CONCEPT_TO_PHYLUM[_name] = _phylum
    except Exception:
        pass


def map_to_coarse_class(
    concept_name: str,
    alignment_csv: Optional[str] = None,
) -> int:
    """映射FathomNet概念到APSMA对齐的YOLO类别ID。
    
    优先级:
    1. 查找对齐矩阵 → 返回APSMA对齐类别ID
    2. 回退到门级APSMA映射
    """
    # 未知门映射到专门的"unknown"类别ID(= nc，即最后一个类别ID之后)
    UNKNOWN_CLASS_ID = len(APSMA_COARSE_CLASS_MAP)  # 9，超出已知9个门

    if alignment_csv:
        apsma_match = lookup_alignment_matrix(concept_name, alignment_csv)
        if apsma_match and "apsma_phylum" in apsma_match:
            return APSMA_COARSE_CLASS_MAP.get(
                apsma_match["apsma_phylum"],
                UNKNOWN_CLASS_ID
            )
    
    phylum = infer_phylum_from_concept(concept_name)
    return APSMA_COARSE_CLASS_MAP.get(phylum, UNKNOWN_CLASS_ID)


def infer_phylum_from_concept(concept_name: str) -> str:
    """从FathomNet概念名推断门级分类。"""
    # 直接匹配
    if concept_name in CONCEPT_TO_PHYLUM:
        return CONCEPT_TO_PHYLUM[concept_name]
    
    # 部分匹配
    for key, phylum in CONCEPT_TO_PHYLUM.items():
        if key.lower() in concept_name.lower() or concept_name.lower() in key.lower():
            return phylum
    
    return "unknown"


def lookup_alignment_matrix(concept_name: str, alignment_csv: str) -> Optional[dict]:
    """查找WoRMS↔APSMA对齐矩阵。

    使用大小写不敏感匹配，与00c_align_annotations_to_apsma.py一致。
    同时检查 matched_rank 和 apsma_morphotype_id 字段。
    """
    try:
        with open(alignment_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("matched_rank", "").lower() == concept_name.lower():
                    return dict(row)
                if row.get("apsma_morphotype_id", "").lower() == concept_name.lower():
                    return dict(row)
    except FileNotFoundError:
        pass
    return None
