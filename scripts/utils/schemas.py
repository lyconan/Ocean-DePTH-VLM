"""DePTH-VLM Pydantic数据模型 — 严格KG构建Schema + 宽松推理Schema (v3.1.2)"""
import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

# CCZ深渊大动物有效门
# 2026-07-17 扩展: 添加 Bryozoa/Ctenophora/Nemertea,以支持 APSMA 图集原始分类
# (APSMA 北太平洋图集包含这些门级形态型,虽非CCZ核心,但需保留以构建完整KG)
VALID_PHYLA = {
    "Porifera", "Echinodermata", "Cnidaria", "Annelida",
    "Arthropoda", "Xenophyophorea", "Mollusca", "Brachiopoda",
    "Hemichordata", "Chordata", "Nematoda", "Foraminifera",
    # APSMA 图集扩展门级
    "Bryozoa",      # 苔藓动物门
    "Ctenophora",   # 栉水母动物门
    "Nemertea",     # 纽形动物门
}

# 拉丁名/形态型ID模式
# 允许: Genus_species, Genus_species_Mo01, 也允许空格分隔的 binomial name
LATIN_NAME_PATTERN = re.compile(r"^[A-Z][a-z]+([ _][a-z]+)*(_Mo\d+)?$")
MORPHOTYPE_ID_PATTERN = re.compile(r"^APSMA_V\d+_[A-Z][a-z]+_[A-Z][a-z]+_Mo\d+$")

# VLM未知哨兵值
UNKNOWN_SENTINEL = "unknown"

# VLM可能输出的退化值集合
DEGENERATE_VALUES = {
    "", " ", "none", "null", "n/a", "na", "not found", "not_found",
    "unidentified", "unidentified.", "uncertain", "unclear",
    "n/a.", "-", "--", "???", "unknown",
}

VALID_PHYLA_OR_UNKNOWN = VALID_PHYLA | {UNKNOWN_SENTINEL}
VALID_CONFIDENCE = {"LOW", "MEDIUM", "HIGH"}


def _normalize_taxonomy_field(v) -> str:
    """将分类字段归一化为干净字符串或unknown哨兵值。"""
    if v is None:
        return UNKNOWN_SENTINEL
    if isinstance(v, (int, float)):
        return UNKNOWN_SENTINEL
    s = str(v).strip()
    if s.lower() in DEGENERATE_VALUES:
        return UNKNOWN_SENTINEL
    return s


# ═══════════════════════════════════════════════════════════
# 严格Schema — KG构建期使用
# ═══════════════════════════════════════════════════════════

class TaxonomyNode(BaseModel):
    """严格分类节点 — KG构建使用"""
    phylum: str
    class_name: str = Field(alias="class")
    order: str
    morphotype: str

    @field_validator("phylum")
    @classmethod
    def validate_phylum(cls, v):
        if v not in VALID_PHYLA:
            raise ValueError(f"Unknown phylum: '{v}'. Must be one of {VALID_PHYLA}")
        return v

    @field_validator("morphotype")
    @classmethod
    def validate_morphotype_name(cls, v):
        if not LATIN_NAME_PATTERN.match(v):
            raise ValueError(f"Invalid morphotype name format: '{v}'")
        return v


class MeasurementRange(BaseModel):
    """测量范围"""
    min_cm: float = Field(ge=0, le=500)
    max_cm: float = Field(ge=0, le=500)

    @field_validator("max_cm")
    @classmethod
    def validate_range(cls, v, info):
        if "min_cm" in info.data and v < info.data["min_cm"]:
            raise ValueError(f"max_cm ({v}) < min_cm ({info.data['min_cm']})")
        return v


class APSMARecord(BaseModel):
    """APSMA形态型记录 — KG构建严格验证"""
    morphotype_id: str
    taxonomy: TaxonomyNode
    features: dict
    image: dict
    occurrence: dict
    source: dict

    @field_validator("morphotype_id")
    @classmethod
    def validate_morphotype_id(cls, v):
        if not MORPHOTYPE_ID_PATTERN.match(v):
            raise ValueError(f"Invalid morphotype ID format: '{v}'")
        return v


# ═══════════════════════════════════════════════════════════
# 宽松Schema — VLM推理期使用 (v3.1.2)
# ═══════════════════════════════════════════════════════════

class VLMClassificationResult(BaseModel):
    """推理期宽松Schema — VLM分类输出 (v3.1.2)
    
    与严格Schema的区别：
    1. 每个分类字段都容忍unknown哨兵值
    2. field_validator在验证前先归一化退化输入
    3. 任何unknown字段强制confidence=LOW
    """
    target_id: int
    phylum: str = UNKNOWN_SENTINEL
    class_name: str = Field(default=UNKNOWN_SENTINEL, alias="class")
    order: str = UNKNOWN_SENTINEL
    morphotype: str = UNKNOWN_SENTINEL
    top3_morphotypes: list = []
    confidence: str = "LOW"
    reasoning: str = ""
    diagnostics: str = ""

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("phylum", "order", "morphotype", "reasoning", "diagnostics", mode="before")
    @classmethod
    def _normalize_str_fields(cls, v):
        return _normalize_taxonomy_field(v)

    @field_validator("class_name", mode="before")
    @classmethod
    def _normalize_class(cls, v):
        return _normalize_taxonomy_field(v)

    @field_validator("phylum")
    @classmethod
    def _validate_phylum(cls, v):
        if v not in VALID_PHYLA_OR_UNKNOWN:
            return UNKNOWN_SENTINEL
        return v

    @field_validator("morphotype")
    @classmethod
    def _validate_morphotype(cls, v):
        if v == UNKNOWN_SENTINEL:
            return v
        # 将空格转下划线以匹配标准格式 (Asthenactis fisheri → Asthenactis_fisheri)
        v_normalized = v.replace(" ", "_")
        if not LATIN_NAME_PATTERN.match(v_normalized):
            return UNKNOWN_SENTINEL
        return v_normalized

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v):
        v = _normalize_taxonomy_field(v)
        if v == UNKNOWN_SENTINEL:
            return "LOW"
        upper = v.upper()
        return upper if upper in VALID_CONFIDENCE else "LOW"

    @model_validator(mode="after")
    def _force_low_confidence_on_unknown(self):
        """任何unknown分类字段 → 强制confidence=LOW，防止'自信但错误'"""
        if any(getattr(self, f) == UNKNOWN_SENTINEL
               for f in ("phylum", "class_name", "order", "morphotype")):
            if self.confidence != "LOW":
                self.confidence = "LOW"
                self.diagnostics = (self.diagnostics + " "
                                    "[auto-downgraded: incomplete taxonomy]").strip()
        return self


def normalize_vlm_result(raw: dict) -> dict:
    """通过VLMClassificationResult归一化VLM原始输出 (v3.1.2 chokepoint)"""
    if not isinstance(raw, dict):
        return {"target_id": -1, "phylum": UNKNOWN_SENTINEL,
                "class": UNKNOWN_SENTINEL, "order": UNKNOWN_SENTINEL,
                "morphotype": UNKNOWN_SENTINEL, "confidence": "LOW",
                "reasoning": "", "diagnostics": "non_dict_vlm_output"}

    try:
        if "target_id" not in raw:
            raw = {**raw, "target_id": -1}
        elif not isinstance(raw["target_id"], int):
            try:
                raw = {**raw, "target_id": int(raw["target_id"])}
            except (TypeError, ValueError):
                raw = {**raw, "target_id": -1}

        result = VLMClassificationResult(**raw)
        return {
            "target_id": result.target_id,
            "phylum": result.phylum,
            "class": result.class_name,
            "order": result.order,
            "morphotype": result.morphotype,
            "top3_morphotypes": result.top3_morphotypes,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "diagnostics": result.diagnostics or "normalized",
        }
    except Exception as e:
        print(f"[NORMALIZE FALLBACK] target_id={raw.get('target_id')}: {e}")
        return {"target_id": raw.get("target_id", -1),
                "phylum": UNKNOWN_SENTINEL,
                "class": UNKNOWN_SENTINEL,
                "order": UNKNOWN_SENTINEL,
                "morphotype": UNKNOWN_SENTINEL,
                "confidence": "LOW",
                "reasoning": str(raw)[:200],
                "diagnostics": f"normalize_failed: {type(e).__name__}"}
