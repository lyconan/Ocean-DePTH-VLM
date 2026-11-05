#!/usr/bin/env python3
"""批量查询 WoRMS API 获取 FathomNet 622 类的门级分类映射。

用法:
    python scripts/07g_worms_phylum_mapping.py
    python scripts/07g_worms_phylum_mapping.py --update-neo4j
"""
import json
import sys
import ssl
import time
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from utils.taxonomy_mapper import infer_phylum_from_concept

CLASSES_PATH = Path(__file__).parent.parent / "data/processed/fathomnet_finegrained_yolo/classes.names"
OUTPUT_PATH = Path(__file__).parent.parent / "data/processed/fathomnet_phylum_mapping.json"

# 非生物类 → 人工标注
NON_BIOLOGICAL = {
    "Suction Sampler": "equipment", "Push Corer": "equipment",
    "manipulator": "equipment", "equipment": "equipment",
    "DeepPIV 3.0": "equipment", "DeepPIV 1.0": "equipment",
    "Benthic Respiration System": "equipment", "Biology box": "equipment",
    "CO2 Corral": "equipment", "mini container array": "equipment",
    "Detritus Sampler": "equipment", "trap": "equipment",
    "55-gallon drum": "equipment", "ship container": "equipment",
    "Krill molt": "detritus", "eggcase": "detritus",
    "shell fragment": "detritus", "shell": "detritus", "skeleton": "detritus",
    "rock": "substrate", "talus": "substrate", "pillow lava": "substrate",
    "bacterial mat": "substrate", "Phyllospadix-Zostera detritus": "detritus",
    "cable": "equipment", "polypropylene line": "equipment",
    "plastic bag": "trash", "plastic": "trash", "trash": "trash",
    "marine organism": "unknown", "amphipod tube mat": "Annelida",
    "coral skeleton": "Cnidaria", "salp chain": "Chordata",
    "site marker": "equipment", "Akoya platinum": "equipment",
    "narella sp.": "Cnidaria",  # Narella is a primnoid coral genus
    "sinker": "equipment",
}

# 分类学推断 — 科/目级 → 门级
TAXON_GROUP_MAP = {
    # Cnidaria
    "Isididae": "Cnidaria", "Paragorgiidae": "Cnidaria", "Primnoidae": "Cnidaria",
    "Plexauridae": "Cnidaria", "Hormathiidae": "Cnidaria", "Actinostolidae": "Cnidaria",
    "Cerianthidae": "Cnidaria", "Actiniidae": "Cnidaria", "Corallimorphidae": "Cnidaria",
    "Clavulariidae": "Cnidaria", "Buccinidae": "Mollusca",
    "Octocorallia": "Cnidaria", "Hexacorallia": "Cnidaria", "Anthozoa": "Cnidaria",
    "Zoantharia": "Cnidaria", "Scleractinia": "Cnidaria",
    "Siphonophorae": "Cnidaria", "Calycophorae": "Cnidaria", "Physonectae": "Cnidaria",
    "Actinostoloidea": "Cnidaria", "Actinopterygii": "Chordata",
    # Echinodermata
    "Synallactidae": "Echinodermata", "Psolidae": "Echinodermata",
    "Pedicellasteridae sp. 2": "Echinodermata", "Antedonidae": "Echinodermata",
    "Comatulida": "Echinodermata", "Comatulidina": "Echinodermata",
    "Euryalidae": "Echinodermata", "Ophiuridae": "Echinodermata",
    "Ophiacanthidae": "Echinodermata", "Goniasteridae": "Echinodermata",
    # Arthropoda
    "Munnopsidae": "Arthropoda", "Lithodidae": "Arthropoda",
    "Galatheidae": "Arthropoda", "Pandalidae": "Arthropoda",
    "Majidae": "Arthropoda", "Cancridae sp. 1": "Arthropoda",
    "Paguroidea": "Arthropoda", "Cirripedia": "Arthropoda",
    "Malacostraca": "Arthropoda", "Crustacea": "Arthropoda",
    "Decapoda": "Arthropoda",
    # Mollusca
    "Vesicomyidae": "Mollusca", "Pectinidae": "Mollusca",
    "Teuthoidea": "Mollusca", "Pteropoda": "Mollusca",
    # Annelida
    "Terebellidae": "Annelida", "Polynoidae": "Annelida",
    "Siboglinidae": "Annelida",
    # Chordata
    "Macrouridae": "Chordata", "Liparidae": "Chordata", "Zoarcidae": "Chordata",
    "Salpida": "Chordata", "Appendicularia": "Chordata",
    # Porifera
    "Cladorhizidae": "Porifera", "Euplectellidae": "Porifera",
    # Ctenophora
    "Lampocteis cruentiventer": "Ctenophora",
    # Chaetognatha
    "Chaetognatha": "Chaetognatha", "Caecosagitta macrocephala": "Chaetognatha",
    # Hemichordata
    "Enteropneusta": "Hemichordata", "Tergivelum baldwinae": "Hemichordata",
    # Radiozoa (Radiolaria)
    "Radiozoa": "Radiozoa",
    # Brachiopoda
    "Terebratulina complex": "Brachiopoda", "Laqueus californicus": "Brachiopoda",
    "Laqueus": "Brachiopoda",
    # Provannidae (Gastropoda)
    "Provannidae": "Mollusca",
}


def query_worms_phylum(name: str, ctx: ssl.SSLContext) -> str:
    """通过 WoRMS API 查询物种的门级分类。"""
    try:
        url = f"https://www.marinespecies.org/rest/AphiaIDByName/{urllib.parse.quote(name)}?marine_only=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Ocean-DePTH-VLM/1.0"})
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            aphia_id = json.loads(resp.read())
        if not aphia_id or aphia_id <= 0:
            return None

        url2 = f"https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/{aphia_id}"
        with urllib.request.urlopen(url2, timeout=15, context=ctx) as resp:
            cls = json.loads(resp.read())

        while cls:
            if cls.get("rank") == "Phylum":
                return cls["scientificname"]
            cls = cls.get("child")
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-neo4j", action="store_true")
    parser.add_argument("--batch-delay", type=float, default=0.15, help="API 请求间隔(秒)")
    args = parser.parse_args()

    # 加载 622 类
    with open(CLASSES_PATH) as f:
        classes = [l.strip() for l in f if l.strip()]

    # 筛选 unknown
    unknown = [cls for cls in classes if infer_phylum_from_concept(cls) == "unknown"]
    print(f"需查询: {len(unknown)}/{len(classes)} 个 unknown 类别")

    # SSL context
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # 加载已有映射
    existing = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        print(f"已有映射: {len(existing)} 条")

    # 逐步查询
    results = dict(existing)
    worms_hits = 0
    local_hits = 0
    api_errors = 0

    for i, name in enumerate(unknown):
        # 1. 优先用本地映射
        if name in NON_BIOLOGICAL:
            phylum = NON_BIOLOGICAL[name]
            results[name] = phylum
            local_hits += 1
            continue

        if name in TAXON_GROUP_MAP:
            phylum = TAXON_GROUP_MAP[name]
            results[name] = phylum
            local_hits += 1
            continue

        # 2. 检查已有映射
        if name in results and results[name] not in ("unknown", "error", "non-marine"):
            local_hits += 1
            continue

        # 3. WoRMS API 查询
        phylum = query_worms_phylum(name, ctx)
        if phylum:
            results[name] = phylum
            worms_hits += 1
        else:
            results[name] = "unknown"
            api_errors += 1

        time.sleep(args.batch_delay)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(unknown)}] WoRMS={worms_hits}, 本地={local_hits}, 失败={api_errors}")

    # 保存映射
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n映射保存: {OUTPUT_PATH} ({len(results)} 条)")

    # 统计
    counts = Counter(results.values())
    print(f"\n门级分布:")
    for p, c in counts.most_common():
        print(f"  {p}: {c}")

    bio_total = sum(c for p, c in counts.items() if p not in ("equipment", "substrate", "detritus", "trash"))
    bio_known = sum(c for p, c in counts.items() if p not in ("unknown", "equipment", "substrate", "detritus", "trash", "error", "non-marine"))
    print(f"\n生物类别: {bio_known}/{bio_total} 已映射 ({bio_known/bio_total*100:.1f}%)")

    # 更新 Neo4j
    if args.update_neo4j:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://172.17.0.1:7687", auth=("neo4j", "password"))
        updated = 0
        with driver.session() as session:
            for name, phylum in results.items():
                if phylum in ("unknown", "error", "non-marine", "equipment", "substrate", "detritus", "trash"):
                    continue
                mid = f"FN_{name}"
                r = session.run(
                    "MATCH (m:Morphotype {morphotype_id: $mid}) SET m.phylum_name = $phylum RETURN m.morphotype_id",
                    mid=mid, phylum=phylum
                )
                if r.single():
                    updated += 1
        driver.close()
        print(f"\nNeo4j 更新: {updated} 个节点 phylum_name 已更新")


if __name__ == "__main__":
    import argparse
    main()
