#!/usr/bin/env python3
"""两阶段修复: 先导入 WoRMS 描述,再统一清理所有 OCR/LaTeX 垃圾。"""

import re
import subprocess
from neo4j import GraphDatabase

NEO4J_URI = "bolt://172.17.0.1:7687"
NEO4J_AUTH = ("neo4j", "password")

BIO_KEYWORDS = [
    "body", "tentacle", "polyp", "colony", "spine", "tube", "shell",
    "segment", "setae", "cirri", "radial", "oral", "specimen",
    "diameter", "length", "width", "color", "surface", "description",
    "morphology", "character", "morph", "dorsal", "ventral", "anterior",
    "posterior", "lateral", "margin", "papilla", "annuli",
    "pedicel", "ossicle", "calyx", "theca", "corallite", "zooecium",
    "lophophore", "proboscis", "peristomium", "prostomium", "parapodium",
    "elytra", "palea", "uncinus", "thorax", "abdomen", "carapace",
    "rostrum", "antenna", "chela", "pereopod", "pleopod", "uropod",
    "scute", "ambulacral", "adambulacral", "madreporite", "pedicellaria",
    "habitat", "depth", "distribution", "range", "species", "genus",
    "type", "holotype", "syntype", "locality", "record", "occurrence",
]


def clean_description(desc: str) -> str:
    """清理 OCR/LaTeX 垃圾,返回干净描述或空字符串。"""
    if not desc:
        return ""
    # 1. 去除 Markdown 图片引用: ![...](images/hash.jpg)
    clean = re.sub(r"!\[.*?\]\(images/[a-f0-9]+\.jpg\)", "", desc)
    # 2. 去除残留的图片哈希: hash.jpg) 或 hash.jpg 模式 (8+位hex,覆盖短哈希)
    clean = re.sub(r"[a-f0-9]{8,}\.jpg\)?", "", clean)
    # 3. 去除 LaTeX 数学: $...$
    clean = re.sub(r"\$[^$]+\$", "", clean)
    # 4. 去除 LaTeX 命令: \command{...} 或 \command
    clean = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", clean)
    clean = re.sub(r"\\[a-zA-Z]+", "", clean)
    # 5. 去除 LaTeX 花括号残留
    clean = re.sub(r"[{}]", "", clean)
    # 6. 去除多余空白
    clean = re.sub(r"\s+", " ", clean).strip()
    # 7. 质量检查
    has_bio = any(kw in clean.lower() for kw in BIO_KEYWORDS)
    if len(clean) < 30 or not has_bio:
        return ""
    return clean[:500]


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    # === 第 1 步: 先导入 WoRMS 描述 ===
    print("=" * 60)
    print("第 1 步: 导入 WoRMS 描述")
    print("=" * 60)
    driver.close()

    result = subprocess.run(
        ["python", "scripts/utils/import_worms_descriptions.py", "--batch-size", "200"],
        capture_output=True, text=True,
        cwd="/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM"
    )
    print(result.stdout)

    # === 第 2 步: 统一清理所有 OCR/LaTeX 垃圾 ===
    print("=" * 60)
    print("第 2 步: 统一清理 OCR/LaTeX 垃圾")
    print("=" * 60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as s:
        all_records = list(s.run("""
            MATCH (m:Morphotype)
            WHERE m.description IS NOT NULL AND m.description <> ""
            RETURN m.morphotype_id AS id, m.description AS d
        """))
        print(f"有描述的节点: {len(all_records):,}")

        # 找所有含垃圾的描述
        junk_patterns = [".jpg)", ".jpg ", "\\mathrm", "\\text", "\\\\"]
        junk_records = [r for r in all_records
                        if any(p in r["d"] for p in junk_patterns)]
        print(f"含OCR/LaTeX垃圾: {len(junk_records):,}")

        cleaned_keep = 0
        cleaned_clear = 0
        updates = []
        for r in junk_records:
            result_desc = clean_description(r["d"])
            updates.append({"id": r["id"], "desc": result_desc})
            if result_desc:
                cleaned_keep += 1
            else:
                cleaned_clear += 1

        print(f"\n清理结果:")
        print(f"  清理后保留(有生物学术语): {cleaned_keep}")
        print(f"  清空(无有效内容): {cleaned_clear}")

        if updates:
            for i in range(0, len(updates), 200):
                batch = updates[i:i + 200]
                s.run("""
                    UNWIND $batch AS row
                    MATCH (m:Morphotype {morphotype_id: row.id})
                    SET m.description = row.desc
                """, batch=batch)
            print(f"  已更新: {len(updates)}")

    driver.close()

    # === 第 3 步: 最终统计 ===
    print("\n" + "=" * 60)
    print("第 3 步: 最终统计")
    print("=" * 60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as s:
        total = s.run("MATCH (m:Morphotype) RETURN count(m) AS c").single()["c"]
        print(f"总节点: {total:,}")

        for src, prefix in [("APSMA", "APSMA_V1"), ("WoRMS", "WORMS_"), ("FathomNet", "FN_")]:
            cnt = s.run(f'MATCH (m:Morphotype) WHERE m.morphotype_id STARTS WITH "{prefix}" RETURN count(m) AS c').single()["c"]
            has_desc = s.run(f'MATCH (m:Morphotype) WHERE m.morphotype_id STARTS WITH "{prefix}" AND m.description IS NOT NULL AND m.description <> "" RETURN count(m) AS c').single()["c"]
            has_phylum = s.run(f'MATCH (m:Morphotype) WHERE m.morphotype_id STARTS WITH "{prefix}" AND ((m.phylum IS NOT NULL AND m.phylum <> "") OR (m.phylum_name IS NOT NULL AND m.phylum_name <> "")) RETURN count(m) AS c').single()["c"]
            has_class = s.run(f'MATCH (m:Morphotype) WHERE m.morphotype_id STARTS WITH "{prefix}" AND ((m.class IS NOT NULL AND m.class <> "") OR (m.class_name IS NOT NULL AND m.class_name <> "")) RETURN count(m) AS c').single()["c"]
            print(f"\n  {src}: {cnt:,}个")
            print(f"    有描述: {has_desc:,} ({has_desc/cnt*100:.1f}%)")
            print(f"    无描述: {cnt - has_desc:,}")
            print(f"    有phylum: {has_phylum:,} ({has_phylum/cnt*100:.1f}%)")
            print(f"    有class: {has_class:,} ({has_class/cnt*100:.1f}%)")

        # 垃圾残留检查
        ocr_junk = s.run(
            'MATCH (m:Morphotype) WHERE m.description CONTAINS $p RETURN count(m) AS c',
            p=".jpg)"
        ).single()["c"]
        # LaTeX: 查包含 \mathrm 或 \text 的
        latex_junk = s.run(
            'MATCH (m:Morphotype) WHERE m.description CONTAINS $p1 OR m.description CONTAINS $p2 RETURN count(m) AS c',
            p1="\\mathrm", p2="\\text"
        ).single()["c"]
        print(f"\n残留OCR垃圾(.jpg): {ocr_junk}")
        print(f"残留LaTeX垃圾(\\mathrm/\\text): {latex_junk}")

        total_desc = s.run('MATCH (m:Morphotype) WHERE m.description IS NOT NULL AND m.description <> "" RETURN count(m) AS c').single()["c"]
        print(f"\n总有效描述: {total_desc:,} / {total:,} ({total_desc/total*100:.1f}%)")

        # 描述质量样本
        print("\n--- 描述质量样本 ---")
        for src, prefix in [("APSMA", "APSMA_V1"), ("WoRMS", "WORMS_")]:
            print(f"\n  [{src}]")
            samples = list(s.run(f'''
                MATCH (m:Morphotype)
                WHERE m.morphotype_id STARTS WITH "{prefix}"
                  AND m.description IS NOT NULL AND m.description <> ""
                RETURN m.morphotype_id AS id, m.description AS d, m.class AS cls
                LIMIT 3
            '''))
            for r in samples:
                d = r["d"][:120] + "..." if len(r["d"]) > 120 else r["d"]
                print(f"    {r['id']}(class={r['cls']}): {d}")

    driver.close()
    print("\n✅ 全部完成!")


if __name__ == "__main__":
    main()
