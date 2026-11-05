"""将 WoRMS MD 描述结构化为简洁摘要。

对每个 MD 文件:
1. 提取前 2000 字符作为摘要
2. 搜索关键词(Description/Diagnosis/Distribution/形态/分布/特征)
3. 提取关键词周围的上下文
4. 生成结构化 JSON 描述

输入:
    data/processed/species_knowledge_base/descriptions/{Phylum}/{Species}.md

输出:
    data/processed/species_knowledge_base/structured/{Phylum}/{Species}.json
"""
import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM")
DESC_DIR = PROJECT_ROOT / "data/processed/species_knowledge_base/descriptions"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/species_knowledge_base/structured"

# 关键词(英文+中文)
KEYWORDS = {
    "description": ["Description", "DESCRIPTION", "描述", "形态特征", "形态"],
    "diagnosis": ["Diagnosis", "DIAGNOSIS", "鉴别", "鉴别特征", "诊断"],
    "distribution": ["Distribution", "DISTRIBUTION", "地理分布", "分布"],
    "habitat": ["Habitat", "HABITAT", "生境", "栖息地", "生物学资料"],
    "remarks": ["Remarks", "REMARKS", "讨论", "备注"],
    "etymology": ["Etymology", "ETYMOLOGY", "词源"],
    "synonymy": ["Synonymy", "Synonyms", "同物异名", "异名"],
}


def extract_structured_description(md_path):
    """从 MD 文件提取结构化描述。

    Returns:
        {
            "summary": str,  # 前2000字符摘要
            "sections": {keyword_type: [snippets]},
            "word_count": int,
            "has_images": bool,
        }
    """
    try:
        with open(md_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return None

    # 检查是否有图片引用
    has_images = "![](images/" in content or "![](" in content

    # 前 2000 字符摘要(去除空行)
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    summary = ' '.join(lines[:50])[:2000]

    # 按关键词提取章节
    sections = defaultdict(list)
    for section_type, keywords in KEYWORDS.items():
        for kw in keywords:
            # 找到关键词位置,提取前后 500 字符
            for match in re.finditer(rf'(?i)\b{re.escape(kw)}\b', content):
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 500)
                snippet = content[start:end].replace('\n', ' ').strip()
                if len(snippet) > 50:  # 过滤太短的
                    sections[section_type].append(snippet)
                    break  # 每个关键词只取第一个匹配

    return {
        "summary": summary,
        "sections": dict(sections),
        "word_count": len(content.split()),
        "char_count": len(content),
        "has_images": has_images,
    }


def main():
    print("=" * 70)
    print("MD 描述结构化")
    print("=" * 70)

    total = 0
    success = 0
    failed = 0
    by_phylum = defaultdict(int)

    # 遍历所有描述文件
    for phylum_dir in DESC_DIR.iterdir():
        if not phylum_dir.is_dir():
            continue
        phylum = phylum_dir.name
        output_phylum_dir = OUTPUT_DIR / phylum
        output_phylum_dir.mkdir(parents=True, exist_ok=True)

        for md_file in phylum_dir.iterdir():
            if not (md_file.is_symlink() or md_file.is_file()):
                continue
            total += 1

            # 解析目标文件(如果是符号链接)
            target = md_file.resolve() if md_file.is_symlink() else md_file
            result = extract_structured_description(target)
            if result is None:
                failed += 1
                continue

            # 保存为 JSON
            output_file = output_phylum_dir / f"{md_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            success += 1
            by_phylum[phylum] += 1

    print(f"\n✅ 结构化完成!")
    print(f"  总文件数: {total:,}")
    print(f"  成功: {success:,}")
    print(f"  失败: {failed:,}")

    print(f"\n按门级分布:")
    for phylum, count in sorted(by_phylum.items(), key=lambda x: -x[1]):
        print(f"  {phylum:25s}: {count:5,}")

    # 统计压缩效果
    if success > 0:
        original_size = sum(f.stat().st_size for f in DESC_DIR.rglob('*') if f.is_file() or f.is_symlink())
        new_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file())
        print(f"\n压缩效果:")
        print(f"  原始大小: {original_size/1024/1024:.1f} MB (符号链接)")
        print(f"  结构化大小: {new_size/1024/1024:.1f} MB")
        # 实际原始大小需要解析符号链接
        actual_orig = 0
        for f in DESC_DIR.rglob('*'):
            if f.is_symlink():
                try:
                    actual_orig += f.resolve().stat().st_size
                except Exception:
                    pass
        print(f"  原始实际大小: {actual_orig/1024/1024:.1f} MB")
        print(f"  压缩率: {new_size/actual_orig*100:.1f}%")


if __name__ == "__main__":
    main()
