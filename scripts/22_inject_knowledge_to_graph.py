#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_inject_knowledge_to_graph.py — 三路五节知识文本注入 Neo4j
来源(互不重叠, 合计2641):
  batch_100/batch_100_results.jsonl  -> src=web_batch100 (网页版100)
  full_2641/batch_results.jsonl      -> src=full2641     (批量259)
  maas_full/results.jsonl            -> src=模型名        (双模型2282)
写入: Morphospecies {knowledge_text, kb_source, kb_updated, description}
安全: 先做1条中文往返自检(防容器Java ASCII字符集损坏), 通过后再全量执行
"""
import json
import subprocess
from datetime import date

KB = "/mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM/data/processed/species_kb"
TODAY = date.today().isoformat()
DOCKER = ["docker", "exec", "-i", "-e", "LANG=C.UTF-8", "-e", "JAVA_TOOL_OPTIONS=-Dfile.encoding=UTF-8",
          "depth_vlm_neo4j", "cypher-shell", "-u", "neo4j", "-p", "password", "--format", "plain"]


def compose(rec: dict) -> str:
    """优先用现成 text, 否则由 sections 拼装"""
    if rec.get("text"):
        return rec["text"]
    parts = []
    for s in rec.get("sections") or []:
        if isinstance(s, dict) and s.get("text"):
            parts.append(f"## {s.get('title','').strip()}\n\n{s['text'].strip()}")
    return "\n\n".join(parts)


def esc(s: str) -> str:
    """Cypher 字符串转义"""
    return (s.replace("\\", "\\\\").replace('"', '\\"')
             .replace("\r", "").replace("\n", "\\n").replace("\t", " "))


def load():
    mids = {}
    n = 0
    for line in open(f"{KB}/batch_100/batch_100_results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("ok") and r.get("mid"):
            mids[r["mid"]] = (compose(r), "web_batch100")
            n += 1
    print(f"batch_100: {n}")
    n = 0
    for line in open(f"{KB}/full_2641/batch_results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("ok") and r.get("mid"):
            mids[r["mid"]] = (compose(r), "full2641")
            n += 1
    print(f"full_2641: {n}")
    n = 0
    for line in open(f"{KB}/maas_full/results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("ok") and r.get("mid"):
            mids[r["mid"]] = (compose(r), r.get("source") or "maas")
            n += 1
    print(f"maas_full: {n}")
    return mids


def run_cypher(cypher: str) -> str:
    p = subprocess.run(DOCKER, input=cypher.encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    return p.stdout.decode("utf-8", "replace")


def roundtrip_test() -> bool:
    """中文往返自检: 写入->读回比对"""
    probe = "深海生物测试：六放海绵纲未定种2，海葵科（2026）。"
    out = run_cypher(
        'MATCH (m:Morphospecies {morphospecies_id:"DES_002"}) '
        f'SET m._enc_probe = "{esc(probe)}" RETURN m._enc_probe;'
    )
    ok = probe in out
    run_cypher('MATCH (m:Morphospecies {morphospecies_id:"DES_002"}) REMOVE m._enc_probe;')
    return ok


def main():
    mids = load()
    print(f"合计待注入: {len(mids)}")
    if not roundtrip_test():
        print("编码往返自检失败, 终止写入!")
        return
    print("编码往返自检通过")

    items = list(mids.items())
    done = 0
    for i in range(0, len(items), 200):
        batch = items[i:i + 200]
        rows = ", ".join(
            '{mid:"%s", txt:"%s", src:"%s"}' % (mid, esc(txt), esc(src))
            for mid, (txt, src) in batch)
        cy = ("UNWIND [%s] AS row MATCH (m:Morphospecies {morphospecies_id: row.mid}) "
              "SET m.knowledge_text = row.txt, m.description = row.txt, "
              "m.kb_source = row.src, m.kb_updated = '%s';" % (rows, TODAY))
        out = run_cypher(cy)
        if "error" in out.lower() or "exception" in out.lower():
            print(f"批次 {i} 异常: {out[:300]}")
            return
        done += len(batch)
        print(f"已写入 {done}/{len(items)}")

    print("完成, 校验库内计数...")
    out = run_cypher("MATCH (m:Morphospecies) WHERE m.knowledge_text IS NOT NULL RETURN count(m);")
    print(out.strip().splitlines()[-1] if out.strip() else "?")


if __name__ == "__main__":
    main()
