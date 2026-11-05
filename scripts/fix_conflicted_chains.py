"""修复 attach 链冲突: 重建冲突种 taxonomy_worms 全阶元链 + 嫁接骨干树边。

背景 (attach 报 59 种 63 条链冲突, 20 个唯一节点对), 三类根因:
  A. 元数据损坏: DEC/ART 链 subphylum=Chelicerata(aphia:1274) —
     build_morphospecies_index 的名称→aphia 缓存错配, 甲壳类应为
     Crustacea(aphia:1066)
  B. 分类漂移: FNM 链 order=Perciformes/Beryciformes — 当前 WoRMS 已把
     Sebastidae/Scorpaenidae 等移入 Scorpaeniformes
  C. 骨干树缺中间阶元边: 记录链法只含固定阶元, 缺 subphylum/superfamily
     等中间边, 如 Pleocyemata(suborder)→Caridea(infraorder)

WoRMS 的 AphiaClassificationByID 与 AphiaChildrenByID 均已 404 下线,
改用 AphiaRecordByAphiaID.parentNameUsageID 向上回溯。

对每个冲突种 (morphospecies_attachments.json 中 conflicts 非空):
  1. 锚点 = 链中最深的有 aphia_id 阶元 (校验锚点名与记录名一致)
  2. 自锚点逐级回查父记录至 kingdom, 得当前 WoRMS 完整链
  3. 替换 morphospecies_index.json 的 taxonomy_worms
     (同阶元同 aphia_id 时保留原 gbif_key)
  4. 链上节点/边嫁接进 taxon_backbone.json (缺失才加, 多父 DAG 只增不删)
链即骨干树真实路径, 重跑 attach 后这些种的冲突应清零。
"""
import json
import os
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB_DIR = os.path.join(BASE, "data/processed/taxon_backbone")
TB_JSON = os.path.join(TB_DIR, "taxon_backbone.json")
ATT_JSON = os.path.join(TB_DIR, "morphospecies_attachments.json")
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")

WORMS = "https://www.marinespecies.org/rest"
UA = {"User-Agent": "Ocean-DePTH-VLM/1.0 (research; chain fix)"}

CHAIN_ORDER = ["kingdom", "phylum", "subphylum", "infraphylum", "superclass",
               "class", "subclass", "infraclass", "superorder", "order",
               "suborder", "infraorder", "superfamily", "family", "subfamily",
               "tribe", "subtribe", "genus", "subgenus", "species",
               "subspecies"]

_rec_cache = {}


def http_json(url, retries=3):
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 204:
                    return None
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                return None
            if i == retries:
                return None
        except Exception:
            if i == retries:
                return None
        time.sleep(1.5 * (i + 1))
    return None


def get_record(aphia_id):
    if aphia_id in _rec_cache:
        return _rec_cache[aphia_id]
    time.sleep(0.3)
    r = http_json(f"{WORMS}/AphiaRecordByAphiaID/{aphia_id}")
    _rec_cache[aphia_id] = r
    return r


def walk_up(anchor_id, max_hops=20):
    """锚点向上回溯到 kingdom, 返回 [kingdom..anchor] 记录列表 (根→叶)。"""
    chain, seen = [], set()
    cur = anchor_id
    for _ in range(max_hops):
        if cur in seen:
            return None
        seen.add(cur)
        rec = get_record(cur)
        if not rec or not rec.get("AphiaID"):
            return None
        chain.append(rec)
        if (rec.get("rank") or "").lower() == "kingdom":
            return chain[::-1]
        p = rec.get("parentNameUsageID")
        if not p or p == rec["AphiaID"]:
            # 未置位的异名: 跳到 valid 记录再回溯
            v = rec.get("valid_AphiaID")
            if v and v != rec["AphiaID"]:
                cur = v
                continue
            return None
        cur = p
    return None


def main():
    tb = json.load(open(TB_JSON))
    att = json.load(open(ATT_JSON))
    msp_doc = json.load(open(MSP_JSON))
    nodes, edges = tb["nodes"], tb["edges"]
    edge_set = {tuple(e) for e in edges}

    conflicted = sorted(mid for mid, v in att["attachments"].items()
                        if v.get("conflicts"))
    print(f"冲突种 {len(conflicted)} 个", flush=True)
    by_id = {m["morphospecies_id"]: m for m in msp_doc["morphospecies"]}

    n_fix = n_skip = 0
    added_nodes = added_edges = 0
    for mid in conflicted:
        m = by_id.get(mid)
        if not m or not m.get("taxonomy_worms"):
            print(f"[{mid}] 无 taxonomy_worms, 跳过", flush=True)
            n_skip += 1
            continue
        old_tw = m["taxonomy_worms"]
        # 锚点: 最深的有 aphia_id 阶元
        anchor_rank = anchor = None
        for rk in reversed(CHAIN_ORDER):
            node = old_tw.get(rk)
            if node and node.get("aphia_id"):
                anchor_rank, anchor = rk, node
                break
        if not anchor:
            print(f"[{mid}] 链中无 aphia_id 锚点, 跳过", flush=True)
            n_skip += 1
            continue
        # 校验: 锚点 aphia 记录名应与链中名一致 (防缓存错配)
        rec0 = get_record(anchor["aphia_id"])
        if not rec0 or (rec0.get("scientificname") or "").lower() != \
                (anchor.get("name") or "").lower():
            print(f"[{mid}] 锚点 {anchor_rank} {anchor.get('name')} "
                  f"aphia:{anchor['aphia_id']} 名称不一致 "
                  f"({rec0.get('scientificname') if rec0 else '无记录'}), 跳过",
                  flush=True)
            n_skip += 1
            continue
        chain = walk_up(anchor["aphia_id"])
        if not chain:
            print(f"[{mid}] 回溯未达 kingdom, 跳过", flush=True)
            n_skip += 1
            continue

        # 1) 替换 taxonomy_worms (同 aphia_id 保留 gbif_key)
        new_tw = {}
        for rec in chain:
            rk = (rec.get("rank") or "").lower()
            if rk not in CHAIN_ORDER:
                continue
            gk = None
            old = old_tw.get(rk)
            if old and old.get("aphia_id") == rec["AphiaID"]:
                gk = old.get("gbif_key")
            new_tw[rk] = {"name": rec["scientificname"],
                          "aphia_id": rec["AphiaID"], "gbif_key": gk}
        m["taxonomy_worms"] = new_tw

        # 2) 嫁接骨干树: 链上连续节点对 (缺失才加)
        phylum_name = next((r["scientificname"] for r in chain
                             if (r.get("rank") or "").lower() == "phylum"),
                            m.get("phylum") or "")
        prev = None
        for rec in chain:
            rk = (rec.get("rank") or "").lower()
            if rk not in CHAIN_ORDER:
                continue
            aid = rec["AphiaID"]
            nid = f"aphia:{aid}"
            if nid not in nodes:
                nodes[nid] = {
                    "node_id": nid,
                    "name": rec["scientificname"],
                    "rank": rk,
                    "worms_aphia_id": aid,
                    "worms_status": rec.get("status"),
                    "parent_aphia_id": prev,
                    "is_marine": rec.get("isMarine", 1),
                    "is_terrestrial": rec.get("isTerrestrial", 0),
                    "is_freshwater": rec.get("isFreshwater", 0),
                    "gbif_keys": [],
                    "aliases": [],
                    "phylum": phylum_name,
                }
                added_nodes += 1
            if prev is not None:
                e = (f"aphia:{prev}", nid)
                if e not in edge_set:
                    edges.append(list(e))
                    edge_set.add(e)
                    added_edges += 1
            prev = aid
        n_fix += 1
        print(f"[{mid}] {m.get('latin')} 锚={anchor_rank}: "
              f"{len(new_tw)} 阶元 "
              f"({' > '.join(new_tw[r]['name'] for r in CHAIN_ORDER
                              if r in new_tw)})", flush=True)

    # 保存
    msp_doc.setdefault("params", {})["chain_fix"] = {
        "fixed_species": n_fix, "skipped": n_skip,
        "method": "AphiaRecordByAphiaID parentNameUsageID 回溯重建",
    }
    tmp = MSP_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(msp_doc, f, ensure_ascii=False)
    os.replace(tmp, MSP_JSON)

    tb["statistics"]["nodes"] = len(nodes)
    tb["statistics"]["edges"] = len(edges)
    tb["statistics"]["chain_fix"] = {
        "fixed_species": n_fix, "skipped": n_skip,
        "added_nodes": added_nodes, "added_edges": added_edges,
        "note": "冲突种链重建 + WoRMS 父链嫁接",
    }
    tmp = TB_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tb, f, ensure_ascii=False)
    os.replace(tmp, TB_JSON)

    print(f"\n完成: 修复 {n_fix} 种, 跳过 {n_skip} 种; "
          f"骨干树新增 {added_nodes} 节点 {added_edges} 边 "
          f"(现 {len(nodes)} 节点 {len(edges)} 边)", flush=True)


if __name__ == "__main__":
    main()
