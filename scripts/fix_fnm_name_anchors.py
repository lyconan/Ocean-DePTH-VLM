"""补充修复: 链中全阶元无 aphia_id 的 FNM 形态种。

fix_conflicted_chains.py 跳过的 8 个 FNM 种 (链 aphia_id 全 None),
其 order=Perciformes/Beryciformes 为旧分类, 经名称匹配挂到骨干旧节点
而 family 已被 WoRMS 移入 Scorpaeniformes 等 → attach 链冲突。

方法: 取链中最深阶元 (genus 优先) 名称, AphiaRecordsByName 查询
(精确名 + accepted + 同 rank) 得锚点 AphiaID, 再
AphiaRecordByAphiaID.parentNameUsageID 回溯至 kingdom 重建全链,
并嫁接骨干树 (与 fix_conflicted_chains.py 相同的嫁接逻辑)。
"""
import json
import os
import time
import urllib.request
import urllib.parse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB_DIR = os.path.join(BASE, "data/processed/taxon_backbone")
TB_JSON = os.path.join(TB_DIR, "taxon_backbone.json")
MSP_JSON = os.path.join(BASE, "data/processed/atlas_lookup/morphospecies_index.json")

WORMS = "https://www.marinespecies.org/rest"
UA = {"User-Agent": "Ocean-DePTH-VLM/1.0 (research; fnm name anchor fix)"}

CHAIN_ORDER = ["kingdom", "phylum", "subphylum", "infraphylum", "superclass",
               "class", "subclass", "infraclass", "superorder", "order",
               "suborder", "infraorder", "superfamily", "family", "subfamily",
               "tribe", "subtribe", "genus", "subgenus", "species",
               "subspecies"]

TARGETS = ["FNM_00215", "FNM_00320", "FNM_01235", "FNM_01246",
           "FNM_01253", "FNM_01260", "FNM_01262", "FNM_01273"]

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


def name_to_aphia(rank, name):
    """AphiaRecordsByName: 精确名 + accepted + 同 rank -> AphiaID"""
    time.sleep(0.3)
    recs = http_json(f"{WORMS}/AphiaRecordsByName/"
                     f"{urllib.parse.quote(name)}?marine_only=true")
    if not recs:
        return None
    cands = [r for r in recs
             if (r.get("scientificname") or "").lower() == name.lower()
             and r.get("status") == "accepted"
             and (r.get("rank") or "").lower() == rank]
    if len(cands) == 1:
        return cands[0]["AphiaID"]
    return None


def walk_up(anchor_id, max_hops=20):
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
            v = rec.get("valid_AphiaID")
            if v and v != rec["AphiaID"]:
                cur = v
                continue
            return None
        cur = p
    return None


def main():
    tb = json.load(open(TB_JSON))
    msp_doc = json.load(open(MSP_JSON))
    nodes, edges = tb["nodes"], tb["edges"]
    edge_set = {tuple(e) for e in edges}

    by_id = {m["morphospecies_id"]: m for m in msp_doc["morphospecies"]}
    n_fix = n_skip = added_nodes = added_edges = 0
    for mid in TARGETS:
        m = by_id.get(mid)
        if not m or not m.get("taxonomy_worms"):
            print(f"[{mid}] 未找到/无 taxonomy_worms, 跳过", flush=True)
            n_skip += 1
            continue
        old_tw = m["taxonomy_worms"]
        # 最深阶元 (genus 优先) 作为名称锚点
        anchor_rank = anchor_name = None
        for rk in reversed(CHAIN_ORDER):
            node = old_tw.get(rk)
            if node and node.get("name"):
                anchor_rank, anchor_name = rk, node["name"]
                break
        if not anchor_name:
            print(f"[{mid}] 链中无名称, 跳过", flush=True)
            n_skip += 1
            continue
        aphia_id = name_to_aphia(anchor_rank, anchor_name)
        if not aphia_id:
            print(f"[{mid}] 名称 {anchor_rank} {anchor_name} 未唯一解析, 跳过",
                  flush=True)
            n_skip += 1
            continue
        chain = walk_up(aphia_id)
        if not chain:
            print(f"[{mid}] aphia:{aphia_id} 回溯未达 kingdom, 跳过", flush=True)
            n_skip += 1
            continue

        # 1) 重建 taxonomy_worms (原链 aphia 全 None, 同名同阶元保留 gbif_key)
        new_tw = {}
        for rec in chain:
            rk = (rec.get("rank") or "").lower()
            if rk not in CHAIN_ORDER:
                continue
            gk = None
            old = old_tw.get(rk)
            if old and (old.get("aphia_id") == rec["AphiaID"]
                        or (old.get("name") or "").lower()
                        == (rec.get("scientificname") or "").lower()):
                gk = old.get("gbif_key")
            new_tw[rk] = {"name": rec["scientificname"],
                          "aphia_id": rec["AphiaID"], "gbif_key": gk}
        m["taxonomy_worms"] = new_tw

        # 2) 嫁接骨干树
        phylum_name = next((r["scientificname"] for r in chain
                            if (r.get("rank") or "").lower() == "phylum"), "")
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
        print(f"[{mid}] {m.get('latin')} 锚={anchor_rank}:{anchor_name}"
              f"→aphia:{aphia_id} {len(new_tw)} 阶元 "
              f"({' > '.join(new_tw[r]['name'] for r in CHAIN_ORDER
                              if r in new_tw)})", flush=True)

    msp_doc.setdefault("params", {})["fnm_name_anchor_fix"] = {
        "fixed": n_fix, "skipped": n_skip}
    tmp = MSP_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(msp_doc, f, ensure_ascii=False)
    os.replace(tmp, MSP_JSON)

    tb["statistics"]["nodes"] = len(nodes)
    tb["statistics"]["edges"] = len(edges)
    tb["statistics"]["fnm_name_anchor_fix"] = {
        "fixed": n_fix, "skipped": n_skip,
        "added_nodes": added_nodes, "added_edges": added_edges}
    tmp = TB_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tb, f, ensure_ascii=False)
    os.replace(tmp, TB_JSON)

    print(f"\n完成: 修复 {n_fix} 种, 跳过 {n_skip} 种; "
          f"骨干树新增 {added_nodes} 节点 {added_edges} 边 "
          f"(现 {len(nodes)} 节点 {len(edges)} 边)", flush=True)


if __name__ == "__main__":
    main()
