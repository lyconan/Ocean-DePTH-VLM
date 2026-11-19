#!/usr/bin/env python3
"""console_api.py — 深海底栖生物知识系统控制台 后端 (纯标准库, 零依赖)

可视化控制台数据服务, 默认端口 8089:
  GET /api/stats                  总览统计 (图谱/知识库/检索库)
  GET /api/species                物种列表 (q/phylum/cls/page/size)
  GET /api/species/{mid}          物种详情 (全字段 + knowledge_text + 证据图集)
  GET /api/entries                检索库证据图列表 (source/cls/q/page/size)
  GET /api/pipeline/summary       盲测管线报告 (各阶段指标)
  GET /api/pipeline/objects       管线逐对象结果 (page/hit)
  GET /api/pipeline/object/{uuid} 单对象候选回放 + 查询图
  GET /api/vlm/status             VLM 终判接口状态 (预留)
  GET /api/vlm/judge/{uuid}       构造 11 图选择题并调用 VLM (未配置时返回就绪态)
  /media/{path}                   项目文件静态服务 (证据图/框切图)
  /                               控制台页面 webapp/index.html

启动: python3 webapp/console_api.py [--port 8089] [--host 0.0.0.0]
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import urllib.parse
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 环境变量可覆盖, 便于第三方在自己服务器上接入自有 Neo4j 与数据
NEO4J_HTTP = os.environ.get("DEPTH_NEO4J_HTTP", "http://localhost:7474/db/neo4j/tx/commit")
NEO4J_AUTH = base64.b64encode(os.environ.get("DEPTH_NEO4J_AUTH", "neo4j:password").encode()).decode()
REPORT = Path(os.environ.get("DEPTH_BLIND_REPORT",
                             str(ROOT / "data/experiments/blindtest_e2e/e2e_report_postjedi.json")))
PER_OBJ = Path(os.environ.get("DEPTH_BLIND_OBJECTS",
                              str(ROOT / "data/experiments/blindtest_e2e/per_object_postjedi.jsonl")))
MANIFEST = Path(os.environ.get("DEPTH_BLIND_MANIFEST", "/tmp/exp/blindtest/manifest.json"))
DET_CACHE = Path(os.environ.get("DEPTH_DET_CACHE", "/tmp/exp/blindtest/det_cache.pkl"))
CFG = ROOT / "config/graphrag_inference.yaml"

_objects_cache = None


def cypher(q, params=None):
    body = {"statements": [{"statement": q, "parameters": params or {}}]}
    req = urllib.request.Request(
        NEO4J_HTTP, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Basic " + NEO4J_AUTH})
    data = json.load(urllib.request.urlopen(req, timeout=30))
    if data.get("errors"):
        raise RuntimeError(str(data["errors"]))
    out = []
    for res in data["results"]:
        out = [d["row"] for d in res.get("data", [])]
    return out


def load_objects():
    global _objects_cache
    if _objects_cache is None:
        _objects_cache = []
        if not PER_OBJ.exists():
            return _objects_cache  # 盲测数据缺失时优雅降级
        for line in PER_OBJ.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["detected"] = str(r.get("detected")) == "True"
                _objects_cache.append(r)
    return _objects_cache


SRC_LABELS = [
    ("fathomnet", "FathomNet 框切图"), ("yolo_boxes", "标注队列帧图"),
    ("gbif_supplement", "GBIF 补充"), ("atlas", "图鉴整版插图"),
    ("benthicnet", "BenthicNet 补充"), ("manual", "人工兜底"),
]


def src_of(path):
    p = str(path).lower()
    for key, _ in SRC_LABELS:
        if key in p:
            return key
    return "other"


STAGE_META = [
    {"step": 1, "key": "detect", "name": "YOLO 检测",
     "desc": "34 纲 YOLO 双路检测 (960+TTA ∪ 1920), 检出框与 GT IoU≥0.5 匹配"},
    {"step": 2, "key": "coarse", "name": "粗分类",
     "desc": "每个检出框输出 34 纲之一 (Hexactinellida / Malacostraca / ...)"},
    {"step": 3, "key": "embed", "name": "BioCLIP 嵌入",
     "desc": "BioCLIP 2.5 + LoRA v2 嵌入 (水平翻转 TTA), 30,629 库向量余弦检索"},
    {"step": 4, "key": "retrieve", "name": "多策略检索融合",
     "desc": "base / soft / hard / cfusion / rrf 五策略 + 质心融合 (cfusion 定版)"},
    {"step": 5, "key": "vlm", "name": "VLM 终判 (接口预留)",
     "desc": "11 图选择题送 VLM 复核 top-10 候选 — 微调完成后在 /api/vlm/judge 生效"},
]


# ---------------------------------------------------------------- handlers
def h_stats():
    nodes = cypher(
        "MATCH (m:Morphospecies) RETURN count(m) AS species, "
        "sum(CASE WHEN m.knowledge_text IS NOT NULL AND size(m.knowledge_text)>0 "
        "THEN 1 ELSE 0 END) AS with_desc, "
        "sum(CASE WHEN m.kb_updated IS NOT NULL AND toString(m.kb_updated) <> 'false' "
        "THEN 1 ELSE 0 END) AS kb_updated")[0]
    by_phylum = cypher(
        "MATCH (m:Morphospecies) RETURN m.phylum AS p, count(m) AS n "
        "ORDER BY n DESC LIMIT 12")
    entry_total = cypher("MATCH (i:EntryImage) RETURN count(i) AS n")[0][0]
    src_rows = cypher("MATCH (i:EntryImage) RETURN i.image_id LIMIT 40000")
    cnt = Counter(src_of(r[0]) for r in src_rows)
    by_src = [{"key": k, "name": dict(SRC_LABELS).get(k, k), "n": cnt.get(k, 0)}
              for k, _ in SRC_LABELS]
    by_src.append({"key": "other", "name": "其他", "n": cnt.get("other", 0)})
    return {"species": nodes[0], "with_desc": nodes[1], "kb_updated": nodes[2],
            "entry_images": entry_total,
            "by_phylum": [{"name": r[0] or "?", "n": r[1]} for r in by_phylum],
            "by_source": by_src}


def _kb_flag(v):
    if isinstance(v, str):
        return v.lower() not in ("", "false", "0", "no")
    return bool(v)


def h_species(qs):
    cond, params = [], {}
    q = qs.get("q", "")
    if q:
        cond.append("(toLower(m.display_name) CONTAINS toLower($q) OR "
                    "toLower(m.morphospecies_id) CONTAINS toLower($q) OR "
                    "toLower(coalesce(m.name_cn,'')) CONTAINS toLower($q))")
        params["q"] = q.lower()
    if qs.get("phylum"):
        cond.append("m.phylum = $ph"); params["ph"] = qs["phylum"]
    if qs.get("cls"):
        cond.append("m.class_name = $cls"); params["cls"] = qs["cls"]
    where = ("WHERE " + " AND ".join(cond)) if cond else ""
    page, size = int(qs.get("page", 1)), min(int(qs.get("size", 30)), 100)
    params.update({"skip": (page - 1) * size, "lim": size})
    rows = cypher(
        f"MATCH (m:Morphospecies) {where} "
        "RETURN m.morphospecies_id AS mid, m.display_name AS name, m.latin AS latin, "
        "m.name_cn AS cn, m.phylum AS phylum, m.class_name AS cls, "
        "m.image_count AS imgs, m.kb_updated AS kb_updated, m.aphia_id AS aphia "
        "ORDER BY m.image_count DESC SKIP $skip LIMIT $lim", params)
    total = cypher(f"MATCH (m:Morphospecies) {where} RETURN count(m)", params)[0][0]
    return {"total": total, "page": page, "size": size, "items": [
        {"mid": r[0], "name": r[1] or r[2], "latin": r[2], "cn": r[3],
         "phylum": r[4], "cls": r[5], "imgs": r[6], "kb_updated": _kb_flag(r[7]),
         "aphia": r[8]} for r in rows]}


def h_species_detail(mid):
    rows = cypher(
        "MATCH (m:Morphospecies {morphospecies_id:$mid}) "
        "OPTIONAL MATCH (m)-[:HAS_IMAGE]->(i:EntryImage) "
        "WITH m, i ORDER BY i.image_id "
        "WITH m, collect(i.image_id)[..24] AS imgs, count(i) AS n_img RETURN "
        "m.morphospecies_id AS mid, m.display_name AS name, m.latin AS latin, "
        "m.name_cn AS cn, m.phylum AS phylum, m.class_name AS cls, m.order_name AS ord, "
        "m.family_name AS fam, m.genus AS genus, m.rank AS rank, m.rank_cn AS rank_cn, "
        "m.aphia_id AS aphia, m.worms_url AS worms_url, m.kb_source AS kb_source, "
        "m.kb_updated AS kb_updated, m.knowledge_text AS knowledge_text, "
        "m.description AS description, imgs, n_img", {"mid": mid})
    if not rows:
        return None
    keys = ["mid","name","latin","cn","phylum","cls","ord","fam","genus","rank",
            "rank_cn","aphia","worms_url","kb_source","kb_updated","knowledge_text",
            "description","imgs","n_img"]
    d = dict(zip(keys, rows[0]))
    d["kb_updated"] = _kb_flag(d.get("kb_updated"))
    return d


def h_entries(qs):
    cond, params = [], {}
    if qs.get("mid"):
        cond.append("m.morphospecies_id = $mid")
        params["mid"] = qs["mid"]
    if qs.get("source"):
        cond.append("toLower(i.image_id) CONTAINS $src")
        params["src"] = qs["source"].lower()
    if qs.get("q"):
        cond.append("toLower(i.image_id) CONTAINS toLower($q)")
        params["q"] = qs["q"]
    where = ("WHERE " + " AND ".join(cond)) if cond else ""
    page, size = int(qs.get("page", 1)), min(int(qs.get("size", 48)), 120)
    params.update({"skip": (page - 1) * size, "lim": size})
    rows = cypher(
        f"MATCH (m:Morphospecies)-[:HAS_IMAGE]->(i:EntryImage) {where} "
        "RETURN i.image_id AS p, m.morphospecies_id AS mid, m.display_name AS name "
        "ORDER BY i.image_id SKIP $skip LIMIT $lim", params)
    total = cypher(
        f"MATCH (m:Morphospecies)-[:HAS_IMAGE]->(i:EntryImage) {where} RETURN count(i)",
        params)[0][0]
    return {"total": total, "page": page, "size": size, "items": [
        {"path": r[0], "mid": r[1], "name": r[2], "src": src_of(r[0])} for r in rows]}


def h_pipeline_summary():
    if not REPORT.exists():
        return {"stages": STAGE_META, "metrics": None,
                "note": "盲测报告数据缺失 — 管线回放不可用; 知识库/检索库功能不受影响"}
    rep = json.load(open(REPORT, encoding="utf-8"))
    return {"stages": STAGE_META, "metrics": rep}


def h_pipeline_objects(qs):
    objs = load_objects()
    hit = qs.get("hit", "all")
    if hit == "hit":
        objs = [o for o in objs if o.get("retrieval_top10_cfusion") and
                o["mid"] == o["retrieval_top10_cfusion"][0]]
    elif hit == "miss":
        objs = [o for o in objs if o.get("detected") and
                (not o.get("retrieval_top10_cfusion") or
                 o["mid"] != o["retrieval_top10_cfusion"][0])]
    page, size = int(qs.get("page", 1)), min(int(qs.get("size", 30)), 100)
    occ = {}
    for o in objs:  # 全量计数, 防止同场景对象跨页时序号错位
        occ[o["uuid"]] = occ.get(o["uuid"], -1) + 1
        o["oidx"] = occ[o["uuid"]]
    return {"total": len(objs), "page": page, "size": size,
            "items": objs[(page - 1) * size: page * size]}


_det_cache = None

TILE_DIRS = [ROOT / "data/processed/apsma_aligned_yolo/images/val",
             ROOT / "data/processed/apsma_aligned_yolo/images/train",
             ROOT / "data/processed/apsma_aligned_yolo/images/test"]
CROP_CACHE = ROOT / "webapp/cache/crops"


def _dets_for(uuid):
    """读取盲测检测缓存: {tile_uuid: [{box,conf,cls,name}, ...]}"""
    global _det_cache
    if _det_cache is None:
        _det_cache = {}
        try:
            import pickle
            raw = pickle.load(open(DET_CACHE, "rb")).get("dets", {})
            _det_cache = {str(k): v for k, v in raw.items()}
        except Exception:
            pass
    return _det_cache.get(uuid, [])


def _tile_crop(uuid, det_box):
    """按检测框从原图瓦片现裁查询图, 落盘缓存后返回相对路径"""
    key = f"{uuid}_{det_box[0]}_{det_box[1]}.jpg"
    out = CROP_CACHE / key
    if out.exists():
        return str(out.relative_to(ROOT))
    tile = None
    for d in TILE_DIRS:
        p = d / f"{uuid}.png"
        if p.exists():
            tile = p
            break
    if tile is None:
        return ""
    try:
        from PIL import Image
        img = Image.open(tile).convert("RGB")
        x1, y1, x2, y2 = [int(v) for v in det_box[:4]]
        x2 = min(x2, img.width); y2 = min(y2, img.height)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return ""
        img.crop((x1, y1, x2, y2)).save(out, quality=90)
        return str(out.relative_to(ROOT))
    except Exception:
        return ""


def _query_img_for(uuid, mid, det_box=None):
    """查询图三级回退: manifest 框切图 → 检测框现裁"""
    scenes = {}
    if MANIFEST.exists():
        scenes = json.load(open(MANIFEST, encoding="utf-8")).get("scenes", {}).get(uuid, {})
        for b in scenes.get("boxes", []):
            if b.get("mid") == mid:
                return scenes, b.get("image_id", "")
    if det_box and len(det_box) >= 2:
        for d in _dets_for(uuid):
            if d.get("box", [])[:2] == list(det_box[:2]):
                return scenes, _tile_crop(uuid, d["box"])
    return scenes, ""


def h_pipeline_object(uuid):
    objs = [o for o in load_objects() if o["uuid"] == uuid]
    if not objs:
        return None
    o = objs[0]
    scenes, _ = _query_img_for(uuid, o["mid"], o.get("det_box"))
    for ob in objs:
        _, ob["query_img"] = _query_img_for(uuid, ob["mid"], ob.get("det_box"))
    mids = sorted({m for m in (o.get("retrieval_top10_cfusion") or [])})
    thumbs = {}
    for mid in mids[:12]:
        rows = cypher(
            "MATCH (m:Morphospecies {morphospecies_id:$mid})-[:HAS_IMAGE]->(i:EntryImage) "
            "RETURN i.image_id LIMIT 1", {"mid": mid})
        if rows:
            thumbs[mid] = rows[0][0]
    return {"objects": objs, "scene_boxes": scenes.get("boxes", []),
            "thumbs": thumbs, "query_img": objs[0].get("query_img", "")}


YOLO_NAMES = {0: "Porifera", 1: "Echinodermata", 2: "Cnidaria", 3: "Annelida",
              4: "Arthropoda", 5: "Xenophyophorea", 6: "Mollusca",
              7: "Brachiopoda", 8: "Foraminifera", 9: "Unknown"}


def h_pipeline_scene(uuid):
    """场景原图 + 检测框 + GT 标注框 (用于原图放大查看)"""
    tile = ""
    for d in TILE_DIRS:
        p = d / f"{uuid}.png"
        if p.exists():
            tile = str(p.relative_to(ROOT))
            break
    if not tile:
        return None
    gts, lbl = [], None
    for sp in ("val", "train", "test"):
        p = ROOT / f"data/processed/apsma_aligned_yolo/labels/{sp}/{uuid}.txt"
        if p.exists():
            lbl = p
            break
    if lbl:
        for line in lbl.read_text(encoding="utf-8").splitlines():
            ps = line.split()
            if len(ps) >= 5:
                cx, cy, w, h = map(float, ps[1:5])
                gts.append({"cls": YOLO_NAMES.get(int(ps[0]), ps[0]),
                            "box": [cx - w / 2, cy - h / 2, w, h]})
    dets = _dets_for(uuid)
    objs = [dict(o) for o in load_objects() if o["uuid"] == uuid]
    for ob in objs:
        ob["det_box_full"] = next(
            (d["box"] for d in dets
             if d.get("box", [])[:2] == list(ob.get("det_box", [])[:2])), None)
    return {"tile": tile, "dets": dets, "gts": gts, "objects": objs}


def _vlm_cfg():
    txt = CFG.read_text(encoding="utf-8")
    url = re.search(r'vlm_api_url:\s*"?([^"\n#]+)', txt)
    model = re.search(r'vlm_model:\s*"?([^"\n#]+)', txt)
    return (url.group(1).strip() if url else ""), (model.group(1).strip() if model else "")


def h_vlm_status():
    url, model = _vlm_cfg()
    configured = bool(url and "API_URL" not in url and not url.startswith("${"))
    alive = False
    if configured:
        try:
            req = urllib.request.Request(url.rstrip("/") + "/models")
            urllib.request.urlopen(req, timeout=2)
            alive = True
        except Exception:
            alive = False
    if configured and not alive:
        note = ("已配置但端点不可达 — 旧 VLM 服务未在线。微调后的 VLM 以 vLLM 部署后, "
                "更新 vlm_api_url/vlm_model 即启用终判")
        state = "配置指向离线端点（预留待接入）"
    elif configured:
        note = "终判接口就绪, 可在下方输入盲测对象 UUID 在线试跑"
        state = "在线"
    else:
        note = ("VLM 未配置 — 接口已预留, 微调完成后在 config/graphrag_inference.yaml "
                "填入 vlm_api_url/vlm_model 即启用")
        state = "未配置（预留）"
    return {"configured": configured, "alive": alive, "state": state,
            "api_url": url, "model": model, "note": note}


def h_vlm_judge(uuid):
    objs = [o for o in load_objects() if o["uuid"] == uuid]
    if not objs:
        return None
    o = objs[0]
    cand = o.get("retrieval_top10_cfusion") or o.get("retrieval_top10") or []
    _, q_img = _query_img_for(uuid, o["mid"], o.get("det_box"))
    refs = []
    for mid in cand[:10]:
        rows = cypher(
            "MATCH (m:Morphospecies {morphospecies_id:$mid})-[:HAS_IMAGE]->(i:EntryImage) "
            "RETURN i.image_id LIMIT 1", {"mid": mid})
        refs.append({"mid": mid, "ref": rows[0][0] if rows else ""})
    url, model = _vlm_cfg()
    payload = {"protocol": "11-image MCQ (1 query crop + 10 candidate refs, output letter)",
               "query_image": q_img, "gt_mid": o["mid"], "candidates": refs,
               "answer": None, "vlm_api_url": url, "vlm_model": model}
    vlm_ready = bool(url) and "API_URL" not in url and not url.startswith("${")
    q_ok = bool(q_img) and (ROOT / q_img).exists()
    if vlm_ready and q_ok:
        try:
            content = [{"type": "image_url",
                        "image_url": {"url": "file://" + str(ROOT / q_img)}}]
            for r in refs:
                if r["ref"]:
                    content.append({"type": "image_url",
                                    "image_url": {"url": "file://" + str(ROOT / r["ref"])}})
            opt_lines = "\n".join(
                f"{chr(65+i)}. {r['mid']}（参考图{i+2}）" for i, r in enumerate(refs))
            content.append({"type": "text", "text":
                "第1张图为待鉴定的深海底栖生物裁剪图。随后10张参考图依次对应候选物种 A-J。"
                "请判断第1张图与哪个候选是同一物种，只输出一个选项字母。\n" + opt_lines})
            body = json.dumps({"model": model, "messages": [{"role": "user",
                               "content": content}], "temperature": 0,
                               "max_tokens": 8}).encode()
            req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                         data=body,
                                         headers={"Content-Type": "application/json"})
            resp = json.load(urllib.request.urlopen(req, timeout=60))
            letter = (resp["choices"][0]["message"]["content"] or "").strip()[:1]
            payload["answer"] = letter
            if letter and letter.isalpha():
                payload["answer_mid"] = refs[ord(letter.upper()) - 65]["mid"]
        except Exception as e:
            payload["error"] = f"VLM 调用失败: {type(e).__name__}: {str(e)[:120]}"
    else:
        payload["status"] = ("接口预留 — vlm_api_url 未配置或查询图缺失, "
                             "候选与参考图已构造完毕")
    return payload


# ---------------------------------------------------------------- HTTP server
ROUTES = [
    (re.compile(r"^/api/stats$"), lambda m, qs: h_stats()),
    (re.compile(r"^/api/species$"), lambda m, qs: h_species(qs)),
    (re.compile(r"^/api/species/([^/]+)$"), lambda m, qs: h_species_detail(urllib.parse.unquote(m.group(1)))),
    (re.compile(r"^/api/entries$"), lambda m, qs: h_entries(qs)),
    (re.compile(r"^/api/pipeline/summary$"), lambda m, qs: h_pipeline_summary()),
    (re.compile(r"^/api/pipeline/objects$"), lambda m, qs: h_pipeline_objects(qs)),
    (re.compile(r"^/api/pipeline/object/([^/]+)$"), lambda m, qs: h_pipeline_object(urllib.parse.unquote(m.group(1)))),
    (re.compile(r"^/api/pipeline/scene/([^/]+)$"), lambda m, qs: h_pipeline_scene(urllib.parse.unquote(m.group(1)))),
    (re.compile(r"^/api/vlm/status$"), lambda m, qs: h_vlm_status()),
    (re.compile(r"^/api/vlm/judge/([^/]+)$"), lambda m, qs: h_vlm_judge(urllib.parse.unquote(m.group(1)))),
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默访问日志

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path, qs = parsed.path, dict(urllib.parse.parse_qsl(parsed.query))
        try:
            for pat, fn in ROUTES:
                m = pat.match(path)
                if m:
                    out = fn(m, qs)
                    if out is None:
                        self._json({"error": "not found"}, 404)
                    else:
                        self._json(out)
                    return
            if path == "/" or path == "/index.html":
                f = ROOT / "webapp/index.html"
                data = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/media/"):
                rel = urllib.parse.unquote(path[len("/media/"):])
                p = ROOT / rel
                # 防目录穿越: 逐段检查 (允许符号链接指向 NAS 原始数据)
                if (".." in p.parts or not str(p).startswith(str(ROOT))
                        or not p.resolve().is_file()):
                    self._json({"error": "forbidden"}, 403)
                    return
                f = p.resolve()
                ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(f.stat().st_size))
                self.end_headers()
                with open(f, "rb") as fp:
                    while chunk := fp.read(1 << 20):
                        self.wfile.write(chunk)
                return
            self._json({"error": "unknown route"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._json({"error": f"{type(e).__name__}: {str(e)[:200]}"}, 500)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8089)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"控制台服务已启动: http://{args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
