#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_install.py — 部署冒烟测试 (smoke test)

逐步检查并打印 PASS/FAIL:
  1. 资产就位: YOLO 权重 / BioCLIP 2.5 LoRA v2 适配器 / 质心 npz / morphospecies 索引
  2. Neo4j 连接 (Morphospecies / EntryImage / 向量索引统计)
  3. 质心加载 (centroids_v2.npz)
  4. 内置 demo 小图 YOLO 推理
  5. BioCLIP 2.5 (ViT-H/14, +LoRA v2) 嵌入
  6. 向量检索 top-5

用法:
  python scripts/setup/verify_install.py                 # 宿主机 (需已装依赖)
  docker compose --profile infer run --rm infer \
      python scripts/setup/verify_install.py             # 推理容器内 (推荐)

权重缺失时不崩溃, 给出 deploy/download_assets.sh 修复提示; 任一步 FAIL 则退出码非 0。
"""
import os
import sys
import json
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "scripts"))

WORK = os.environ.get("DEPTH_WORK", BASE)
YOLO_WEIGHTS = os.environ.get(
    "YOLO_WEIGHTS", os.path.join(WORK, "runs/detect/yolo11m_class34-2/weights/best.pt"))
LORA_DIR = os.path.join(WORK, "data/models/bioclip_lora/lora_adapter_v2")
CENTROIDS = os.path.join(WORK, "data/experiments/ft_20261102/centroids_v2.npz")
MSP_JSON = os.path.join(WORK, "data/processed/atlas_lookup/morphospecies_index.json")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

DOWNLOAD_HINT = ("修复提示: bash deploy/download_assets.sh  "
                 "(或 export MODELSCOPE_MODEL_REPO=<org>/<repo> 后重跑; 见 deploy/README.md)")

_results = []
_STATE = {}  # 跨步骤共享: det (YOLO 检测框) / emb (查询嵌入)


def step(name, fn):
    """执行一步, 记录 PASS/FAIL/SKIP 并即时打印。返回 True 表示通过。"""
    print(f"\n[步骤] {name}", flush=True)
    try:
        msg = fn() or ""
        print(f"  ✅ PASS  {msg}", flush=True)
        _results.append((name, "PASS", msg))
        return True
    except SkipStep as e:
        print(f"  ⏭️  SKIP  {e}", flush=True)
        _results.append((name, "SKIP", str(e)))
        return True
    except Exception as e:
        print(f"  ❌ FAIL  {e}", flush=True)
        if os.environ.get("VERIFY_VERBOSE", "0") == "1":
            traceback.print_exc()
        _results.append((name, "FAIL", str(e)))
        return False


class SkipStep(Exception):
    """前置条件不满足, 跳过 (不计 FAIL)"""


# --------------------------------------------------------------------------- #
# 步骤实现
# --------------------------------------------------------------------------- #

def check_assets():
    missing, found = [], []
    for label, path in (("YOLO 权重", YOLO_WEIGHTS),
                        ("BioCLIP 2.5 LoRA v2 适配器", LORA_DIR),
                        ("质心 npz", CENTROIDS),
                        ("morphospecies 索引", MSP_JSON)):
        (found if os.path.exists(path) else missing).append(f"{label}: {path}")
    for f in found:
        print(f"  - OK  {f}")
    if missing:
        for m in missing:
            print(f"  - 缺失  {m}")
        raise FileNotFoundError(f"缺失 {len(missing)} 项资产。{DOWNLOAD_HINT}")


def check_neo4j():
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        drv.verify_connectivity()
        with drv.session() as s:
            n_msp = s.run("MATCH (m:Morphospecies) RETURN count(m) AS n").single()["n"]
            n_img = s.run("MATCH (e:EntryImage) RETURN count(e) AS n").single()["n"]
        if n_msp == 0:
            raise RuntimeError(
                "Morphospecies 为 0 — 知识图谱尚未导入, 见 deploy/README.md '导入知识图谱'")
        idx_names = []
        with drv.session() as s:
            for r in s.run("SHOW INDEXES YIELD name, type WHERE type = 'VECTOR'"):
                idx_names.append(r["name"])
        print(f"  - Bolt {NEO4J_URI} 连接成功; Morphospecies={n_msp}, EntryImage={n_img}")
        print(f"  - 向量索引: {idx_names or '无 (可运行 36/38 号基准前重建)'}")
        if n_img == 0:
            raise SkipStep("EntryImage 为 0 — 检索步骤将退化为仅质心比对")
        return f"Morphospecies={n_msp}, EntryImage={n_img}"
    finally:
        drv.close()


def check_centroids():
    import numpy as np
    z = np.load(CENTROIDS)
    C, mids = z["C"], z["mids"]
    if C.shape[0] == 0:
        raise RuntimeError("质心矩阵为空")
    return f"质心矩阵 {C.shape}, dtype={C.dtype} ({len(mids)} 个形态种)"


def _make_demo_image():
    """内置合成 demo 图 (免外发素材): 深蓝背景 + 类生物亮斑"""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (960, 720))
    px = img.load()
    for y in range(img.height):  # 深海蓝渐变
        for x in range(img.width):
            px[x, y] = (8 + x * 10 // img.width, 24 + y * 20 // img.height, 58)
    dr = ImageDraw.Draw(img)
    dr.ellipse([380, 300, 560, 470], fill=(212, 180, 140))   # 主体
    dr.ellipse([420, 330, 470, 380], fill=(240, 230, 200))   # 高光
    dr.ellipse([640, 480, 720, 560], fill=(150, 130, 110))   # 次要目标
    return img


def check_yolo():
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)
    img = _make_demo_image()
    res = model.predict(img, imgsz=960, conf=0.05,
                        device=None, verbose=False)[0]
    n = 0 if res.boxes is None else len(res.boxes)
    print(f"  - 检出 {n} 个框 (合成图检出为 0 属正常, 只验证权重可加载与推理链路)")
    boxes = res.boxes
    if n == 0:
        _STATE["det"] = None  # 无框时后续用整图做嵌入
        return "YOLO 权重加载与推理链路正常 (无检出)"
    b = boxes[0]
    xyxy = [float(v) for v in b.xyxy[0].tolist()]
    cls_name = model.names.get(int(b.cls[0]), str(int(b.cls[0])))
    _STATE["det"] = {"box": tuple(xyxy), "name": cls_name, "conf": float(b.conf[0])}
    return f"检出框 {_STATE['det']['box']} ({cls_name} {_STATE['det']['conf']:.2f})"


def check_embed():
    from utils.bioclip_embed import get_bioclip_model, extract_embedding
    from PIL import Image
    model, proc = get_bioclip_model()  # 默认叠加 LoRA v2 (缺失时打印警告退回基座)
    img = _make_demo_image()
    det = _STATE.get("det")
    crop = img.crop(det["box"]) if det else img
    emb = extract_embedding(crop, model, proc)
    if emb is None or float(emb.norm()) < 0.5:
        raise RuntimeError("嵌入向量异常 (范数过小)")
    import torch
    v = emb / emb.norm().clamp_min(1e-12)
    _STATE["emb"] = v.cpu().numpy().astype("float32")
    return f"嵌入维度 {_STATE['emb'].shape[0]} (查询源: {'YOLO检测框' if det else '整图'})"


def check_retrieve():
    import numpy as np
    emb = _STATE["emb"]
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    hits = []
    try:
        with drv.session() as s:
            try:
                rows = s.run("""
                    CALL db.index.vector.queryNodes('entryimage_embeddings', 5, $emb)
                    YIELD node, score RETURN node.msp_id AS mid, score
                """, emb=emb.tolist()).data()
                hits = [(r["mid"], r["score"]) for r in rows]
            except Exception:
                # 回退: 暴力拉样例嵌入比对 (仅冒烟用)
                print("  - 向量索引不可用, 回退为抽样暴力比对")
                data = s.run(
                    "MATCH (e:EntryImage) RETURN e.msp_id AS mid, e.embedding AS emb "
                    "LIMIT 2000").data()
                if not data:
                    raise SkipStep("库中无嵌入可检索 (EntryImage=0), 跳过检索验证")
                M = np.asarray([d["emb"] for d in data], dtype="float32")
                M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
                sims = M @ emb
                top = np.argsort(-sims)[:5]
                hits = [(data[i]["mid"], float(sims[i])) for i in top]
    finally:
        drv.close()
    # mid -> latin 显示名
    latin = {}
    try:
        idx = json.load(open(MSP_JSON, encoding="utf-8"))
        for rec in idx["morphospecies"]:
            latin[rec.get("morphospecies_id")] = rec.get("latin") or ""
    except Exception:
        pass
    print(f"  - 查询: {'YOLO检测框 ' + str(_STATE['det']['box']) if _STATE.get('det') else '整图(无检出)'}")
    for i, (mid, sc) in enumerate(hits, 1):
        print(f"    top{i}  {mid}  {latin.get(mid, '')}  score={sc:.4f}")
    if not hits:
        raise RuntimeError("检索无结果")
    return f"top1={hits[0][0]} ({latin.get(hits[0][0], '')})"


def main():
    print("=" * 64)
    print("Ocean-DePTH-VLM 部署冒烟测试")
    print(f"  BASE={BASE}\n  NEO4J_URI={NEO4J_URI}\n  YOLO_WEIGHTS={YOLO_WEIGHTS}")
    print("=" * 64)

    step("1/6 资产就位 (权重/适配器/质心/索引)", check_assets)
    step("2/6 Neo4j 连接与图谱统计", check_neo4j)
    step("3/6 质心加载", check_centroids)
    if step("4/6 YOLO 推理 (内置 demo 图)", check_yolo):
        if step("5/6 BioCLIP 2.5 嵌入", check_embed):
            step("6/6 向量检索 top-5", check_retrieve)
        else:
            print("\n(嵌入失败, 跳过检索)")

    # ---- 汇总 ----
    print("\n" + "=" * 64)
    n_pass = sum(1 for _, s, _ in _results if s == "PASS")
    n_fail = sum(1 for _, s, _ in _results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in _results if s == "SKIP")
    for name, s, msg in _results:
        print(f"  [{s:^4}] {name}" + (f" — {msg}" if s == "FAIL" else ""))
    print(f"\n总计: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")
    print("=" * 64)
    if n_fail:
        print("冒烟测试未通过。若提示缺权重: " + DOWNLOAD_HINT)
        sys.exit(1)
    print("冒烟测试通过 ✅  可继续: 复现盲测基准 (36 构建 → 38 跑分), 见 deploy/README.md")


if __name__ == "__main__":
    main()
