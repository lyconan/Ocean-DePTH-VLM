#!/usr/bin/env python3
"""全量 BioCLIP 2.5 嵌入提取 — 2,641 条目 / 27,908 张图 (每图一个 1024 维嵌入) (v1.0)

输入: data/processed/atlas_lookup/morphospecies_index.json (新图像库主索引)
输出: data/processed/bioclip25_embeddings/
      - emb_shard{i}.npz          (image_paths + embeddings fp16)
      - cache_shard{i}.json       (断点续跑: 已完成的 image_id 集合, 原子保存)
      - extraction_report.json    (汇总报告)

设计:
- 分片并行: --shard 0 --num-shards 2 在 GPU0, --shard 1 --num-shards 2 在 GPU1
- 断点续跑: cache 记录已完成的 image_id, 重启自动跳过
- 原子保存: checkpoint 先写 .tmp 再 os.replace, 每 CHECKPOINT_EVERY 张 + 结束时
- 失败图片记录在 cache["failed"], 不阻塞整体

用法 (双卡, 项目根目录):
  docker run --rm --gpus '"device=0"' -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
      -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
      -v $PWD:/work -w /work yolo34_train:v1 \
      /root/miniconda3/envs/vllm/bin/python scripts/07h_extract_bioclip25_embeddings.py --shard 0 --num-shards 2
  # 另起一个容器: --shard 1
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MSP_JSON = BASE / "data/processed/atlas_lookup/morphospecies_index.json"
OUT_DIR = BASE / "data/processed/bioclip25_embeddings"
CHECKPOINT_EVERY = 5000
BATCH_SIZE = 64
NUM_WORKERS = 0

sys.path.insert(0, str(BASE / "scripts"))
from utils.bioclip_embed import get_bioclip_model, extract_embeddings_batch, get_embedding_dim, MODEL_HUB_NAME


def collect_images():
    """从 morphospecies_index.json 收集全部 (msp_id, 相对路径) 对 (去重)."""
    with open(MSP_JSON, encoding="utf-8") as f:
        idx = json.load(f)
    recs = idx["morphospecies"]
    seen = {}
    missing = 0
    for msp in recs:
        msp_id = msp["morphospecies_id"]
        for rel in msp.get("images", []):
            if rel in seen:
                continue
            if (BASE / rel).exists():
                seen[rel] = msp_id
            else:
                missing += 1
    items = [(rel, msp_id) for rel, msp_id in seen.items()]
    print(f"收集图片: {len(items)} 张 (唯一路径), {missing} 张文件不存在(跳过)")
    return items


def atomic_save_npz(path, paths, embs):
    tmp = Path(str(path).replace('.npz', '.tmp.npz'))
    np.savez_compressed(tmp, image_paths=np.array(paths), embeddings=embs)
    os.replace(tmp, path)


def atomic_save_json(path, obj):
    tmp = Path(str(path).replace('.json', '.tmp.json'))
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = collect_images()
    shard_items = items[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(shard_items)} 张")

    cache_path = OUT_DIR / f"cache_shard{args.shard}.json"
    npz_path = OUT_DIR / f"emb_shard{args.shard}.npz"

    done = {}
    done_embs = {}
    failed = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        done = cache.get("done", {})
        failed = cache.get("failed", {})
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=False)
            prev_paths = data["image_paths"].tolist()
            prev_embs = data["embeddings"]
            for p, e in zip(prev_paths, prev_embs):
                if p in done:
                    done_embs[p] = e
            print(f"断点恢复: {len(done_embs)} 嵌入 + {len(done)} cache + {len(failed)} failed")
    todo = [(rel, msp) for rel, msp in shard_items if rel not in done and rel not in failed]
    print(f"待提取: {len(todo)} 张 (已完成 {len(done)}, 失败 {len(failed)})")

    model, processor = get_bioclip_model(args.device)
    dim = get_embedding_dim(model)
    device = next(model.parameters()).device
    print(f"device={device}, dim={dim}")

    t0 = time.time()
    n_done_this_run = 0
    for b0 in range(0, len(todo), args.batch_size):
        batch = todo[b0:b0 + args.batch_size]
        paths = [rel for rel, _ in batch]
        try:
            embs = extract_embeddings_batch(paths, model, processor,
                                            batch_size=args.batch_size,
                                            device=str(device),
                                            num_workers=NUM_WORKERS)
            for (rel, msp), e in zip(batch, embs):
                done[rel] = msp
                done_embs[rel] = e.numpy().astype(np.float16)
        except Exception as e:
            print(f"  batch 失败 ({e}), 降级单张...")
            for rel, msp in batch:
                try:
                    from utils.bioclip_embed import extract_embedding
                    emb = extract_embedding(rel, model, processor, device=str(device))
                    done[rel] = msp
                    done_embs[rel] = emb.numpy().astype(np.float16)
                except Exception as e2:
                    failed[rel] = str(e2)[:200]
                    print(f"  FAIL {rel}: {e2}")

        n_done_this_run += len(batch)
        if n_done_this_run % CHECKPOINT_EVERY < args.batch_size:
            embs_arr = np.stack([done_embs[r] for r in done_embs]) if done_embs else np.zeros((0, dim), np.float16)
            atomic_save_npz(npz_path, list(done_embs.keys()), embs_arr)
            atomic_save_json(cache_path, {"done": done, "failed": failed,
                                          "model": MODEL_HUB_NAME, "dim": dim})
            speed = n_done_this_run / max(time.time() - t0, 1)
            remain = (len(todo) - n_done_this_run) / max(speed, 0.1)
            print(f"  [ckpt] {len(done_embs)}/{len(shard_items)} ({speed:.1f} img/s, 剩余~{remain/60:.0f}min)")

    embs_arr = np.stack([done_embs[r] for r in done_embs]) if done_embs else np.zeros((0, dim), np.float16)
    atomic_save_npz(npz_path, list(done_embs.keys()), embs_arr)
    atomic_save_json(cache_path, {"done": done, "failed": failed,
                                  "model": MODEL_HUB_NAME, "dim": dim})
    dt = time.time() - t0
    print(f"shard {args.shard} 完成: {len(done_embs)} 成功, {len(failed)} 失败, {dt/60:.1f} min")
    print(f"输出: {npz_path}")

    report_path = OUT_DIR / "extraction_report.json"
    all_ok = sum((OUT_DIR / f"cache_shard{i}.json").exists() for i in range(args.num_shards))
    if all_ok == args.num_shards:
        total_done, total_fail = 0, 0
        for i in range(args.num_shards):
            with open(OUT_DIR / f"cache_shard{i}.json", encoding="utf-8") as f:
                c = json.load(f)
            total_done += len(c.get("done", {}))
            total_fail += len(c.get("failed", {}))
        atomic_save_json(report_path, {
            "model": MODEL_HUB_NAME, "dim": dim, "num_shards": args.num_shards,
            "total_images": len(items), "extracted": total_done, "failed": total_fail,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"全部分片完成: {total_done} 提取 / {total_fail} 失败 -> {report_path}")


if __name__ == "__main__":
    main()
