#!/usr/bin/env bash
# =============================================================================
# download_assets.sh — 从 ModelScope (魔搭) 下载推理所需模型权重/数据资产
#
# 下载内容:
#   模型仓库 (MODEL_REPO, ~265MB):
#     yolo11m_class34_best.pt   34 纲 YOLO 检测权重 (161MB)
#                               -> runs/detect/yolo11m_class34-2/weights/best.pt
#     bioclip_lora_v2/          BioCLIP 2.5 LoRA v2 适配器 (84MB)
#                               -> data/models/bioclip_lora/lora_adapter_v2/
#     centroids_v2.npz          2,638 形态种质心 (检索融合用)
#                               -> data/experiments/ft_20261102/centroids_v2.npz
#   数据集仓库 (DATASET_REPO, ~365GB, 8 个 tar 分片, 后续上传):
#     fathomnet_class_yolo_part{00..07}.tar -> 解包至 data/processed/fathomnet_class_yolo/
#
# 用法:
#   bash deploy/download_assets.sh                  # 仅下载模型权重 (~300MB)
#   bash deploy/download_assets.sh --with-dataset   # 额外下载 365GB 训练数据集
#
# 前置: pip install modelscope  (脚本会自动检测并安装)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 默认仓库已发布; 如需镜像/私有副本, 用环境变量覆盖:
#   export MODELSCOPE_MODEL_REPO=<org>/<model-repo>
#   export MODELSCOPE_DATASET_REPO=<org>/<dataset-repo>
# 详见 docs/03_distribution.md 的发布清单。
# ---------------------------------------------------------------------------
MODEL_REPO="${MODELSCOPE_MODEL_REPO:-ConanLee126/depth-vlm-assets}"
DATASET_REPO="${MODELSCOPE_DATASET_REPO:-ConanLee126/depth-vlm-dataset}"
# ---------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/data/models"
DATA_DIR="$ROOT/data"
STAGE="$(mktemp -d /tmp/depth_assets.XXXXXX)"
WITH_DATASET=0
[[ "${1:-}" == "--with-dataset" ]] && WITH_DATASET=1

log()  { printf '\n\033[1;34m[download_assets]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[download_assets] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

trap 'rm -rf "$STAGE"' EXIT

log "仓库根目录: $ROOT"
log "模型仓库:   $MODEL_REPO (~265MB)"
[[ $WITH_DATASET -eq 1 ]] && log "数据集仓库: $DATASET_REPO (~365GB, 确认磁盘空间充足; 数据集仓库后续上传)"

# --- 0. modelscope CLI ---
if ! python3 -c "import modelscope" >/dev/null 2>&1; then
    log "安装 modelscope ..."
    pip install -q modelscope
fi

# --- 1. 下载模型仓库到暂存区 ---
log "拉取模型仓库 -> $STAGE"
python3 - "$MODEL_REPO" "$STAGE" <<'PY'
import sys
from modelscope import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
print("模型仓库下载完成")
PY

# --- 2. 落位: YOLO 权重 ---
YOLO_DST="$ROOT/runs/detect/yolo11m_class34-2/weights"
YOLO_SRC="$(find "$STAGE" -name 'yolo11m_class34_best.pt' | head -1)"
if [[ -n "$YOLO_SRC" ]]; then
    mkdir -p "$YOLO_DST"
    install -m 644 "$YOLO_SRC" "$YOLO_DST/best.pt"
    log "YOLO 权重 -> $YOLO_DST/best.pt ($(du -h "$YOLO_DST/best.pt" | cut -f1))"
else
    log "警告: 暂存区未找到 yolo11m_class34_best.pt (请核对模型仓库内容)"
fi

# --- 3. 落位: BioCLIP 2.5 LoRA v2 适配器 ---
LORA_DST="$MODELS_DIR/bioclip_lora/lora_adapter_v2"
if [[ -d "$STAGE/bioclip_lora_v2" ]]; then
    mkdir -p "$(dirname "$LORA_DST")"
    rm -rf "$LORA_DST"
    cp -r "$STAGE/bioclip_lora_v2" "$LORA_DST"
    log "BioCLIP 2.5 LoRA v2 -> $LORA_DST ($(du -sh "$LORA_DST" | cut -f1))"
else
    log "警告: 暂存区未找到 bioclip_lora_v2/ 目录 (无适配器将退回 BioCLIP 2.5 基座 ViT-H/14)"
fi

# --- 4. 落位: 质心 npz ---
CEN_DST="$DATA_DIR/experiments/ft_20261102"
CEN_SRC="$(find "$STAGE" -name 'centroids_v2.npz' | head -1)"
if [[ -n "$CEN_SRC" ]]; then
    mkdir -p "$CEN_DST"
    install -m 644 "$CEN_SRC" "$CEN_DST/centroids_v2.npz"
    log "质心索引   -> $CEN_DST/centroids_v2.npz"
else
    log "警告: 暂存区未找到 centroids_v2.npz (cfusion 变体将不可用)"
fi

# --- 5. 落位: Neo4j 图谱 dump (检索/元数据必需, ~458MB) ---
NK_DST="$DATA_DIR/neo4j"
NK_SRC="$(find "$STAGE" -maxdepth 2 -name 'neo4j.dump' | head -1)"
if [[ -n "$NK_SRC" ]]; then
    mkdir -p "$NK_DST"
    install -m 644 "$NK_SRC" "$NK_DST/neo4j.dump"
    log "图谱 dump   -> $NK_DST/neo4j.dump"
    log "  导入方法见 deploy/README.md §4 路线 B (neo4j-admin database load)"
else
    log "警告: 暂存区未找到 neo4j.dump (Neo4j 将为空库, 检索环节无法运行!)"
fi

# --- 6. 可选: 365GB 训练数据集 (8 个 tar 分片) ---
if [[ $WITH_DATASET -eq 1 ]]; then
    DS_DST="$DATA_DIR/dataset_dl"
    mkdir -p "$DS_DST"
    log "拉取数据集仓库 -> $DS_DST (耗时较长)"
    python3 - "$DATASET_REPO" "$DS_DST" <<'PY'
import sys
from modelscope import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
print("数据集仓库下载完成")
PY
    log "合并分片并解包 -> $DATA_DIR/processed/fathomnet_class_yolo/"
    mkdir -p "$DATA_DIR/processed/fathomnet_class_yolo"
    cat "$DS_DST"/fathomnet_class_yolo_part*.tar | tar -xf - -C "$DATA_DIR/processed/fathomnet_class_yolo"
    log "数据集解包完成; 可删除 $DS_DST 释放空间"
fi

log "完成。下一步: docker compose up -d neo4j, 然后 python scripts/setup/verify_install.py"
