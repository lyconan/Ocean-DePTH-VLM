#!/usr/bin/env bash
# =============================================================================
# train_ms_swift.sh — Qwen3-VL-32B QLoRA 微调一键脚本 (ms-swift, 方案A/推荐)
#
# 用法:
#   bash deploy/finetune/train_ms_swift.sh \
#       --data ./sft_out/swift_train.jsonl \
#       --val  ./sft_out/swift_val.jsonl \
#       [--model ./Qwen3-VL-32B-Instruct] [--out ./qwen3vl32b_sft_out]
#
# 数据格式: scripts/39_build_sft_dataset.py 或 39b_sft_to_swift.py 产出的
#           ms-swift messages 格式 jsonl (11图选择题 / 单图命名)
# 前置: pip install -r deploy/finetune/requirements_finetune.txt
# 硬件: 单卡 46GB (QLoRA 4bit) 起步; 双卡可加 GPUS=2
# =============================================================================
set -euo pipefail

DATA="" ; VAL="" ; MODEL="./Qwen3-VL-32B-Instruct" ; OUT="./qwen3vl32b_sft_out"
GPUS="${GPUS:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)  DATA="$2"  ; shift 2 ;;
    --val)   VAL="$2"   ; shift 2 ;;
    --model) MODEL="$2" ; shift 2 ;;
    --out)   OUT="$2"   ; shift 2 ;;
    *) echo "未知参数 $1"; exit 1 ;;
  esac
done
[[ -z "$DATA" || -z "$VAL" ]] && { echo "必须提供 --data 与 --val"; exit 1; }

# ---- 关键超参 (46GB 单卡 QLoRA 起点, 按显存实测调整) ----
LR=1e-4              # LoRA 学习率; 数据 <5k 条建议降为 5e-5, epoch 升到 3
EPOCH=2              # 本系统 96k 条协议数据的经验值
RANK=32              # LoRA 秩; 简单任务可降 16 省显存
MAXPIXEL="448x448_per_image"   # 与系统推理端一致, 训练/推理务必同尺寸

DSDATA="$DATA"
EXTRA=""
[[ -n "$VAL" ]] && EXTRA="--val_dataset $VAL"

echo ">>> QLoRA 微调: model=$MODEL gpus=$GPUS out=$OUT"
if [[ "$GPUS" -gt 1 ]]; then
  torchrun --nproc_per_node "$GPUS" $(command -v swift) sft \
    --model "$MODEL" --dataset "$DSDATA" $EXTRA \
    --train_type lora --quant_method bnb --quant_bits 4 \
    --lora_rank "$RANK" --lora_alpha $((RANK*2)) --lora_dropout 0.05 \
    --target_modules all-linear \
    --num_train_epochs "$EPOCH" --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 --learning_rate "$LR" --warmup_ratio 0.05 \
    --max_length 8192 --max_pixels "$MAXPIXEL" \
    --deepspeed zero2 \
    --save_steps 200 --save_total_limit 3 \
    --output_dir "$OUT"
else
  swift sft \
    --model "$MODEL" --dataset "$DSDATA" $EXTRA \
    --train_type lora --quant_method bnb --quant_bits 4 \
    --lora_rank "$RANK" --lora_alpha $((RANK*2)) --lora_dropout 0.05 \
    --target_modules all-linear \
    --num_train_epochs "$EPOCH" --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 --learning_rate "$LR" --warmup_ratio 0.05 \
    --max_length 8192 --max_pixels "$MAXPIXEL" \
    --save_steps 200 --save_total_limit 3 \
    --output_dir "$OUT"
fi

echo ">>> 完成。LoRA 适配器位于: $OUT/*/checkpoint-*/"
echo ">>> 部署: 见 docs/第三方接入教程_自建数据集微调Qwen3-VL-32B.md §5 (vLLM --lora-modules)"
