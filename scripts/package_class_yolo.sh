#!/bin/bash
# package_class_yolo.sh — 并行分片打包 34 纲 YOLO 数据集 (~358G) 供转移到训练服务器
#
# 用法: bash scripts/package_class_yolo.sh [分片数] [输出目录]
#       分片数默认 8; 输出默认 data/processed/
#
# 为什么不用单个 tar:
#   1. NFS 上 15 万个小文件 + tar 默认 10KB 记录块 → 实测仅 2 MB/s (46 小时)
#   2. -b 4096 (2MB 记录块) 单流实测 27 MB/s, 8 分片并行 ≈ 150-200 MB/s
#   3. PNG 已压缩, 不用 gzip (零收益且慢数倍)
#
# 转移与解压 (训练服务器):
#   scp fathomnet_class_yolo_part*.tar user@trainserver:/path/
#   for f in fathomnet_class_yolo_part*.tar; do tar xf "$f" -C Ocean-DePTH-VLM/data/processed/; done
#   python scripts/05d_train_yolo_class.py --data-root data/processed/fathomnet_class_yolo
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/data/processed/fathomnet_class_yolo"
N="${1:-8}"
OUTDIR="${2:-$ROOT/data/processed}"

[ -f "$SRC/data.yaml" ] || { echo "先运行 06d_build_class_yolo.py"; exit 1; }

echo "== 生成分片列表 ($N 片, round-robin) =="
python3 - "$SRC" "$OUTDIR" "$N" <<'PYEOF'
import sys
from pathlib import Path
src, outdir, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
base = src.name  # fathomnet_class_yolo
files = ['data.yaml', 'class_table.json']
files += [f'labels/train/{p.name}' for p in sorted((src/'labels/train').glob('*.txt'))]
files += [f'labels/val/{p.name}' for p in sorted((src/'labels/val').glob('*.txt'))]
files += [f'images/train/{p.name}' for p in sorted((src/'images/train').glob('*.png'))]
files += [f'images/val/{p.name}' for p in sorted((src/'images/val').glob('*.png'))]
lists = [open(outdir/f'.partlist_{i:02d}.txt', 'w') for i in range(n)]
for i, f in enumerate(files):
    lists[i % n].write(f'{base}/{f}\n')
for fh in lists:
    fh.close()
print(f'共 {len(files)} 个文件 → {n} 个分片列表')
PYEOF

echo "== 并行打包 (tar -b 4096, 符号链接解引用) =="
T0=$(date +%s)
pids=()
for i in $(seq -f "%02g" 0 $((N-1))); do
    tar -cf "$OUTDIR/fathomnet_class_yolo_part$i.tar" \
        -b 4096 -C "$(dirname "$SRC")" -h \
        --files-from "$OUTDIR/.partlist_$i.txt" &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "!! 有分片打包失败, 检查上方 tar 错误"; exit 1; }

echo "== 完成 (耗时 $(( ($(date +%s)-T0)/60 )) 分钟) =="
ls -lh "$OUTDIR"/fathomnet_class_yolo_part*.tar | awk '{print $5, $9}'
du -shc "$OUTDIR"/fathomnet_class_yolo_part*.tar | tail -1
rm -f "$OUTDIR"/.partlist_*.txt
echo "解压: for f in fathomnet_class_yolo_part*.tar; do tar xf \$f -C Ocean-DePTH-VLM/data/processed/; done"
