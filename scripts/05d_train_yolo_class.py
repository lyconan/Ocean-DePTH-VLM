#!/usr/bin/env python3
"""
05d_train_yolo_class.py — 34 纲级 YOLO11 检测器训练 (在训练服务器上运行)

用法 (在数据包解压后的项目根目录):
  python scripts/05d_train_yolo_class.py                  # 默认 yolo11m, imgsz=960
  python scripts/05d_train_yolo_class.py --model yolo11l.pt --epochs 150
  python scripts/05d_train_yolo_class.py --batch 16       # 显存不够时手动指定

环境 (本机 101, 双 A40):
  # yolo34_train:v1 = 由 depth_vlm_vlm_72b 容器 commit 而来 (含 ultralytics 8.4.130 / torch 2.7.0+cu126)
  # 注意: 必须 -e CUDA_VISIBLE_DEVICES=0,1 覆盖镜像里带的 =0, 否则只看到 1 张卡
  docker run -d --name yolo34_class34 --gpus all -e CUDA_VISIBLE_DEVICES=0,1 \
      --ipc=host \
      -v /mnt/oneocean_200TBnas_feitian/oneastronomy/lishaojie/Ocean-DePTH-VLM:/work \
      -v /mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet:/mnt/oneocean_200TBnas_feitian/opc-03/Sea_Data/FathomNet:ro \
      -w /work \
      yolo34_train:v1 \
      /root/miniconda3/envs/vllm/bin/python scripts/05d_train_yolo_class.py --device 0,1 --workers 12 --batch 32
  # 注: 第二个 -v 必须, images/ 是指向 FathomNet 原图的符号链接, 不挂载则容器内读不到图

训练配置依据:
  - MBARI/FathomNet 实践: 大框量 (本项目 68 万框/12.6 万图) + ultralytics 标准增强
  - 推理阈值建议 conf=0.025, iou=0.4 (MBARI monster 检测器生产配置, 高召回)
  - 34 类长尾已比 FathomNet 官方 2553 类大幅缓解, focal γ=1.0 即可
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/processed/fathomnet_class_yolo',
                    help='数据集根目录 (含 data.yaml/images/labels)')
    ap.add_argument('--model', default='yolo11m.pt',
                    help='初始权重: yolo11m.pt / yolo11l.pt / 本地任意 .pt')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--batch', type=int, default=-1, help='-1=按显存自适应(60%%)')
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--name', default='yolo11m_class34')
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    if not (data_root / 'data.yaml').exists():
        sys.exit(f'找不到 {data_root}/data.yaml, 请先运行 06d_build_class_yolo.py 或解压数据包')

    # data.yaml 的 path 写成运行时的绝对路径, 避免跨服务器路径解析歧义
    yaml_text = (data_root / 'data.yaml').read_text()
    lines = ['path: ' + str(data_root)] + [l for l in yaml_text.splitlines()
                                            if not l.startswith('path:')]
    local_yaml = data_root / 'data.local.yaml'
    local_yaml.write_text('\n'.join(lines) + '\n')

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=str(local_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        # ultralytics 8.4: 相对路径 project 会被拼到 runs_dir/<task>/ 下造成嵌套, 必须绝对路径
        project=str(Path('runs/detect').resolve()),
        # --- 优化器 ---
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        # --- 长尾/小目标增强 ---
        # 注: fl_gamma 已被 ultralytics 8.4 移除 (BCE 默认), 不再可调
        copy_paste=0.3,        # 稀有类关键增强
        mixup=0.1,
        mosaic=1.0,
        close_mosaic=15,       # 最后15轮关mosaic稳定收敛
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # 深海色调敏感, 适度
        degrees=10, translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5,
        # --- 正则/调度 ---
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=30,
        cache=False,
        amp=True,
    )

    # 训练完自动出 per-class mAP 报告
    best = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    print(f'\nbest.pt: {best}')
    metrics = YOLO(str(best)).val(data=str(local_yaml), imgsz=args.imgsz,
                                   device=args.device, split='val',
                                   conf=0.025, iou=0.4)   # MBARI 生产阈值
    print(f"\nmAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print('完成. per-class 明细见上方验证输出 / runs/detect/' + args.name)


if __name__ == '__main__':
    main()
