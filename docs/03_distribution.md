# 03 · 发布清单 (Distribution Checklist)

开源发布前, 按本清单把"代码仓库之外"的资产分发到三个渠道:
**魔搭模型仓库** (推理权重)、**魔搭数据集仓库** (训练数据)、**Docker Hub** (推理镜像)。

---

## 1. 魔搭模型仓库 (`<org>/depth-vlm-assets`, ~300MB)

仓库内容 (文件名与 `deploy/download_assets.sh` 的落位逻辑对应, 勿改名):

| 仓库内路径 | 内容 | 大小 |
|---|---|---|
| `yolo11m_class34_best.pt` | 34 纲 YOLO11m 检测权重 (即 `runs/detect/yolo11m_class34-2/weights/best.pt`) | 161MB |
| `bioclip_lora_v2/` | BioCLIP 2.5 LoRA v2 适配器 (即 `data/models/bioclip_lora/lora_adapter_v2/` 整目录) | 117MB |
| `centroids_v2.npz` | 2,638 形态种质心矩阵 (`C` + `mids`) | ~20MB |
| `atlas_lookup/morphospecies_index.json` | 形态种主索引 (与仓库内 `data/processed/atlas_lookup/` 同源, 冗余分发便于离线) | 7.4MB |
| `taxonomy/yolo34_class_alias_map.json` | WoRMS 纲别 → 34 类别名映射 | <1KB |
| `README.md` | 模型卡 (指标、环境、引用方式) | — |

> 注: `*.pt/*.npz` 已被仓库 `.gitignore` 排除, 因此必须经模型仓库分发;
> 两个 json 索引已随代码仓库分发, 模型仓库内为冗余备份。
> BioCLIP 2.5 基座 (ViT-H/14, `imageomics/bioclip-2.5-vith14`, ~2.5GB) **不随模型仓库分发**,
> 首次运行时从 HuggingFace 自动下载 (支持 `HF_HUB_OFFLINE=1` 离线); 仓库只分发 LoRA v2 适配器。

上传命令:

```bash
pip install modelscope
modelscope login --token <YOUR_MS_TOKEN>

# 整目录上传 (推荐, 首传后增量同步)
modelscope upload <org>/depth-vlm-assets yolo11m_class34_best.pt --repo-type model
modelscope upload <org>/depth-vlm-assets bioclip_lora_v2/ --repo-type model
modelscope upload <org>/depth-vlm-assets centroids_v2.npz --repo-type model
modelscope upload <org>/depth-vlm-assets atlas_lookup/ --repo-type model
modelscope upload <org>/depth-vlm-assets taxonomy/ --repo-type model
```

发布者最后一步: 把 `deploy/download_assets.sh` 顶部的
`MODELSCOPE_MODEL_REPO` 占位符改为 `<org>/depth-vlm-assets`。

## 2. 魔搭数据集仓库 (`<org>/depth-vlm-dataset`, ~365GB) — **后续上传**

内容: 34 纲 YOLO 检测训练集 `fathomnet_class_yolo/` (images 符号链接已解包为实文件)。

> 状态: **后续上传** — 训练数据集 (~365GB) 尚未上传至魔搭, 上传前模型仓库已可支撑
> 全部推理与评测链路 (检测权重 / LoRA 适配器 / 质心 / 索引)。

打包为 8 个 tar 分片 (用仓库自带 `scripts/package_class_yolo.sh`, 默认 8 分片并行打包):

```bash
cd <repo-root>
bash scripts/package_class_yolo.sh 8            # 生成 fathomnet_class_yolo_part{00..07}.tar
md5sum data/processed/fathomnet_class_yolo_part*.tar > fathomnet_class_yolo.md5
```

上传:

```bash
modelscope upload <org>/depth-vlm-dataset fathomnet_class_yolo_part00.tar --repo-type dataset
# ... 依次上传 part01..part07 与 fathomnet_class_yolo.md5
```

用户侧恢复: `bash deploy/download_assets.sh --with-dataset` (脚本内自动合并分片 + tar 解包)。

## 3. Docker Hub (`lyconan/depth-vlm-infer:v0.1`)

```bash
# 构建 (仓库根目录执行; 瘦身推理镜像, 目标 <10GB)
docker build -f deploy/Dockerfile.infer -t lyconan/depth-vlm-infer:v0.1 .

docker images lyconan/depth-vlm-infer          # 确认体积
docker run --rm --gpus all lyconan/depth-vlm-infer:v0.1 python -c "import ultralytics, open_clip, neo4j; print('deps ok')"

docker login
docker push lyconan/depth-vlm-infer:v0.1
```

镜像内只含 `scripts/ utils/ config/` 与 pip 依赖 — 不含数据与权重, 用户经
`deploy/docker-compose.yml` 的 `infer` profile 挂载仓库目录使用。

## 4. License 建议

| 部分 | 建议 | 说明 |
|---|---|---|
| 代码 (scripts/ utils/ config/ deploy/) | **Apache-2.0** | 宽松、含专利授权条款, 适合科研工程; 仓库根放 `LICENSE` |
| FathomNet 图像/标注 | **CC-BY-4.0** (遵循上游) | 数据不可用 Apache 闭源; 需署名, 引用格式见下 |
| ultralytics (YOLO11) | **AGPL-3.0** (上游) | 注意: 用 ultralytics 训练/推理的代码分发受 AGPL 传染, 模型卡中需声明; 商用授权找 ultralytics |
| BioCLIP 2.5 | 遵循 `imageomics/bioclip-2.5-vith14` 上游许可 | LoRA 适配器随模型仓库分发时注明 |
| WoRMS 分类数据 | 遵循 WoRMS 使用条款 (非商用署名) | 图谱节点来源需在模型卡/README 说明 |

FathomNet 引用格式 (模型卡与论文中标注):

```bibtex
@article{katz2022fathomnet,
  title   = {FathomNet: A global image database for enabling artificial intelligence in the ocean},
  author  = {Katz, Steven M. and Orenstein, Erick and others},
  journal = {Scientific Reports},
  year    = {2022},
  doi     = {10.1038/s41598-022-19939-z},
  url     = {https://fathomnet.org}
}
```

数据声明模板: *"This work uses data from FathomNet, licensed under CC-BY-4.0."*

## 5. 发布前检查项

- [ ] 模型仓库 6 项资产上传完毕, 用干净目录跑 `deploy/download_assets.sh` 验证落位
- [ ] 数据集 8 分片 md5 校验通过, 下载端 `--with-dataset` 解包成功 (**后续上传, 暂未就绪**)
- [ ] Docker Hub 镜像拉取 + `scripts/setup/verify_install.py` 6 步全 PASS
- [ ] `deploy/download_assets.sh` 占位 repo id 已替换为真实地址
- [ ] 仓库根 `LICENSE` (Apache-2.0) 与数据声明就位
- [ ] `deploy/README.md` 指标表与最新定版一致 (cfusion 49.6 / 68.9 / 72.6)
