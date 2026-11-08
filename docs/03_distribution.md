# 03 · 发布清单 (Distribution Checklist)

开源发布前, 按本清单把"代码仓库之外"的资产分发到三个渠道:
**魔搭模型仓库** (推理权重)、**魔搭数据集仓库** (训练数据)、**Docker Hub** (推理镜像)。

---

## 1. 魔搭模型仓库 (`ConanLee126/depth-vlm-assets`, ~265MB) — **已发布** (2026-11-05)

仓库地址: https://modelscope.cn/models/ConanLee126/depth-vlm-assets (公开可读, 匿名可下载, License: Apache-2.0)

仓库内容 (文件名与 `deploy/download_assets.sh` 的落位逻辑对应, 勿改名):

| 仓库内路径 | 内容 | 大小 |
|---|---|---|
| `yolo11m_class34_best.pt` | 34 纲 YOLO11m 检测权重 (即 `runs/detect/yolo11m_class34-2/weights/best.pt`) | 161MB |
| `bioclip_lora_v2/` | BioCLIP 2.5 LoRA v2 适配器 (即 `data/models/bioclip_lora/lora_adapter_v2/` 整目录) | 84MB |
| `centroids_v2.npz` | 2,638 形态种质心矩阵 (`C` + `mids` + `cnt`) | 11MB |
| `atlas_lookup/morphospecies_index.json` | 形态种主索引 (与仓库内 `data/processed/atlas_lookup/` 同源, 冗余分发便于离线) | 7.5MB |
| `taxonomy/` | WoRMS 纲别→34 类别名映射与中文物种名对照 (6 个文件, 含 `yolo34_class_alias_map.json`) | 0.9MB |
| `configuration.json` | ModelScope 仓库元数据 (上传 SDK 自动生成) | <1KB |
| `README.md` | 模型卡 (文件用途、基座标注 BioCLIP 2.5 ViT-H/14、下载方法、指标、许可) | — |

> 注: `*.pt/*.npz` 已被仓库 `.gitignore` 排除, 因此必须经模型仓库分发;
> 两个 json 索引已随代码仓库分发, 模型仓库内为冗余备份。
> BioCLIP 2.5 基座 (ViT-H/14, `imageomics/bioclip-2.5-vith14`, ~2.5GB) **不随模型仓库分发**,
> 首次运行时从 HuggingFace 自动下载 (支持 `HF_HUB_OFFLINE=1` 离线); 仓库只分发 LoRA v2 适配器 (模型卡中已标注基座)。

下载验证 (2026-11-05, 匿名无 token): 仓库文件列表可匿名读取;
`modelscope download` 拉取 README/质心 npz/别名映射大小与源一致;
对 `yolo11m_class34_best.pt` 发起 Range 请求返回 206 且首字节为合法 PyTorch zip 头。

用户下载命令:

```bash
pip install modelscope
# 整仓下载
modelscope download --model ConanLee126/depth-vlm-assets --local_dir ./depth-vlm-assets
# 或一键下载并自动落位
bash deploy/download_assets.sh
```

发布者增量同步命令 (首传已完成, 后续改动资产后重跑对应行即可):

```bash
pip install modelscope
modelscope login --token <YOUR_MS_TOKEN>

modelscope upload ConanLee126/depth-vlm-assets yolo11m_class34_best.pt --repo-type model
modelscope upload ConanLee126/depth-vlm-assets bioclip_lora_v2/ --repo-type model
modelscope upload ConanLee126/depth-vlm-assets centroids_v2.npz --repo-type model
modelscope upload ConanLee126/depth-vlm-assets atlas_lookup/ --repo-type model
modelscope upload ConanLee126/depth-vlm-assets taxonomy/ --repo-type model
```

`deploy/download_assets.sh` 顶部的 `MODELSCOPE_MODEL_REPO` 默认值已回填为
`ConanLee126/depth-vlm-assets` (环境变量可覆盖)。

## 2. 魔搭数据集仓库 (`ConanLee126/depth-vlm-dataset`, ~365GB) — **已发布 (2026-11-08)**

内容: 34 纲 YOLO 检测训练集 `fathomnet_class_yolo/` (images 符号链接已解包为实文件)。

> 状态: **已发布** — part00~07 共 8 分片 (383.7GB) 已全部上传至魔搭, 远端文件大小逐片校验通过, 匿名可下载。

打包为 8 个 tar 分片 (用仓库自带 `scripts/package_class_yolo.sh`, 默认 8 分片并行打包):

```bash
cd <repo-root>
bash scripts/package_class_yolo.sh 8            # 生成 fathomnet_class_yolo_part{00..07}.tar
md5sum data/processed/fathomnet_class_yolo_part*.tar > fathomnet_class_yolo.md5
```

上传:

```bash
modelscope upload ConanLee126/depth-vlm-dataset fathomnet_class_yolo_part00.tar --repo-type dataset
# ... 依次上传 part01..part07 与 fathomnet_class_yolo.md5
```

用户侧恢复: `bash deploy/download_assets.sh --with-dataset` (脚本内自动合并分片 + tar 解包)。

## 3. Docker Hub (`conanlee126/depth-vlm-train:v1.1`) — 已推送 (v1: 2026-11-06 / v1.1: 2026-11-06)

```bash
# 拉取 (全量镜像 ~26.9GB, 含训练+推理全部依赖, 首推耗时 ~17h 国内出口带宽受限)
docker pull conanlee126/depth-vlm-train:v1.1

docker run --rm --gpus all conanlee126/depth-vlm-train:v1.1 \
    python -c "import ultralytics, open_clip, neo4j, peft; print('deps ok')"
```

> **v1.1 (2026-11-06)**: 修复 v1 缺陷 — 镜像内缺 `peft` (BioCLIP LoRA v2 适配器加载必需)。
> v1 的容器内 pip 默认走中科大镜像源 (该网络环境不通) 导致安装静默失败; v1.1 改用
> 官方源 `pypi.org` 安装 `peft 0.21.2` 后提交, 基础层与 v1 完全复用。
> **新用户请一律使用 v1.1**; v1 保留但不再维护。

> 注: 当前镜像为 yolo34_train 训练环境全量打包 (26.9GB), 已满足推理与复现需求;
> 瘦身推理镜像 (`Dockerfile.infer`, 目标 <10GB) 可后续按下方构建流程另行发布 v0.2。

> 镜像不含数据集与推理权重 — 权重走魔搭 (`deploy/download_assets.sh`), 数据集走
> `--with-dataset`; 使用时经 `deploy/docker-compose.yml` 的 `infer` profile 挂载仓库目录。

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

- [x] 模型仓库资产已上传至 `ConanLee126/depth-vlm-assets` (2026-11-05, 265MB/14 文件, 匿名下载验证通过); 干净目录跑 `deploy/download_assets.sh` 验证落位仍建议发布前做一次
- [ ] 数据集 8 分片 md5 校验通过, 下载端 `--with-dataset` 解包成功 (**分片已发布; md5 清单生成中**)
- [ ] Docker Hub 镜像拉取 + `scripts/setup/verify_install.py` 6 步全 PASS
- [x] `deploy/download_assets.sh` 占位 repo id 已替换为真实地址 (`ConanLee126/depth-vlm-assets`)
- [ ] 仓库根 `LICENSE` (Apache-2.0) 与数据声明就位
- [ ] `deploy/README.md` 指标表与最新定版一致 (cfusion 49.6 / 68.9 / 72.6)
