# Ocean-DePTH-VLM 一键部署指南

本目录提供从零复现"检测 → 嵌入 → 知识图谱向量检索"全链路的最短路径:
**装环境 → 拉权重 → 起图库 → 冒烟验证 → 复现盲测基准**。

---

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| GPU | NVIDIA ≥ 8GB 显存 (推荐 24GB+; 38 号盲测跑分建议 A100/4090 级) |
| 驱动 | NVIDIA Driver ≥ 535 (CUDA 12.1 兼容), `nvidia-container-toolkit` 已安装 |
| Docker | Docker ≥ 24, Docker Compose v2 (`docker compose version` 可用) |
| 磁盘 | ≥ 50GB (镜像 ~10GB + 权重 ~1GB; 若下载训练数据集另需 ~300GB) |
| Python (宿主机直跑时) | ≥ 3.10, 见根目录 `requirements.txt` |

验证 GPU 对容器可见:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 2. 获取代码与资产

```bash
git clone <repo-url> Ocean-DePTH-VLM && cd Ocean-DePTH-VLM

# 下载推理权重 (~300MB: YOLO best.pt / BioCLIP LoRA v2 / 质心 npz)
# 发布前请先在脚本顶部把占位 repo id 改成实际魔搭仓库 (TODO 注释处)
bash deploy/download_assets.sh

# 可选: 下载 287GB 训练数据集 (6 个 tar 分片, 解包到 data/processed/fathomnet_class_yolo)
bash deploy/download_assets.sh --with-dataset
```

下载落位 (与脚本内默认路径一致, 可用环境变量覆盖):

- `runs/detect/yolo11m_class34-2/weights/best.pt` — 34 纲 YOLO 检测权重
- `data/models/bioclip_lora/lora_adapter_v2/` — BioCLIP 2.5 LoRA 适配器
- `data/experiments/ft_20261102/centroids_v2.npz` — 2,638 形态种质心
- `data/processed/atlas_lookup/morphospecies_index.json` — 随仓库分发 (已含)

## 3. 启动知识图谱库 (Neo4j)

```bash
docker compose -f deploy/docker-compose.yml up -d neo4j
# 就绪检查: http://localhost:7474 (neo4j / password)
```

## 4. 导入知识图谱 (两条路任选其一)

### 路线 A — 从零重建 (可复现, 耗时数小时)

需要原始语料 (图集 PDF 解析产物、WoRMS 分类骨架等) 齐备:

```bash
docker compose -f deploy/docker-compose.yml --profile infer run --rm infer \
    python scripts/09c_rebuild_kg_bioclip25.py
# 如有图集/分类增量, 依次跑 09d/09e/09f (见脚本头部注释)
```

### 路线 B — 从 neo4j-dump 恢复 (分钟级, 推荐)

向发布者索取 `neo4j.dump` (图库物理备份, 含全部 Morphospecies/EntryImage/向量索引):

```bash
docker compose -f deploy/docker-compose.yml stop neo4j
docker run --rm -v depth_neo4j_data:/data -v "$PWD/deploy":/backup neo4j:5-community \
    cp /backup/neo4j.dump /data/neo4j.dump
docker run --rm -v depth_neo4j_data:/data neo4j:5-community \
    neo4j-admin database load neo4j --overwrite-destination=true --from-path=/data
docker compose -f deploy/docker-compose.yml up -d neo4j
```

> 两种路线都会得到可查询的 `entryimage_embeddings` 向量索引 (1024 维 cosine)。

## 5. 冒烟验证

```bash
docker compose -f deploy/docker-compose.yml --profile infer run --rm infer \
    python scripts/setup/verify_install.py
```

6 步全 PASS 即部署完成: 资产就位 → Neo4j 连接 → 质心加载 → YOLO 推理 →
BioCLIP 嵌入 → 检索 top-5。任一步 FAIL 会给出修复提示 (缺权重时提示重跑
`download_assets.sh`, 不会崩溃)。

## 6. 复现盲测基准 (1,744 盲测场景)

前置: 场景图目录 (FathomNet/Val 的 png+xml+txt) 已就位, 并挂载/指到
`DEPTH_VAL_DIR`; Neo4j 已导入且 **36 号已执行过盲测剔除**。

```bash
# 1) 构建盲测集 (seed=42 可复现; 从检索库物理移除盲测同源图, 输出 manifest)
docker compose -f deploy/docker-compose.yml --profile infer run --rm infer \
    python scripts/36_build_blindtest.py

# 2) 盲测跑分 (双路 YOLO → 嵌入 → 五种检索变体; 耗时随 GPU 数小时)
docker compose -f deploy/docker-compose.yml --profile infer run --rm infer \
    python scripts/38_blindtest_detect_retrieval.py

# 结果: /exp/blindtest/e2e_report.json (容器内) = 仓库根 exp/blindtest/
```

**目标指标 (cfusion 变体, 定版)**:

| species top-1 | species top-5 | species top-10 |
|---|---|---|
| **49.6** | **68.9** | **72.6** |

对照字段: `species_top1_cfusion` / `species_top5_cfusion` / `species_top10_cfusion`。
换权重/参数重跑时注意 38 号会命中 `/exp/blindtest/` 下的检测框与嵌入缓存
(签名不匹配会自动重建)。

## 7. 常见问题

- **Neo4j 连不上**: 推理容器内用服务名 `bolt://neo4j:7687`; 宿主机脚本用
  `NEO4J_URI=bolt://localhost:7687`。密码默认 `password`, 改密码时同步导出
  `NEO4J_PASSWORD` (11c 系列旧脚本读 `NEO4J_PASS`)。
- **显存不足**: 38 号双路推理峰值较高, 可先把 `paths` 改单路, 或减小 batch。
- **向量索引缺失**: 报 `no such index` 时, 在 Neo4j 里执行 36/09c 中
  `CREATE VECTOR INDEX entryimage_embeddings ...` 语句重建。
- **数据集 path 报错**: `config/yolo_finegrained.yaml` 的 `path:` 是相对占位,
  指向你的数据集实际位置 (默认 `./data/fathomnet_class_yolo`)。
