# 更新记录

## 2026-11-19 新增：可视化控制台（webapp/）

新增可交付的可视化系统，第三方部署方可在自己服务器上直接展示整个知识系统：

- **五个页面**：总览（图谱统计 + 管线流程）/ 知识库（2,641 条目详情 + 知识描述 + 证据图集）/ 检索库（30,629 张证据图网格 + 来源筛选）/ 管线回放（盲测逐对象回放：查询图 + 候选对照 + 原图框标注叠加）/ VLM 终判（接口预留状态 + 在线试跑）
- **纯 Python 标准库实现**，零第三方依赖，`python3 webapp/console_api.py` 即启动
- **环境变量可配置**：`DEPTH_NEO4J_HTTP` / `DEPTH_NEO4J_AUTH` / 盲测数据路径，接入自有 Neo4j
- **优雅降级**：盲测数据缺失时管线回放提示不可用，知识库/检索库不受影响
- **VLM 终判接口预留**：`config/graphrag_inference.yaml` 填入 `vlm_api_url` / `vlm_model` 即自动启用
- 部署步骤见 [webapp/README.md](../webapp/README.md)

## 2026-11-18（对比 11-10 发布版）

本次更新涉及**检索库整体换血**：剔除了标签不可信的 JEDI 数据源，并新增完整的第三方微调套件。
已有部署方需重新拉取魔搭资产（见 §3），代码侧执行 `git pull` 即可。

### 1. JEDI 数据下架 —— 检索库 34,026 → 30,629 条证据

- **原因**：JODD 官方仅提供 20 个粗类别，此前管线机器挂接的精细标签未经人工校准，
  实测为"标签污染"——下架后盲测三项指标反而全部微升
- **影响范围**：图谱 EntryImage（JEDI 源 3,397 张全部移除）、嵌入向量库、morphospecies 索引
- **盲测全量复跑**（n=5,184 检索对象，1,744 场景）：

  | 指标 | 旧版（含 JEDI） | 新版（净库） |
  |---|---|---|
  | YOLO 检测 recall@0.5 | 88.1% | 88.1%（不变，训练集不含 JEDI） |
  | 34 纲粗类别准确率 | 95.7% | 95.7%（不变） |
  | 物种检索 top-1 / top-5 / top-10（cfusion） | 49.6 / 68.9 / 72.6% | **49.8 / 69.6 / 73.2%** |

### 2. 魔搭资产同步更新（部署方必读）

`bash deploy/download_assets.sh` 会自动拉到新版，**无需改任何脚本**：

| 资产 | 变化 |
|---|---|
| `neo4j.dump` | **479MB → 495MB 新版**（30,629 EntryImage，无 JEDI，含重建后的向量索引） |
| `centroids_v2.npz` | **已替换**（按新库重建的 2,638 形态种质心，条目数不变） |
| YOLO 权重 / BioCLIP LoRA v2 / taxonomy / atlas_lookup | 无变化 |

### 3. 第三方微调套件（全新）

- `scripts/39_build_sft_dataset.py`：自建数据集（每物种一文件夹）→ 11 图选择题 SFT 数据
  生成器。内置防坑设计：GT 位置随机防位置偏置、train/val 图像级隔离防泄漏、
  同纲物种优先做难例干扰项、直出 ms-swift 训练格式
- `scripts/39b_sft_to_swift.py`：系统现成 96k 净化 SFT 数据一键转 ms-swift 格式（路线 B）
- `docs/第三方接入教程_自建数据集微调Qwen3-VL-32B.md`：**全流程教程**——
  数据规范 → SFT 制作 → ms-swift QLoRA 4bit 微调 Qwen3-VL-32B → vLLM LoRA 部署
  → 改 `vlm_api_url` 两行接入系统 → 乱序评测验收标准（含位置偏置教训）→ FAQ
- **训练工具与配置（§4.0）**：默认 ms-swift（Qwen3-VL 官方生态，一键脚本
  `deploy/finetune/train_ms_swift.sh`），LLaMA-Factory 作为替代方案提供完整配置模板
  `deploy/finetune/llamafactory_qwen3vl_lora.yaml`；依赖清单
  `deploy/finetune/requirements_finetune.txt` 两方案共用
- `docs/VLM终判问题汇总_供Gemini分析.md`：7 模型 × 3 协议终判实验全景（背景资料）

### 4. 工程加固

- `scripts/38`（端到端基准）：嵌入/检测缓存每 5 窗口增量落盘（长跑崩溃不清零）、
  嵌入失败自动降级 + 逐图隔离毒图（POISON 集不计入指标）

### 升级步骤（已有部署）

```bash
git pull
bash deploy/download_assets.sh        # 自动拉新版 neo4j.dump / centroids
# 按 deploy/README.md §4 路线 B 重新导入图谱
```
