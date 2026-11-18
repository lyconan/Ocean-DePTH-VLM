# 第三方接入教程：自建数据集制作微调数据 → QLoRA 微调 Qwen3-VL-32B → 接入 DePTH-VLM 系统

> 本教程面向要在自己服务器上，用**自己的深海生物图像数据集**微调 Qwen3-VL-32B
> 多模态模型，并接入 DePTH-VLM 识别系统"检索候选 → VLM 终判"环节的合作方。
> 全程不需要理解本系统内部的知识图谱实现，只需按约定格式准备图像和模型服务。

## 0. 总览

```
你的图像数据集 (每物种一个文件夹)
   ↓  scripts/39_build_sft_dataset.py          [§2-§3]
SFT 微调数据 (11图选择题: 1查询图 + 10参考图 → 选项字母)
   ↓  ms-swift QLoRA 微调 Qwen3-VL-32B-Instruct  [§4]
LoRA 适配器
   ↓  vLLM 部署为 OpenAI 兼容 API               [§5]
模型服务 http://<你的服务器>:8000/v1
   ↓  修改本系统 config/graphrag_inference.yaml  [§6]
系统推荐链路: YOLO检测 → BioCLIP检索top-10 → 你的VLM终判 → 推荐结果
```

**分工边界**：你负责数据、训练、模型服务；系统负责检测、检索、候选生成、结果融合与展示。

## 1. 环境准备

### 1.1 硬件要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| GPU | 1×46GB（A40/A6000，QLoRA 4bit） | 2×46GB（数据并行/更大 batch） |
| 显存说明 | 32B 4bit 权重 ≈ 20GB + LoRA/激活/视觉 token 开销 | 11 张图/样本，视觉 token 较多 |
| 磁盘 | 模型 ~65GB + 数据集 + 适配器 ~2GB | SSD/NVMe |
| 系统 | CUDA 12.x, Linux, Docker 可选 | — |

### 1.2 软件

```bash
# Python 3.10+
pip install "ms-swift" torchvision "transformers>=4.51" accelerate deepspeed
pip install qwen_vl_utils   # Qwen-VL 图像预处理

# 模型下载 (ModelScope 国内快)
modelscope download --model Qwen/Qwen3-VL-32B-Instruct --local_dir ./Qwen3-VL-32B-Instruct
# 或 HuggingFace: huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir ...
```

> 注：ms-swift 的 CLI 参数随版本略有变化（本教程基于 ms-swift ≥3.x 语法），
> 训练前请对照所用版本 `swift sft --help` 核对参数名。**所有训练参数均为推荐起点，
> 需按你的显存实测调整**。

## 2. 数据集规范（你的数据 → 规范布局）

### 2.1 目录布局（最简约定：一个物种一个文件夹）

```
dataset_root/
  Heterochone_calyx/        ← 文件夹名 = 物种 ID（建议拉丁名/唯一编码，下划线代空格）
    img_0001.jpg
    img_0002.png ...
  Farrea_occA/
    ...
```

### 2.2 可选谱系元数据：类文件夹下放 `_meta.json`

```json
{"latin": "Heterochone calyx", "genus": "Heterochone", "class": "Hexactinellida"}
```

- 不提供时默认 `latin=文件夹名`，`genus=名称首词`，`class=Unknown`
- 提供 `class` 的好处：39 号脚本会**优先用同纲物种做干扰项**（难例训练），显著提升细粒度判别

### 2.3 数量与质量要求（重要）

| 要求 | 说明 |
|---|---|
| 类别数 | ≥10 类才能构造 10 选项选择题；建议 ≥30 类 |
| 每类图像 | **建议 ≥20 张**（其中留 val）；上限 --cap-per-class 默认 100/类（防大类主导） |
| 每类 ≥2 张 | 参考图与查询图必须不同图，单图类无法出题 |
| 图像形态 | **物种级裁剪图**（框内单个体为主），与推理时输入一致——不要放整景大图 |
| 质量 | 分辨率 ≥224px；避免严重模糊/器械遮挡为主体的图 |
| **标签可信** | ⚠ 本系统曾因第三方数据集（JEDI）机器挂接标签不可信而全量下架——**入库前请人工抽查每类图像是否确为该物种**，宁缺毋滥 |

## 3. 制作 SFT 微调数据

**两条路线任选**：
- **路线 A（第三方自建数据）**：按本节用 `scripts/39_build_sft_dataset.py` 从你自己的图像集出题
- **路线 B（使用系统现成数据）**：系统已备好净化后的 96,141 条 SFT 数据
  （JEDI 过滤后：species 11 图选择题 59,407 + name 开放命名 27,509 + class/genus 选择题 9,225，
  存于 `data/processed/sft/`），用 `scripts/39b_sft_to_swift.py` 一键转成训练格式：

```bash
python3 scripts/39b_sft_to_swift.py \
    --src data/processed/sft --out data/processed/sft_swift
# 输出 sft_train_swift.jsonl (67,939) / sft_name_train_swift.jsonl (27,509) /
#      sft_val_swift.jsonl (693), §4 训练时 --dataset 指向 *_swift.jsonl 即可
```

路线 B 无需自备图像，适合先跑通全流程再换自有数据。其余步骤两路线完全一致。

### 3.1 一条命令（路线 A）

```bash
python3 scripts/39_build_sft_dataset.py \
    --src /path/to/dataset_root \
    --out ./sft_out \
    --val-ratio 0.1 --cap-per-class 100 --variants 1 --seed 42
```

### 3.2 输出物

| 文件 | 用途 |
|---|---|
| `sft_train.jsonl` / `sft_val.jsonl` | 本系统协议格式（供检查与复现） |
| `swift_train.jsonl` / `swift_val.jsonl` | **ms-swift 直读格式，§4 训练用这个** |
| `dataset_stats.json` | 每类样本数/划分摘要，先检查它 |

### 3.3 数据格式（本系统协议，每条 JSON）

```json
{"task": "species", "variant": "v1",
 "query_path": "dataset_root/Heterochone_calyx/img_0001.jpg",
 "mid": "Heterochone_calyx",
 "refs": [".../ref1.jpg", ".../ref2.jpg", "... 共10张"],
 "options": [{"letter": "A", "mid": "...", "latin": "...", "genus": "...", "class": "...", "ref": "..."}, ...],
 "answer": "C",
 "cand_ranks": null}
```

语义：第 1 张图是查询裁剪图；10 个选项 A-J 各配 1 张该物种参考图（`refs` 与 `options[].ref` 同序）；模型只需输出答案字母。

### 3.4 内置的防坑设计（不必操作，但验收时检查）

- **GT 位置随机**：答案字母分布均匀（用 `python3 -c "import json,collections;print(collections.Counter(json.loads(l)['answer'] for l in open('sft_out/sft_train.jsonl')))"` 核查）——防止模型学会"偏答某字母"的位置偏置（本系统实测教训：位置偏置曾虚增 14pp）
- **split 隔离**：val 题目的干扰项和参考图只取自 val 图，train/val 零图像重叠
- **参考图 ≠ 查询图**：GT 选项的参考图必为同类的另一张图（真实"同种异图"判别）
- **干扰项梯度**：同纲物种优先做干扰项（难例），不足时随机补足

## 4. QLoRA 微调 Qwen3-VL-32B

### 4.1 训练命令（ms-swift，QLoRA 4bit 起点）

```bash
# 单卡 46GB 起点；两张卡加 --nproc_per_node 2
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model ./Qwen3-VL-32B-Instruct \
    --dataset ./sft_out/swift_train.jsonl \
    --val_dataset ./sft_out/swift_val.jsonl \
    --train_type lora \
    --quant_method bnb --quant_bits 4 \
    --lora_rank 32 --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --max_length 8192 \
    --max_pixels 448x448_per_image \
    --save_steps 200 --save_total_limit 3 \
    --output_dir ./qwen3vl32b_sft_out
```

参数说明：
- `--max_pixels`：每图像素预算。训练与推理务必一致（系统推理端用 448px 裁剪图）
- 数据量参考：本系统协议数据（净版 96,141 条）训练 2 epoch；小数据集（<5k 条）可加到 3 epoch 并把 lr 降到 5e-5
- 若 OOM：降 `--max_pixels` 或确认 4bit 生效；仍紧张换 `--deepspeed zero2`

### 4.2 训练监控

- 观察 `train_loss` 收敛与 `eval` 输出：**验证集答案应随机分布且非全 A**（全答 A = 位置偏置未除尽/欠拟合，检查数据）
- 训练完成产物：`<output_dir>/xxx-xxx/checkpoint-*/` 内含 LoRA 适配器

## 5. 部署模型服务（OpenAI 兼容 API）

### 5.1 vLLM + LoRA 热加载（推荐，无需合并权重）

```bash
vllm serve ./Qwen3-VL-32B-Instruct \
    --served-model-name depth-vlm-endpoint \
    --max-model-len 16384 \
    --limit-mm-per-prompt image=11 \
    --lora-modules depth-vlm-lora=./qwen3vl32b_sft_out/checkpoint-XXX \
    --port 8000
```

- 显存不够可加 `--quantization awq`（需先 `swift export --merge_lora false --to awq` 量化，二选一）
- 也可先合并再裸部署：`swift export --adapters <ckpt> --merge_lora true` 得到完整权重后 `vllm serve`（多卡 tensor parallel `--tensor-parallel-size 2`）

### 5.2 冒烟测试（curl）

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "depth-vlm-endpoint",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "file:///path/query.jpg"}},
    {"type": "image_url", "image_url": {"url": "file:///path/refA.jpg"}},
    ...共11张...,
    {"type": "text", "text": "第1张图为待鉴定...只输出一个选项字母。A. xxx B. xxx ..."}
  ]}],
  "temperature": 0, "max_tokens": 8
}'
# 期望回复形如 {"choices":[{"message":{"content":"C"}}]}
```

**服务契约（系统按此调用，务必满足）**：
1. OpenAI 兼容 `/v1/chat/completions`，多图（1 查询 + 10 参考）按顺序排列
2. `temperature=0` 下输出为**单个选项字母**（可含空白，系统取首字母）
3. 模型名、端口任意，接入时告知即可

## 6. 接入 DePTH-VLM 系统

系统侧"检索→终判"链路（[config/graphrag_inference.yaml](../config/graphrag_inference.yaml)）：

```
场景图 → YOLO 34纲检测 → 检出框裁剪
      → BioCLIP 检索 top-10 物种候选 (mid) + 各候选 medoid 参考图
      → 构造 11 图选择题 (与 §3.3 格式同构)
      → 调用 vlm_api_url 指向的 OpenAI 兼容接口
      → 解析答案字母 → 对应 mid → 排序输出推荐结果 (附知识图谱证据)
```

只需修改配置三行，把终判指向你的服务：

```yaml
# config/graphrag_inference.yaml
vlm_api_url: http://<你的服务器IP>:8000/v1
vlm_model: depth-vlm-endpoint        # 与 vllm --served-model-name 一致
vlm_api_key: EMPTY
vlm_temperature: 0.1                 # 系统默认; 终判解析只取首字母
```

接入后建议先跑系统自带的盲测回归（38 号脚本）确认端到端指标无回退，再切生产。

## 7. 评测与验收标准（本系统实测经验值）

| 指标 | 参考值（本系统, n=5,184 盲测检索对象） |
|---|---|
| 检索 top-1 / top-5 / top-10 | 49.8% / 69.6% / 73.2% |
| VLM 未微调原序选择题最优 | 30.2%（乱序后 16.1%，位置偏置虚高） |
| 端到端天花板 | ≤ 检索 top-10 = 73.2%（GT 不在池内必错） |

**验收建议**（前车之鉴，务必照做）：
1. 用**乱序复测**（同一验证集，候选顺序打乱两次）测你的模型：两次成绩差应 <2pp，否则存在位置偏置
2. 微调后 VLM 终判目标：乱序 top-1 ≥ 检索 rank-1（49.8%），或"VLM 翻案 + 检索回退"混合策略端到端 ≥ rank-1 + 5pp
3. 混合策略（推荐实现）：仅对检索分低/候选分接近的样本用 VLM 翻案，其余直接采信检索 rank-1——VLM 与检索错误存在互补（本系统 Oracle 并集 54.1% vs 单用检索 45.2%），但裸融合会被 VLM 噪声拖累（本系统 pairwise 融合实测更差）

## 8. 常见问题

| 症状 | 原因/处理 |
|---|---|
| `类别数不足` | 39 号脚本需 ≥10 类；合并相近类或补数据 |
| val 记录数远少于 val 图数 | 每类 val 图 <2 张无法出题；每类 ≥5 张时才留 val，建议每类 ≥20 张 |
| 训练 OOM | 确认 4bit 生效 → 降 max_pixels → deepspeed zero2 → 双卡 |
| 训练后全答 A | 数据答案分布不均/位置偏置；用 39 号脚本重建（GT 位置随机）并重训 |
| 评测虚高 | 验证集图像泄漏进训练；39 号脚本已按图隔离，勿自行合并 split |
| 系统侧输出错乱 | 服务未满足 §5.2 契约（单字母输出/多图顺序）；先跑 curl 冒烟 |
| 检索候选池外物种 | 终判只能在检索 top-10 内选择；池召回是端到端天花板，需先提升检索侧 |
