# 第一部分：YOLO 检测 — 数据集格式、自训与双路推理指南

> 目标读者：拿到本仓库代码、希望**用自己的数据复现或替换**检测环节的外部研究者。
> 所有命令、参数名、路径均与仓库内脚本逐条核对过（校验日期：2026-11-05，脚本版本见 git 初始 commit）。

---

## 1. 概述：YOLO 在整个 pipeline 中的角色

Ocean-DePTH-VLM 的完整识别链路是：

```
YOLO 双路检测 → 检出框 tight 裁剪 → BioCLIP 2.5 框级嵌入 → Neo4j 图谱多视图向量检索
              (+ 形态型质心 cfusion 融合重排) → 物种级检索结果
```

YOLO 在这里承担两个职责：

1. **目标提议（object proposal）**：在整幅 ROV 原位图里找出所有底栖生物框。检测层刻意"宁多勿漏"——用低置信度阈值（conf=0.05）多检出，误检交给后续检索层的拒识阈值（相似度 ~0.90）与 VLM 细判兜底；
2. **34 纲粗分类提示**：检出框附带 34 个纲级类别名（`name` 字段），经 `use_yolo_hint` 作为 GraphRAG 检索 soft_boost 与 VLM prompt 的先验参考（实测剔除 Other 纲后粗分类 acc 94.1%）。

**物种级身份不由 YOLO 给出**——34 纲（yolo34）只是粗类别，物种级结论由检索链路输出（morphospecies mid）。这是理解本系统的关键：即使你把 YOLO 换成只输出"生物/非生物"两类的模型，只要框召回够高，检索链路仍然能工作，只是失去粗分类提示的加权重排收益。

**最终使用的权重**：`yolo11m` 在 34 纲数据集上自训的 `best.pt`
（生产路径 `runs/detect/yolo11m_class34-2/weights/best.pt`，训练脚本 `scripts/05d_train_yolo_class.py`），
不是 Ultralytics 官方预训练权重。权重文件不入库（见 `.gitignore` 的 `*.pt`），需自行训练或从发布附件获取。

相关脚本一览：

| 脚本 | 作用 |
|---|---|
| `scripts/01_download_fathomnet.py` | FathomNet API 下载（CCZ 概念子集）→ COCO |
| `scripts/02_filter_benthicnet.py` | BenthicNet 元数据 CSV 筛选（深度 >3500 m） |
| `scripts/03_convert_to_yolo.py` | COCO → YOLO 格式（门级 10 类映射 + 分层划分） |
| `scripts/06d_build_class_yolo.py` | FathomNet 官方 2553 类 YOLO 数据 → 34 纲 YOLO 数据集（**生产数据集**） |
| `scripts/package_class_yolo.sh` | 34 纲数据集并行分片打包（~358 GB）供跨服务器转移 |
| `scripts/05_train_yolo.py` | 基线训练（yolo11s / 10 门类 / imgsz 640） |
| `scripts/05b_train_yolo_improved.py` | 改进训练（yolo11m / 10 门类 / 100 epochs） |
| `scripts/05d_train_yolo_class.py` | 34 纲训练（yolo11m / imgsz 960 / 120 epochs，**生产训练脚本**） |
| `scripts/utils/yolo_infer.py` | `DualPathDetector` 双路推理模块（生产推理实现） |
| `scripts/38_blindtest_detect_retrieval.py` | 盲测集端到端基准（含双路检测调用） |

---

## 2. 数据集格式

### 2.1 YOLO txt 标注格式

每张图对应一个同名 `.txt`，每行一个框：

```
<class_id> <cx> <cy> <w> <h>
```

- 所有数值为**相对图像宽高归一化到 [0,1]** 的值（不是像素）；
- `cx, cy` 是框中心点坐标，`w, h` 是框宽高；
- `class_id` 是从 0 开始的整数，对应 `data.yaml` 里的 `names` 顺序；
- 无标注的负样本图可以没有 txt 或 txt 为空（`06d` 对"全部框都是非生物"的图直接不入数据集）。

转换器 `scripts/utils/coco_to_yolo.py` 会把 COCO 的 `[x, y, w, h]`（左上角+宽高，像素）换算成上述中心点格式并裁剪到 [0,1]。

### 2.2 目录结构（推荐布局，与 05d / 配置文件一致）

```
data/processed/fathomnet_class_yolo/          # 数据集根目录（名字可自定）
├── data.yaml                                  # 数据集定义（训练时引用）
├── class_table.json                           # 每类框数统计（06d 产出，非必需）
├── images/
│   ├── train/                                 # 训练图（*.png / *.jpg）
│   │   ├── <uuid>.png
│   │   └── ...
│   └── val/
│       ├── <uuid>.png
│       └── ...
└── labels/
    ├── train/
    │   ├── <uuid>.txt                         # 与图片同名（仅扩展名不同）
    │   └── ...
    └── val/
        └── ...
```

`data.yaml`（`06d` 生成的实际内容格式）：

```yaml
path: .                # 数据集根目录；05d 训练时会自动改写为绝对路径
train: images/train    # 相对 path 的训练图目录
val: images/val        # 相对 path 的验证图目录
nc: 34                 # 类别数
names:                 # 类别 id → 类名
  0: Octocorallia
  1: Malacostraca
  ...
  33: Other
```

> **最少需要的文件**：`data.yaml` + `images/{train,val}/` + `labels/{train,val}/`，共三样。`class_table.json`、`classes.names`、`class_weights.json`（构建器附带产物）只是统计/参考文件，训练不依赖。

### 2.3 另一种布局（03 号脚本的输出）

`scripts/03_convert_to_yolo.py` 产出的是**扁平布局 + 划分清单文件**：

```
data/processed/fathomnet_yolo/
├── images/          # 全部图像（对原图做符号链接）
├── labels/          # 全部 txt（class_id 由 taxonomy_mapper 映射到 10 门类）
├── train.txt        # 每行一个图像文件名，分层划分 0.7/0.15/0.15（seed=42）
├── val.txt
└── test.txt
```

若使用这种布局，`data.yaml` 需写成 `train: images` + `train_list`（或自行拆分目录）。**建议直接采用 2.2 的 train/val 子目录布局**，与本仓库所有训练配置文件保持一致。

### 2.4 类别定义方式

- 类别表写在 `data.yaml` 的 `nc` + `names` 字段，训练与验证都以它为准；
- **生产 34 纲类别表**（`06d` 按 FathomNet 框数降序生成，`Other` 永远是最后一个 id）：

  ```
  Octocorallia, Malacostraca, Holothuroidea, Hexacorallia, Hexactinellida,
  Actinopterygii, Echinoidea, Asteroidea, Hydrozoa, Polychaeta, Demospongiae,
  Cephalopoda, Bivalvia, Scyphozoa, Crinoidea, Elasmobranchii, Gastropoda,
  Ophiuroidea, Thaliacea, Appendicularia, Tentaculata, Ascidiacea, Nuda,
  Thecostraca, Myxini, Rhynchonellata, Holocephali, Sagittoidea, Clitellata,
  Pycnogonida, Enteropneusta, Gymnolaemata, Polyplacophora, Other
  ```

  规模：154,701 图 / 682,894 框（train+val，来自 `class_table.json` 的 `_meta` 与逐类求和）。
- 34 纲的映射规则（`06d`）：FathomNet concept → WoRMS 纲（查 `fathomnet_candidates.json` 的 `taxonomy_worms.class` 链）；`Teleostei→Actinopterygii`、`Stenolaemata→Gymnolaemata` 两个纲名合并；框数 < 50 的纲并入 `Other`；器械/垃圾类（`pass=False` 或无纲）丢弃。
- **10 门类**（`03` + `05/05b`）：`utils/taxonomy_mapper.py` 内置 `APSMA_COARSE_CLASS_MAP`（Porifera/Echinodermata/Cnidaria/Annelida/Arthropoda/Xenophyophorea/Mollusca/Brachiopoda/Foraminifera 共 9 门 + Unknown = id 9），对应 `config/yolo_training.yaml` 的 `nc: 10`。

---

## 3. 用自己数据训练：从零步骤（不依赖 FathomNet 的最小路径）

假设你的数据是若干 ROV 图像 + 矩形框标注（任何来源都行，Pascal VOC / COCO / LabelMe 均可转成 2.1 的 YOLO txt）。

### 第 1 步：组织数据集

按 2.2 的目录布局摆放图像与 txt，写好自己的 `data.yaml`：

```yaml
path: /abs/path/to/my_dataset     # 建议直接写绝对路径，避免歧义
train: images/train
val: images/val
nc: 5                             # 你的类别数
names:
  0: sea_cucumber
  1: sea_star
  2: sponge
  3: fish
  4: other
```

> 注意：`nc` 和 `names` 的顺序就是 txt 里 `class_id` 的含义，训练前后不要重排。

### 第 2 步：选择训练脚本

| 目标 | 用哪个脚本 | 理由 |
|---|---|---|
| **复现生产行为（推荐）** | `05d_train_yolo_class.py` | imgsz=960 + AdamW + 长尾增强，与双路推理的 960 路对齐 |
| 快速基线 / 小数据 | `05_train_yolo.py` | yolo11s，默认 150 epochs |
| 想输出门级粗分类 | `05b_train_yolo_improved.py` | 训练完自动回写 `config/graphrag_inference.yaml` 的 `yolo_weights` |

### 第 3 步：训练命令（05d 为例，逐参数核对过）

```bash
python scripts/05d_train_yolo_class.py \
    --data-root data/processed/my_dataset_yolo \  # 含 data.yaml 的数据集根目录
    --model yolo11m.pt \                           # 初始权重（自动从 Ultralytics 拉取）
    --epochs 120 \                                 # 默认 120
    --imgsz 960 \                                  # 默认 960，与生产推理 960 路一致
    --batch -1 \                                   # -1 = 按显存自适应(60%)，A40 24G 单卡约可到 16
    --device 0 \                                   # 单卡；多卡写 "0,1"
    --workers 8 \
    --name my_yolo11m                              # 运行名，产物在 runs/detect/<name>/weights/best.pt
```

`05d` 的关键训练超参（硬编码在脚本里，与 argparse 默认值一致）：优化器 `AdamW`，`lr0=0.001`，`cos_lr`，`copy_paste=0.3`（稀有类关键增强），`mixup=0.1`，`mosaic=1.0`，`close_mosaic=15`（最后 15 轮关 mosaic），`hsv_h=0.015 / hsv_s=0.7 / hsv_v=0.4`，`degrees=10 / translate=0.1 / scale=0.5 / flipud=0.5 / fliplr=0.5`，`weight_decay=0.0005`，`warmup_epochs=3`，`patience=30`，`amp=True`，`cache=False`。
训练结束自动在 val 上按 **conf=0.025 / iou=0.4**（MBARI 生产口径，高召回）输出 mAP50 / mAP50-95 与 per-class 明细。

### 第 4 步：验证指标

```bash
# 05d 自带验证（训练完自动跑）；手动复跑：
python - <<'EOF'
from ultralytics import YOLO
m = YOLO("runs/detect/my_yolo11m/weights/best.pt")
metrics = m.val(data="data/processed/my_dataset_yolo/data.local.yaml",
                imgsz=960, device=0, split="val", conf=0.025, iou=0.4)
print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
print(f"P: {metrics.box.mp:.4f}  R: {metrics.box.mr:.4f}")
EOF
```

参考目标（脚本注释中的内部验收线）：34 纲模型 mAP50 ≥ 0.75、mAP50-95 ≥ 0.50、单类 AP50 ≥ 0.60 视为 PASS。
**但请注意**：本系统最终交付的是"检测召回"而非 mAP——检索链路宁多勿漏，验收请结合第 5 节的 recall@IoU0.5 口径（脚本 38 输出 `detection_recall@0.5`）。

---

## 4. 替换官方权重：路径在哪传、要改什么

### 4.1 推理时权重路径的两个入口

| 使用场景 | 位置 | 当前值 |
|---|---|---|
| GraphRAG 检索管线（`scripts/10_graphrag_pipeline.py`） | `config/graphrag_inference.yaml` → `yolo_weights` | `./runs/detect/yolo11m_class34-2/weights/best.pt` |
| 盲测基准（`scripts/38_blindtest_detect_retrieval.py`） | 脚本内常量 `YOLO_WEIGHTS`（第 33 行） | `"runs/detect/yolo11m_class34-2/weights/best.pt"` |

换权重最少改这两处（38 号脚本是常量而非读配置，需直接编辑）。另注意：

- `scripts/10_graphrag_pipeline.py` 的单路回退分支还有第三处默认值：`config.get("yolo_weights", "models/yolo_fathomnet.pt")`，仅在 `dual_path: false` 且 yaml 缺字段时生效；
- `scripts/11c2_dual_e2e.py` 走的是推理缓存合并，不直接加载权重，但脚本内有硬编码的 `YOLO_CLASSES` 34 类名列表，换类别体系时需同步。

### 4.2 类别数不一致时要改什么

1. **`data.yaml` 必须与权重训练时的 `nc/names` 完全一致**（`model.val()` 时）——`nc` 不匹配 ultralytics 会直接报错或给出错误结果；
2. `config/graphrag_inference.yaml` 里与类别相关的开关：
   - `use_yolo_hint: true/false` —— 34 纲自训模型可靠（剔 Other acc 94.1%），设 true；若你的模型类别可靠性未知，先设 false 观察纯检索基线；
   - `conf_filter: 0.05`、`iou_nms: 0.55` —— 与权重灵敏度匹配，见第 5 节；
   - `dual_path: true/false` 与 `dual_paths: [[960, true], [1920, false]]` —— 每项 `(imgsz, augment)`，可按你的训练 imgsz 调整；
3. 粗分类映射文件 `data/processed/taxonomy/yolo34_class_alias_map.json`（38 号脚本第 116 行加载）：内容为 *"你的类别名" → [等效类别名列表]*，用于把 GT 的纲别名对齐到模型输出类名。换类别体系时必须重写此文件；空列表 `[]` 表示该 GT 类在模型中无对应类、评测时排除。文件内 `_` 开头的键（`_description` / `_usage` / `_high_rank_fallbacks`）是元信息，加载时自动跳过。
   当前映射示例：`Teleostei → [Actinopterygii]`、`Anthozoa → [Octocorallia, Hexacorallia]`、`Crustacea → [Malacostraca, Thecostraca]`、`Solenogastres → []`。
4. 上游依赖粗类别名的下游模块（11c2 的 `YOLO_CLASSES`、`use_yolo_hint` 的纲→门映射）按新类别名同步。

### 4.3 权重产物路径约定（训练脚本自动落盘）

| 脚本 | 训练产物 | 额外复制 |
|---|---|---|
| 05 | `results/yolo_training/depth_vlm_yolo/weights/best.pt` | → `models/yolo_fathomnet.pt` |
| 05b | `runs/detect/results/yolo_training/depth_vlm_yolo-4-improved/weights/best.pt` | → `models/yolo_fathomnet_improved.pt`，并回写 `graphrag_inference.yaml` |
| 05d | `runs/detect/<name>/weights/best.pt`（默认 name=`yolo11m_class34`） | 无复制，生产直接引用 runs 路径 |

---

## 5. 双通道推理详解

实现唯一来源：`scripts/utils/yolo_infer.py`（`DualPathDetector` + `merge_detections`）。**以下是代码里的真实参数，非估计值。**

### 5.1 两路参数表

| 路 | imgsz | TTA（`augment`） | batch（自动） | 说明 |
|---|---|---|---|---|
| 路 1 | 960 | **True** | 64 | 与训练 imgsz 一致的主路；TTA 为 ultralytics 内置多尺度+翻转增广推理 |
| 路 2 | 1920 | False | 16（imgsz ≥ 1536 强制 ≤16 防 OOM） | 原生高分辨率路，对小目标敏感 |

- 调用方式（`10_graphrag_pipeline.py` / `38` 一致）：
  `DualPathDetector(weights, device=..., paths=((960, True), (1920, False)))` → `predict(imgs, conf=0.05, iou_nms=0.55)`
- `paths` 可经 `config/graphrag_inference.yaml` 的 `dual_paths` 覆盖（默认 `[[960, true], [1920, false]]`）；
- 设备：graphrag 配置 `yolo_device: cuda:0`；38 号脚本硬编码 `cuda:0`；`DualPathDetector` 构造默认 `cuda:1`（不传参时）。每轮 predict 后 `torch.cuda.empty_cache()`。

### 5.2 并集合并逻辑（`merge_detections`）

```
1. 收集两路全部原始检测框（各路都先不做 NMS 以外的过滤）
2. 按 conf >= conf_filter(0.05) 过滤
3. 全部候选按 conf 降序排序
4. 逐类贪心 NMS（iou_nms=0.55）：依次取框，仅当与已保留框"同类且 IoU > 0.55"时丢弃
   —— 不同类之间不互斥（海参与其上的虾都保留）
5. 输出按 conf 降序的合并检测列表
```

每图输出 dict：`{"box": [x1,y1,x2,y2](像素,取整), "conf": float(4位), "cls": int, "name": str}`。

### 5.3 conf=0.05 / nms=0.55 的取值依据

来自 2,848 图验证子集（真值 7,253 框）的消融（记录在 `utils/yolo_infer.py` 头注释）：

| 配置 | recall@IoU0.5 |
|---|---|
| 基线 imgsz=960, conf=0.25 | 78.7% |
| 单路 960+TTA, conf=0.01 | 88.3% |
| 单路 1920 原生, conf=0.01 | 83.1% |
| **双路并集, conf=0.05** | **90.8%** |
| 三路并集（+1280 切片）, conf=0.01 | 91.3%（推理侧饱和，95% 需重训） |

- **conf=0.05**：介于"极限召回 0.01"与"基线 0.25"之间——0.05 以下新增框对 recall 贡献边际递减，而误检交给检索层拒识（相似度 ~0.90）兜底；代码注释标注"定版 conf=0.05（用户决策）"。
- **iou_nms=0.55**：跨路重复框去重阈值。双路对同一目标的框位置略异，0.55 宽于常规 0.45~0.5，避免同一目标被两路各留一框造成下游重复嵌入；但同框跨类不互斥。
- 注意区分三组阈值：**训练验收** conf=0.025/iou=0.4（05d，MBARI 口径）、**生产检测层** conf=0.05/nms=0.55（yolo_infer/graphrag/38）、**11c2 评测匹配** IoU=0.5（GT 配准）。三者用途不同，勿混用。

### 5.4 换自训模型的影响与注意事项

1. **训练 imgsz 与推理 imgsz 不一致**：本项目训练 imgsz=960，1920 路属尺度分布外——实测单路 1920 反而比 960+TTA 差（83.1% vs 88.3%），但与 960+TTA 高度互补，并集净增益 +2.5pt。
   - 你的模型若以 imgsz=640 训练，建议双路改为 `((640, True), (1280, False))` 并在验证子集上重新扫 conf/nms；直接照搬 1920 路大概率有害。
   - 推理 imgsz ≤ 训练 imgsz 通常安全；大幅超训（如 640 训练 → 1920 推理）会放大小目标误检。
2. **TTA 对小目标的影响**：`augment=True` 的多尺度增广对深海常见的小型底栖生物（小框占比高）收益明显，代价是推理时间约 2~3 倍。若 CPU→GPU 带宽成为瓶颈，可只保留 960+TTA 单路（88.3%）。
3. **类别分布不均衡时的 conf 阈值**：深海长尾分布下，统一 conf 会系统性漏检稀有类（稀有类的 conf 分布整体偏低）。建议：
   - 从 conf=0.05 起步，按类统计 val 上的 recall/conf 曲线；
   - 若下游检索拒识可靠（误检成本低），宁可下调到 0.025~0.01；
   - 若你的类别含"Other/背景"类，考虑对 Other 单独提高 conf 或在合并后过滤——本项目 `Other` 纲平均 conf 明显低于有效纲；
   - NMS 0.55 主要服务"跨路去重"，若你只用单路可回到 0.45~0.5 常规值。
4. **检测框缓存签名**：38 号脚本用 `DET_SIG = {"weights", "conf", "iou_nms", "paths"}` 判断缓存（`det_cache.pkl`）是否失效。换权重/改参数后缓存自动重建；若想强制重建，删除 `/exp/blindtest/det_cache.pkl` 即可。

---

## 6. 数据集与测试集公开策略

FathomNet 与 BenthicNet 均为公开数据集。**本仓库不收录任何图片**（`.gitignore` 排除 `data/*` 与 `*.png` 类产物），只提供下载与转换脚本，由使用者自行获取原始数据：

### 6.1 数据获取

- **FathomNet**（https://fathomnet.org ，CC BY-BNC 2.0 类许可，商用需注意）：
  - 路线 A（`01` 号脚本）：FathomNet API 按 24 个 CCZ 相关概念各取 ≤1000 图 → `data/raw/fathomnet_raw/`（COCO json + 图像），并生成 20% 测试集 UUID 阻止列表 `data/test/fathomnet_test_uuids.txt`（seed=42，防泄漏）；
  - 路线 B（生产数据集所需）：从 FathomNet 官方发布下载**完整标注包**，按如下布局放置（`06d` 硬编码期望的结构，`--fathomnet-dir` 可指定根目录）：

    ```
    <FathomNet 根目录>/
    ├── Train/*.png + *.txt          # 官方 2553 类 YOLO 格式
    ├── Val/*.png + *.txt
    ├── class_mapping.json           # {concept: old_class_id}
    └── Annotations/fathomnet_{train,val}.json   # 官方 COCO
    ```
- **BenthicNet**：`02` 号脚本只做**元数据筛选**（输入 `benthicnet_labelled.csv`，按 `gebco_bathymetry` 绝对值 >3500 m 与 `catami_biota` 非空过滤），输出筛选清单 CSV；图像本体需按清单从 BenthicNet 发布渠道另行获取（脚本不下载图片）。

### 6.2 盲测集构建（`36_build_blindtest.py`）

目的：把抽中的 val 源图的**全部框**从 Neo4j 检索库物理删除，消除"查询命中库里同源图"的开卷效应（内部盲测规模：1,744 场景，seed=42 可复现，每 mid 最多抽 2 个源图，含"mid 清零退回"安全环）。

```bash
# 前置：Neo4j 已建成检索库（EntryImage 带嵌入），且挂载好：
#   /work/data/processed/fathomnet_expansion/boxes/<MID>/val_<uuid>_b<k>.jpg  (框切图)
#   /exp/meta.jsonl   (库清单: {"id","mid"})
#   /exp/emb_ft2.f32  (库嵌入矩阵, 1024 维 float32)
python scripts/36_build_blindtest.py            # 删库 + 备份 + 产出 /exp/blindtest/manifest.json
python scripts/36_build_blindtest.py --restore  # 一键恢复被删节点（评测完归还检索库）
```

随后跑 `38_blindtest_detect_retrieval.py` 完成检测+检索基准。**硬编码路径提示**：`36`/`38` 中 `bolt://depth_vlm_neo4j:7687`、`/work/...`、`/exp/...`、`VAL_DIR`（FathomNet Val 原图目录）等均为内部容器路径，复现时需按注释改成你的本地路径（详见第 7 节问题清单）。

---

## 7. 常见问题 FAQ

**Q1：训练/推理时 ultralytics 报 data.yaml 路径找不到？**
`config/yolo_training.yaml` 用相对路径 `./data/processed/apsma_aligned_yolo`，`config/yolo_training_subset.yaml` 写的是**构建机绝对路径**（指向源工程目录）。复现时把 `path:` 改成你本地数据集根目录即可；`05d` 会自动把 `data.yaml` 改写成 `data.local.yaml`（`path` 替换为运行时绝对路径），无需手工处理。

**Q2：`06d` 报找不到 `fathomnet_candidates.json`？**
该文件由源工程的 `expand_fathomnet.py`（concept 扩展 + WoRMS 分类回填）生成，未随本仓库发布。要么从源工程获取，要么自己写 concept→纲 的映射 JSON（结构：`{"concepts": {"<concept>": {"pass": bool, "taxonomy_worms": {"class": {"name": "<纲名>"}}}}}`）。

**Q3：`03_convert_to_yolo.py` 的输出和训练配置对不上？**
`03` 默认输出扁平布局（`images/`+`labels/`+`{train,val,test}.txt`），而 `config/yolo_training.yaml` 期望 `images/train` 子目录布局。用自己数据时推荐跳过 `03`，直接按第 3 节手工组织数据集。

**Q4：`02_filter_benthicnet.py` 下载了图片吗？**
没有。它只筛元数据 CSV；BenthicNet 图像需另行下载。FathomNet 的图像下载在 `01`（API 路线）里。

**Q5：换成自己的权重后，双路 1920 要不要保留？**
先在 300~500 张 val 图上做单路/双路消融（直接调 `DualPathDetector(paths=...)`）。经验法则：训练 imgsz ≥ 1280 才值得加 1920 路；否则用 `(训练imgsz, True)` 单路或 `(训练imgsz, True)+(2×imgsz, False)`。

**Q6：conf 调多低合适？**
取决于下游拒识成本。本系统检索层相似度拒识阈值 ~0.90，误检框会被自然过滤，所以检测层敢用 0.05。如果你的下游没有拒识（例如直接计数），请上调 conf 或加 VLM 校验。

**Q7：类别 id 顺序变了会怎样？**
权重里的分类头与 id 绑定。**只改 `data.yaml` 的 `names` 顺序而不重训 = 错误结果**。若只是重命名（id 不动），同步改 `data.yaml` 与 4.2 的 alias 映射即可。

**Q8：Neo4j 密码怎么注入？**
`config/graphrag_inference.yaml` 中为占位符 `neo4j_password: YOUR_NEO4J_PASSWORD`，替换为你的值或改用环境变量；`36`/`38` 脚本内 `auth=("neo4j", "password")` 为硬编码，需手动修改。

**Q9：双路推理显存不够？**
1920 路自动限 batch=16（`imgsz>=1536` 时），仍 OOM 则 `DualPathDetector.predict(..., batch=8)` 显式降 batch，或把 1920 路换成 1536/1280。

**Q10：为什么我的复现 recall 低于文档的 90.8%？**
该数字是 2,848 图验证子集、IoU 0.5、7,253 GT 框口径。请核对你的评测口径（IoU 阈值、GT 来源、conf 是否 0.05、TTA 是否开启、paths 是否 `((960,True),(1920,False))`）。
