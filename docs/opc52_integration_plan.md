# opc-52 + Oceanus-all 工作继承与整合方案

> 制定日期: 2026-07-17 | 更新: 2026-07-17 (新增APSMA门级扩展、WoRMS图片库、Oceanus-all整合)
> 项目: DePTH-VLM (西太平洋深海生物多样性)

---

## 一、opc-52 目录现状分析

### 1.1 目录结构概览

```
/mnt/oneocean_200TBnas_feitian/opc-52/
├── WoRMS_download_2025-07-01/   (534MB, 旧版WoRMS DwC-A归档, 无CSV)
├── WoRMS_download_2025-08-01/   (534MB, 同上)
├── WoRMS_download_2025-10-01/   (1.1GB, 最新版, 含CSV)
│   ├── taxon_node.csv           (512MB, 589,562条分类节点)
│   ├── taxonrelationship.csv    (56MB, 589,562条分类关系)
│   ├── taxon.txt                (509MB, DwC-A原始格式)
│   ├── identifier.txt           (10MB)
│   ├── speciesprofile.txt       (30MB)
│   ├── eml.xml                  (11KB)
│   └── meta.xml                 (4KB)
├── obis_20250318_tsv.zip        (18GB, OBIS出现记录)
├── python/
│   └── llm_api.py               (豆包API分类脚本: 深海生物/地质/采矿)
├── Web of Science/              (33,847个PDF论文)
├── 中文文献/                    (648个文件)
│   ├── PDF、HTML、TXT混合
│   ├── 含深海生物、底栖、西太平洋相关文献约17个
│   ├── 含深海采矿、地质等非生物文献约630个
│   └── 含书籍: 10000个科学难题·海洋科学卷 等
```

### 1.2 关键数据资产

| 数据 | 类型 | 大小 | 条数 | 与本项目关系 |
|------|------|------|------|-------------|
| taxon_node.csv | WoRMS分类节点 | 512MB | 589,562 | 🥇最高 — 直接可导入Neo4j构建分类树 |
| taxonrelationship.csv | WoRMS分类关系 | 56MB | 589,562 | 🥇最高 — 与taxon_node配套,构建层级关系 |
| speciesprofile.txt | WoRMS物种属性 | 30MB | ~30,000 | 🥈高 — 物种深度/分布等生态属性 |
| identifier.txt | WoRMS标识符映射 | 10MB | ~10,000 | 🥉中 — 辅助映射 |
| obis_20250318_tsv.zip | OBIS出现记录 | 18GB | 未知(估计百万级) | 🥈高 — 西太平洋出现记录需筛选 |
| llm_api.py | 分类脚本 | 1KB | - | 🥉低 — 分类逻辑可参考 |

---

## 二、APSMA 门级覆盖分析与扩展

### 2.1 WoRMS 中 APSMA 15门覆盖情况

WoRMS 共 144 个门、589,205 条 accepted 记录。APSMA 15门覆盖 525,204 条(89%)，
剩余 64,001 条属于 129 个非 APSMA 门。

**APSMA 15门在 WoRMS 中的分布:**

| 门级 | 条数 | 深海相关性 |
|------|------|-----------|
| Mollusca | 169,087 | 🥇 深海腹足/双壳/头足类丰富 |
| Arthropoda | 113,354 | 🥇 深海甲壳类(端足/等足/十足)丰富 |
| Chordata | 76,201 | 🟡 深海鱼类为主,少量深海海鞘 |
| Cnidaria | 38,735 | 🥇 深海珊瑚/海葵/水螅丰富 |
| Annelida | 31,404 | 🥇 深海多毛类丰富 |
| Porifera | 24,355 | 🥇 深海海绵(六放/寻常)丰富 |
| Echinodermata | 22,626 | 🥇 深海海参/蛇尾/海星丰富 |
| Foraminifera | 19,933 | 🥇 深海有孔虫丰富 |
| Nematoda | 12,993 | 🥈 深海线虫丰富但鉴定困难 |
| Bryozoa | 11,448 | 🟡 深海苔藓虫较少 |
| Nemertea | 3,055 | 🟡 深海纽虫存在 |
| Brachiopoda | 1,179 | 🟡 深海腕足类少量 |
| Ctenophora | 599 | 🟡 深海栉水母少量 |
| Hemichordata | 235 | 🟡 深海半索动物极少 |
| Xenophyophorea | 0 | ❌ WoRMS无此门(异生泡虫,CCZ特有) |

### 2.2 西太平洋深海遗漏门级分析

以下门级在 APSMA 15门中**不存在**，但在西太平洋深海有代表性物种：

| 遗漏门级 | WoRMS条数 | 西太深海代表性 | 是否加入 |
|---------|----------|--------------|---------|
| **Platyhelminthes** (扁形动物门) | 27,833 | 深海涡虫/吸虫/绦虫 | ✅ 加入 — 深海涡虫是深渊区重要类群 |
| **Sipuncula** (星虫动物门) | 1 | 深海星虫 | ✅ 加入 — 中国动物志第46卷含星虫 |
| **Echiura** (螠虫动物门) | 1 | 深海螠虫 | ✅ 加入 — 与星虫同卷 |
| **Priapulida** (曳鳃动物门) | 61 | 深海曳鳃动物 | ✅ 加入 — 深海小型底栖重要类群 |
| **Kinorhyncha** (动吻动物门) | 673 | 深海动吻虫 | ✅ 加入 — 深海小型底栖重要类群 |
| **Loricifera** (铠甲动物门) | 69 | 深海铠甲虫 | ✅ 加入 — 极深海(深渊区)特有 |
| **Tardigrada** (缓步动物门) | 414 | 深海水熊虫 | 🟡 可选 — 少量深海种 |
| **Chaetognatha** (毛颚动物门) | 361 | 深海箭虫 | ✅ 加入 — 中国动物志第38卷含箭虫 |
| **Phoronida** (帚虫动物门) | 61 | 深海帚虫 | 🟡 可选 — 极少量深海种 |
| **Entoprocta** (内肛动物门) | 292 | 深海内肛虫 | 🟡 可选 — 少量深海种 |
| **Gastrotricha** (腹毛动物门) | 758 | 深海腹毛虫 | 🟡 可选 — 小型底栖 |
| **Xenacoelomorpha** (异无腔动物门) | 764 | 深海异涡虫 | 🟡 可选 — 深海无腔动物 |

**最终扩展为 21 门** (15 + 6 新增):
- 新增: Platyhelminthes, Sipuncula(+Echiura), Priapulida, Kinorhyncha, Loricifera, Chaetognatha

---

## 三、Oceanus-all 目录分析

### 3.1 目录结构

```
/mnt/oneocean_200TBnas_feitian/Oceanus-all/oceanus_dataset/
├── 生物分类书籍格式转化路径/
│   ├── pdf-md/
│   │   ├── WoRMS_papers/              # 🥇 7,564个物种分类学论文(PDF→MD已转换)
│   │   │   ├── {物种名}_{DOI}/
│   │   │   │   ├── auto/
│   │   │   │   │   ├── {物种名}.md    # MinerU转换的markdown描述
│   │   │   │   │   ├── images/        # 从PDF提取的物种图片(.jpg)
│   │   │   │   │   ├── _content_list.json
│   │   │   │   │   ├── _layout.pdf
│   │   │   │   │   └── _origin.pdf   # 原始PDF
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── WoRMS_markdown提取/        # 🥇 7,567个纯MD文件(与papers对应)
│   │   └── 生物分类书籍/              # 🥈 中国动物志等30本分类学书籍(PDF→MD)
│   │       ├── 中国动物志 第40卷 棘皮动物门 蛇尾纲/auto/images/ (241张图)
│   │       ├── 中国动物志 环节动物门 多毛纲/... (183张图)
│   │       ├── 中国动物志 节肢动物门 甲壳动物亚门/... (多卷,合计1000+张图)
│   │       ├── 中国动物志 软体动物门/... (多卷,合计1000+张图)
│   │       └── ... (共38本, 约7,000+张图)
│   └── 生物分类书籍/                  # 原始PDF文件
├── 知识图谱数据/                       # 🥉 3个知识图谱参考书籍(41MB)
│   ├── 知识图谱：方法、实践与应用.pdf
│   ├── 从零构建知识图谱 技术、方法与案例.pdf
│   └── 知识图谱_白文杰_组会2025年6月10日.pptx
├── Biology/                           # 🥈 arXiv生物学论文(MD格式,数百个)
├── multi_gpu/                         # 🥈 MinerU多GPU处理结果
│   └── folder_222/                    # 含深海生物相关PDF+MD
│       ├── Deep-Sea Biology Symposium 2018 Abstracts/
│       ├── Benthic Ecology and Demersal Resources/
│       ├── Biological hotspots in the deep sea/
│       └── ... (约20个深海相关文献)
├── MinerU/                            # MinerU工具配置+处理结果
├── Geology/Policy/Others/Resource/    # 非生物相关,不取
└── md_extractor_categ.py             # 文献分类脚本
```

### 3.2 核心数据资产

| 数据 | 条数 | 图片数 | 与本项目关系 |
|------|------|--------|-------------|
| **WoRMS_papers** (物种分类论文+MD+图片) | 7,564 | ~5,000+ | 🥇 物种描述知识库+图片检索库 |
| **WoRMS_markdown提取** (纯MD) | 7,567 | - | 🥇 物种描述知识库(纯文本) |
| **中国动物志** (38卷) | 38 | 7,000+ | 🥈 中国海域物种描述+图片 |
| **multi_gpu/folder_222** (深海文献) | ~20 | - | 🥈 深海生物会议/综述 |

---

## 四、综合整合方案

### 4.1 核心策略: "取最新、筛西太、扩门级、建图库"

1. **取最新**: 只取 WoRMS_download_2025-10-01,07/08两个旧版不取
2. **筛西太**: 从 WoRMS 58万条全球分类节点中,筛选出扩展后21门的子集
3. **扩门级**: APSMA 15门 → 扩展21门(增加扁形/星虫/曳鳃/动吻/铠甲/毛颚)
4. **建图库**: 构建"物种图片检索库"(每物种5张图) + "物种描述知识库"

### 4.2 具体任务清单

#### 任务A: WoRMS 分类数据整合 (✅已完成)

**A1-A4: 已完成** — 245,788条APSMA 15门分类节点+关系+属性已筛选
- 需更新: 重新运行筛选以包含新增6门 → 预计增加~29,000条

#### 任务E: WoRMS 物种图片检索库构建 (高优先级,新增)

**E1: 从 WoRMS API 批量获取物种图片URL**
- 对 APSMA 21门中 rank=Species 的记录,访问 WoRMS 详情页提取图片URL
- 图片来源: `https://images.marinespecies.org/thumbs/{id}_{name}.jpg`
- 去掉 `?w=187` 获得更大尺寸
- 每物种目标: 5张图片
- 预计: ~50,000个物种 × 平均2张/种 = ~100,000张图片URL

**E2: 下载物种图片**
- 多线程下载,保存到 `data/processed/species_image_library/{phylum}/{species}/`
- 每物种最多5张,按质量(分辨率)优先选择
- 预计磁盘: ~100,000 × 200KB = ~20GB

**E3: 从 Oceanus-all WoRMS_papers 提取已有图片**
- 7,564个 WoRMS 论文中已有从PDF提取的图片(~5,000+张)
- 将图片按物种名映射到检索库: `data/processed/species_image_library/`
- 这些图片比WoRMS网站缩略图质量更高

**E4: 从中国动物志提取深海相关图片**
- 筛选与 APSMA 门级相关的卷:
  - 第40卷 棘皮动物门 蛇尾纲 (241张)
  - 环节动物门 多毛纲 (183张)
  - 节肢动物门 甲壳动物亚门 (多卷,1000+张)
  - 软体动物门 (多卷,1000+张)
  - 第38卷 毛颚动物门 箭虫纲 (103张)
  - 第46卷 星虫动物门 螠虫动物门 (117张)
- 合计约 3,000+ 张高质量物种图片

**E5: 合并构建图片检索索引**
- 合并 E2+E3+E4 的图片,按物种名去重
- 生成 `data/processed/species_image_library/index.json`
  - key: species_name (或 aphia_id)
  - value: {phylum, class, order, family, images: [{path, source, width, height}]}
- 统计每物种图片数,不足5张的标记需补充

#### 任务F: WoRMS 物种描述知识库构建 (高优先级,新增)

**F1: 从 Oceanus-all WoRMS_markdown提取 整合物种描述**
- 7,567个 MD 文件,每个包含一个物种的分类学描述
- 筛选属于 APSMA 21门的物种
- 按 AphiaID 或物种名与 taxon_subset 对齐
- 复制到 `data/processed/species_knowledge_base/descriptions/`

**F2: 从中国动物志 MD 整合描述**
- 筛选深海相关卷的MD文件
- 复制到 `data/processed/species_knowledge_base/fauna_sinica/`

**F3: 构建 WoRMS 网页描述**
- 对有图片的物种,从 WoRMS 详情页提取 citation/authority 等文本描述
- 补充 F1 中没有的物种

**F4: 生成知识库索引**
- 合并 F1+F2+F3,生成 `data/processed/species_knowledge_base/index.json`
  - key: species_name
  - value: {aphia_id, phylum, class, ..., descriptions: [{source, path, text_snippet}], images_count}

#### 任务G: Oceanus-all 其他资料整合 (中优先级)

**G1: 复制中国动物志深海相关卷(图片+MD)**
- 7个深海相关卷 → `data/raw/inherited_literature/中国动物志/`

**G2: 复制深海生物会议/综述文献**
- multi_gpu/folder_222 中深海生物相关 → `data/raw/inherited_literature/深海文献/`

**G3: 复制知识图谱参考书籍**
- 3个知识图谱PDF → `data/raw/inherited_literature/知识图谱参考/`

---

## 五、数据流向图 (更新)

```
opc-52/WoRMS_download_2025-10-01/
  ├── taxon_node.csv ──→ [A1筛选21门] ──→ apsma_worms_taxon_subset.csv
  ├── taxonrelationship.csv ──→ [A2] ──→ apsma_worms_relationship_subset.csv
  ├── speciesprofile.txt ──→ [A3] ──→ apsma_worms_species_profile.csv
  └─ 三者合并 ──→ [A4] ──→ apsma_worms_enhanced.jsonl

WoRMS API (images.marinespecies.org) ──→ [E1获取URL] ──→ [E2下载] ──→ species_image_library/

Oceanus-all/WoRMS_papers/ ──→ [E3提取图片] ──→ species_image_library/
Oceanus-all/生物分类书籍/ ──→ [E4提取图片] ──→ species_image_library/
                                    ↓
                              [E5合并索引] ──→ species_image_library/index.json

Oceanus-all/WoRMS_markdown提取/ ──→ [F1整合描述] ──→ species_knowledge_base/descriptions/
Oceanus-all/生物分类书籍/ ──→ [F2整合描述] ──→ species_knowledge_base/fauna_sinica/
WoRMS API 详情页 ──→ [F3网页描述] ──→ species_knowledge_base/
                                    ↓
                              [F4合并索引] ──→ species_knowledge_base/index.json
```

---

## 六、输出目录规划 (更新)

```
data/
├── raw/
│   ├── worms_inherited/                # ✅已存在 — 从opc-52继承的WoRMS数据
│   ├── inherited_literature/           # ✅已存在 — 继承的文献
│   │   ├── 西太平洋深海生物/            # ✅ 16个文献
│   │   ├── 中国动物志/                  # 🆕 深海相关卷(图片+MD)
│   │   ├── 深海文献/                    # 🆕 深海生物会议/综述
│   │   ├── 知识图谱参考/                # 🆕 3个知识图谱PDF
│   │   └── reference_scripts/          # ✅ 分类脚本
│   └── obis_inherited/                 # OBIS出现记录(待处理)
├── processed/
│   ├── species_image_library/          # 🆕 物种图片检索库
│   │   ├── index.json                  # 检索索引
│   │   ├── Porifera/                   # 按门级组织
│   │   │   ├── Haliclona_permollis/    # 每物种一个文件夹
│   │   │   │   ├── img_001.jpg         # 最多5张图片
│   │   │   │   ├── img_002.jpg
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── Echinodermata/
│   │   └── ... (21门)
│   └── species_knowledge_base/         # 🆕 物种描述知识库
│       ├── index.json                  # 检索索引
│       ├── descriptions/               # WoRMS论文MD描述
│       │   ├── Holothuria_scabra.md
│       │   └── ...
│       ├── fauna_sinica/               # 中国动物志MD描述
│       │   ├── 棘皮动物门_蛇尾纲.md
│       │   └── ...
│       └── worms_web/                  # WoRMS网页描述(JSON)
```

---

## 七、WoRMS 图片获取技术方案

### 7.1 图片来源优先级

1. **Oceanus-all WoRMS_papers 提取的图片** (最高质量,PDF原始分辨率)
2. **中国动物志图片** (高质量,物种鉴定级)
3. **WoRMS 网站图片** (`images.marinespecies.org`,中等质量缩略图)

### 7.2 WoRMS 图片获取流程

```python
# 伪代码
for species in apsma_species_subset:
    if species.rank != "Species":
        continue
    # 1. 访问详情页
    url = f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={species.aphia_id}"
    html = requests.get(url).text
    # 2. 提取图片URL
    img_urls = re.findall(r'https://images\.marinespecies\.org/thumbs/([^"?]+)', html)
    # 3. 去掉缩略图参数,获取更大尺寸
    full_urls = [f"https://images.marinespecies.org/bimages/{img}" for img in img_urls]
    # 4. 下载 (限速,避免被封)
    for img_url in full_urls[:5]:
        download(img_url, f"species_image_library/{phylum}/{species_name}/")
```

### 7.3 注意事项

- **WoRMS 无 REST API 图片端点** (`AphiaImagesByAphiaID` 返回 404)
- 需通过 HTML 详情页解析获取图片URL
- 限速: 每次请求间隔 1-2秒,避免被封IP
- 并发: 最多 5 个线程同时下载
- 每物种最多5张图片,优先选择分辨率高的
- 物种名含特殊字符(空格等)需URL编码

---

## 八、与现有项目数据的融合点

| 继承数据 | 现有项目数据 | 融合方式 |
|---------|------------|---------|
| taxon_node.csv → 21门子集 | apsma_knowledge.jsonl (186形态型) | WoRMS精细分类树增强形态型的taxonomy字段 |
| taxonrelationship.csv → 21门子集 | 09脚本Neo4j导入 | 直接作为Neo4j节点/关系源,构建完整分类层级 |
| speciesprofile.txt | APSMA形态型的features字段 | 补充深度范围、分布、底质等生态属性 |
| WoRMS 物种图片 | BioCLIP嵌入(185形态型) | 图片检索库可补充更多物种的视觉特征 |
| WoRMS 物种描述MD | apsma_knowledge.jsonl | 物种描述知识库增强形态型的描述字段 |
| 中国动物志图片+描述 | 生物图集检索库(1,287图) | 补充中国海域物种的详细鉴定特征 |
| OBIS出现记录 | BISMaL DwC-A (977K记录) | OBIS补充缺失的西太出现记录 |
| 中文文献 | Frontiers CCZ文章(4篇) | 补充西太平洋特有文献知识 |

---

## 九、注意事项

1. **不复制原始文件**: WoRMS taxon_node.csv 512MB太大,只复制筛选后的APSMA子集
2. **保留元数据**: eml.xml 和 meta.xml 记录了WoRMS数据版本和下载时间,需保留
3. **物种名称去重**: WoRMS中同一个物种可能有accepted/unaccepted多条记录,需只保存accepted
4. **西太范围**: 经度 100°E-180°E, 深度 >500m (比APSMA的>3500m更宽泛)
5. **OBIS数据**: 18GB压缩包解压需磁盘空间,可能需在NAS临时目录操作
6. **WoS论文**: 3万+个PDF全是DOI编号命名,无法按名筛选,暂不处理
7. **Oceanus-all 路径**: 使用符号链接而非复制,避免占用NAS存储空间
