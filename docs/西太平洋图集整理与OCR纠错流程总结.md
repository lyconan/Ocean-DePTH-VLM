# 西太平洋深海生物图集整理与OCR纠错全流程总结

> 文档生成日期：2026-08-27
> 处理对象：`data/raw/生物图集/西太平洋深渊及临近区底栖生物合集.pdf`（231页，286个物种，最终337张图片）
> 输出文件：`data/processed/atlas_lookup/species_name_index.json`

---

## 一、背景与目标

《西太平洋深渊及临近区底栖生物合集》PDF共231页：
- **第1–24页**：目录与分类系统说明
- **第25–224页**：正文，286个物种条目（序号 + 中文名 + 拉丁名 + 照片），正文排版为"标题在上页底部、照片在下页顶部"的跨页布局
- **第225–231页**：拉丁名索引（两列排版）

出版年份较早，PDF内嵌文字层的OCR质量差，主要错误类型：

| 错误类型 | 示例 |
|---------|------|
| 数字↔字母混淆 | `A1cyonacea`(l→1)、`Paxi110sida`(ll→11)、`Pennatu1acea` |
| 字符替换错误 | `Epizoan:ie`、`B1:ae`、`Paelopati~lel18861Benthothuria` |
| 作者名粘连进学名 | `AciaH1882`(Actiniaria, H. 1882)、`FungiaSars1872` |
| 中文字形误读 | 鼬鳚目→"翩蹦目"、栉水母→"柿水母"、海鳃→"海鲍"、"珊蝴"→珊瑚 |
| 结构性错误 | 同一页多个物种时图片错配；大图被切成上下条带 |

目标是产出一份**权威可检索**的 `species_name_index.json`：每个物种名字在WoRMS可精确命中，taxonomy谱系完整，图片-物种一一对应。

---

## 二、总体流程概览

```
PDF原文 ─┬─→ 图片提取 → 切片合并 → 位置感知映射 → 337张图全部归属正确物种
         ├─→ 物种标题检测（286个序号位置）→ 阅读-顺序所有权判定
         ├─→ 拉丁名索引页提取（225-231页两列解析）→ 权威词表
         └─→ 学名/中文名/taxonomy三层校验修复：
              ① WoRMS AphiaRecordsByName 全量精确匹配
              ② PDF原文逐页核对（OCR乱码处人工判读）
              ③ GBIF分类谱系重建（phylum→genus自动归位）
```

---

## 三、图片提取与物种映射

### 3.1 发现的问题

1. **纯页码映射会54%错配**：原脚本按页码把图片归给同页物种，但正文是跨页排版。典型案例：第27页底部"拂子介属未定种"的标题旁没有它的照片——照片在第28页顶部（`Porifera_p028_90.jpeg`），却被错配给了第28页的下一个物种。
2. **图片对象被切成条带**：PDF排版引擎将大照片切成上下两条存储（如xref 80占y=69-184、xref 81占y=184-299，同宽无缝相接），逐对象提取导致一张照片变两个文件。
3. **29张图片静默丢失**：原提取脚本 try/except 吞异常，8%的图片根本没落盘。
4. **物种列表缺seq 1/2两条**（第25页起始两物种未收录）。

### 3.2 解决方法：位置感知映射

核心思想——**图片文件名里的数字是PDF的xref（图片对象号），可以反查其在页面上的精确坐标**：

```python
# 1) 文件名 Porifera_p027_86.jpeg 中 "86" 是 xref
# 2) get_image_info(xrefs=True) 建立 xref → (page, bbox_y_top) 索引
# 3) 用 PyMuPDF 的 get_text("dict") 找到全部286个物种标题的正则模式：
#    r'^(\d{1,3})\s*[\.．、]\s*([\u4e00-\u9fff])'   # 序号.中文
#    r'^(\d{1,3})\s*[\.．、]\s*([A-Za-z])'          # 序号.拉丁名
# 4) 按 (page, y) 阅读顺序排序标题；每张图片归属其上方最近的标题
#    （允许跨页延续 —— 这是解决跨页错配的关键）
def find_owner(page, y):
    best = None
    for h in headers:  # 已按阅读顺序排序
        if (h.page, h.y) <= (page, y): best = h
        else: break
    return best.seq if best else None
```

需要处理的边界情况：

| 边界情况 | 处理 |
|---------|------|
| 同一seq出现两次（目录页+正文） | 只保留页码大的（正文的） |
| 序号后直接跟拉丁名无中文 | 增加第二正则模式 |
| OCR读错序号（seq17的"17."被读成"n"，seq200-207整页乱码） | 对照species_list期望页码逐页搜索；仍失败则人工目视定位，写入MANUAL字典硬编码 |
| 页眉装饰横幅（宽度100%小高度） | 过滤条件：`width≥200 && height≥100 && xref≠1117` |

### 3.3 切片图片合并

检测算法：同一页内，**bbox的x范围相同（±2px）且上下边无缝相接（±2px）**的连续图片链即为一组切片。全书共检测出**59组切片对**。

处理方式：不用原图字节拼接（PNG/JPEG不可靠），而是**用高分辨率重新渲染联合区域**：

```python
scale = strip_w_px / (x1 - x0)          # 分辨率匹配原像素宽
pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                     clip=pymupdf.Rect(x0, y0, x1, y1))
pix.save(out_path, jpg_quality=92)
```

保留组内最小xref的文件名，删除其余条带的文件和索引条目。

### 3.4 结果

- 396张初始提取 → 合并59组切片 → **337张完整图片**
- 磁盘文件数 = westpac_entries数 = species images总数 = 337
- 286/286物种全部有图片（含补提取的seq 1/2/137/188及21张遗漏图）
- **门-文件夹-文件名前缀一致性审计**：每张图所在文件夹与物种真实phylum逐一对齐（seq85栉水母是独立的Ctenophora门，从Cnidaria文件夹迁出并重命名`Ctenophora_p087_387.jpeg`；seq8玻璃海绵phylum从误填的Cnidaria改回Porifera并移回Porifera文件夹）。文件名前缀 = 所在文件夹 = taxonomy.phylum，三者100%一致

---

## 四、拉丁名索引利用（PDF第225–231页）

书末附拉丁名索引（两列排版）。关键点：

### 4.1 两列感知文本提取

朴素 `page.get_text()` 会把左右两列文字交织，必须用span位置分列：

```python
d = page.get_text("dict")
col_boundary = page.rect.width * 0.48   # x中心 < 48% → 左列
# 每个 span 按 (x_center < boundary ? 左 : 右, y_top) 排序拼接行
```

### 4.2 重要教训

索引本身也有同样的OCR错误！直接用模糊匹配会把正确的名字改错：
- `Pt erasteridae`（多空格，应为Pterasteridae）
- `E1pidiidae`（l→1）

因此必须**先深度清洗索引词表（去空格、修1→l、剔CJK字符），再用于比对**；且只对具有明确OCR特征的现有名进行纠正（相似度>85%、长度差≤2）。

### 4.3 实际发挥的作用

索引条目303条，清洗后125个base_names、265个full_names。主要价值：
1. 作为参照验证数据集覆盖度
2. 校验"sp."格式统一性（30多处 `sp` → `sp.`）
3. 协助识别Author名残留（如`Relicanthus daphneae Daly`应去掉"Daly"）

---

## 五、三级学名校验体系（核心方法论）

### 5.1 第一级：WoRMS数据库全量精确校验

⚠️ **WoRMS REST API 踩坑记录**（旧教程已失效）：

| 端点 | 现状 |
|------|------|
| `/rest/AphiaRecordsByMatchNames/{name}` | ❌ 返回404 |
| `/rest/AphiaClassificationByID/{id}` | ❌ 返回404 |
| `/rest/AphiaRecordsByName/{name}?marine_only=false` | ✅ 可用（404/204=不存在） |
| GBIF `api.gbif.org/v1/species/match?verbose=true` | ✅ 模糊建议备用源（WoRMS数据GBIF同源） |

对文件中**所有120个唯一学名基础名**逐一请求（间隔0.8s限速）：

```
119个精确命中 ✓
1个不存在 → Aldrovandia tornis
```

`tornis`不存在的排查路径：
1. GBIF verbose模糊匹配 → 无建议
2. 列举WoRMS属下所有有效种 → 无相似名
3. **回查PDF原文该页**：`Aldrovandia t{伊'nis (Günther, 1877)` —— 种加词乱码但授权者清晰
4. `Günther, 1877`命名的Aldrovandia仅有**affinis** → "ffi"连字被OCR破坏
5. 结论：修正为 `Aldrovandia affinis`

这就是**"数据库否定 + PDF原文回查 + 分类学文献三角验证"**的标准排错流程。

### 5.2 第二级：PDF原文逐页核对

对WoRMS无法判定的中文/拉丁名乱码，回到`get_text("dict")`看原始OCR输出并人工判读。典型案例：

- 第161页整页乱码 `"码'grosoma"` ——"Hygrosoma"前缀H被误读 → 该物种实为**革海胆属Hygrosoma sp.**而非Sperosoma

⚠️ **seq49的教训（初始判断被推翻）**：第61页原文 `"鞠群海葵眉来定种与'izOlln硝us sp."` 严重乱码。最初依据残缺的`Olindius`字样和"海葵"字样误判为水母属Olindias/笠水母属——**错了**。回查同页分类地位段发现明确写着`群体海葵目 Zoantharia`、`鞘群海葵科 Epizoanthidae`、`鞘群海葵属 Epizoanthus`，`'izOlln硝us`正是`Epizoanthus`的OCR乱码。正确解读是**鞘群海葵属未定种 Epizoanthus sp.**（群体海葵，与水母无关）。教训：判读乱码时要综合中文名+科属后缀+分类地位段三方证据，不能只抓局部相似片段。

### 5.3 第三级：网络文献核实中文名

用WebSearch核实生物分类中文名对应关系（避免凭印象改错）：

| 属名 | 正确中文名 | 易错的OCR |
|------|-----------|----------|
| Peniagone | 叶疣参属 | 叶斑参/叶捷参/叶挠参/时挠番周（14处混用）|
| Sperosoma | 兜海胆属 | 草海胆 |
| Hygrosoma | 革海胆属 | （因上述OCR被误当Sperosoma）|
| Epizoanthus | 鞘群海葵属 | 鞠群海葵眉+'izOlln硝us（严重乱码）|
| Elpidiidae | 东参科 | 东参现朱建科 |
| Bathyplotes | 渊游参属 | 渊游蕃属 |

中文名系统性OCR错误修复清单：
- 鱼类目名：翩蹦目/融制目/舶崩目 → **鼬鳚目**（字体渲染差异致三处不同乱码）
- 科名：长尾白雪科 → **长尾鳕科**、合鲤鳗科 → **合鳃鳗科**
- 种名：野突吻蟹 → **野突吻鳕**（Coryphaenoides rudis是鱼类不是蟹）
- 后缀：海参细→海参纲、翅海星料→翅海星科、真盹尾目→真蛇尾目
- 形近字：珊蝴→珊瑚(15处)、柿水母→栉水母、烟灰峭→烟灰蛸、海鲍/海鲤→海鳃、深海奇铜→深海奇鲷

### 5.4 编号缺失的推导

序号类未定种缺失编号时用邻居推断：
- `Elpidiidae sp.`缺编号，上下邻居是sp.1和sp.3 → 推导为**sp. 2**
- `Psychropotes sp.`(seq206)，seq207是sp.2 → 推导为**sp. 1**
- `Paelopatides/Benthothuria`，中文名标着"未定种9" → 补为**sp. 9**

---

## 六、Taxonomy分类树重建

原始数据的taxonomy字段问题最严重：全量扫描出243个唯一rank值，其中约100个是OCR垃圾（`Hexacora1lia`、`Panda1idae`、`Ghyrothuriidae`……），还有**阶元放错位**（12个物种的genus字段装的是科名甚至目名）、部分物种taxonomy完全为空。

### 6.1 Phase 1：规则归一化

建立40余条OCR映射规则表一次性清洗：
```python
RULES = {
  'A1cyonacea': 'Alcyonacea', 'Paxi110sida': 'Paxillosida',
  'Hexacora1lia': 'Hexacorallia', 'Occora11ia': 'Octocorallia',
  'AciaH1882': 'Actiniaria',       # 作者粘连
  'FungiaSars1872': 'Fungiacyathus', # author粘连+错误属
  'Flastera': 'Freyastera',        # l→丢失
  ...
}
```
归一化得192个唯一名。

### 6.2 Phase 2：逐名WoRMS验证

192个归一化后的taxon name再次全部过API → **187通过，5失败**均修复：
```
Actinemidae → Actinernidae    Anoptilidae → Anthoptilidae
Actinemoidea → Actinernoidea  Epizoan:ie → Olindiidae
Synallactidal → Synallactida
```

### 6.3 Phase 3：GBIF谱系自动填充

因WoRMS `AphiaClassificationByID`已404，改用GBIF获取完整谱系：
```python
g = requests.get('https://api.gbif.org/v1/species/match',
                 params={'name': anchor, 'strict': 'true'}).json()
d = requests.get(f"https://api.gbif.org/v1/species/{g['usageKey']}").json()
# d 直接含 phylum/class/order/family/genus 完整链
```

锚定策略：取每个物种**最深的有效rank值**作为锚（genus > subfamily > family > ... > phylum），从锚向两端展开。

⚠️ **GBIF新分类系统的陷阱（必须回滚）**：GBIF采用2016年后的新系统，会把传统分类"现代化"改写，但**图集使用传统分类，必须以图集为准**。初版曾自动采纳GBIF结果，后被逐一回滚：

| GBIF新系统 | 图集/传统分类 | 回滚对象 |
|-----------|--------------|---------|
| Alcyonacea拆分为Scleralcyonacea/Malacalcyonacea | **Alcyonacea软珊瑚目** | 27种八放珊瑚 |
| Pennatulacea并入Scleralcyonacea | **Pennatulacea海鳃目** | seq83花羽海鳃 |
| Paragorgiidae并入Coralliidae | **Paragorgiidae拟柳珊瑚科** | seq75-79 |
| Alcyoniidae并入Coralliidae | **Alcyoniidae+Alcyoniina+Anthomastinae** | seq81 Anthomastus |

GBIF的正确用法：**只用它补全和归位谱系骨架（phylum/class/order的上下文），阶元名称本身以图集原文为准，发现冲突时回滚到图集写法**。

### 6.4 次级阶元（suborder/superfamily/subfamily）的教训

初版曾从PDF"当前页+下一页"扫描填充次级阶元，填了约60处。但后续审计发现**同页多物种时块边界不严格会造成串块污染**（如seq91扇贝科被塞进了同页seq90海牛亚目的Doridina），最终确立的铁律：

1. **以物种标题为块边界**：次级阶元只从"本物种标题 → 下一物种标题"之间的文本提取，防跨物种污染
2. **PDF没写的阶元一律不填**：图集对很多类群（海绵、海参、鱼类）本来就只列到目或科，次级阶元留空是正确的，补出来反而是污染
3. **逐字段回验**：每个非空taxonomy字段与该物种自己的分类地位段做模糊匹配（阈值0.75，对整块文本防块边界漏检），匹配不上即清除
4. **genus字段专项审计**：清除一切"伪genus"——凡是与更高阶元字段相同、或匹配任何阶元词尾（`formes|idae|inae|oidea|acea|aria|ida`）、或不符合`^[A-Z][a-z]+$`属名形态的一律清空（曾发现`Ophidiiformes`目名、`Pectinidae`科名塞进genus字段的漏网案例，后缀正则必须覆盖全）

### 6.5 手动兜底

33个GBIF strict匹配失败的（深水海绵类GBIF收录不全、超科级GBIF不支持等），依据分类学知识手动填phylum/class/order。

⚠️ **"疑似OCR错误"也可能是真名**：seq41/42手遗海葵的分类地位段写着`Helenmonae`，初版以为是OCR错误改成了Enthemonae——实际**Helenmonae是Daly & Rodríguez 2019年真实发表的亚目**（专为无肌海葵建立）。图集出版后新发表的分类单元必须查证原文标注的命名年份再判断，不能见怪名就改。

---

## 七、最终交付状态

`data/processed/atlas_lookup/species_name_index.json` 的质量指标：

| 指标 | 数值 |
|------|------|
| 物种总数 | 286（seq 1–286连续无缺口）|
| 图片条目 | 337（与磁盘文件一一对应，无孤儿文件）|
| 无图片物种 | 0 |
| 学名WoRMS命中率 | **286/286（100%）** |
| 无phylum的物种 | 0 |
| 名字中的非ASCII残留 | 0（仅中文专用字段外）|
| OCR特征残留（数字混入/冒号/~/'）| 0 |
| sp格式 | 统一为 `sp.`（含编号）|
| taxonomy字段可溯源 | **每个非空字段均能在PDF原文该物种自己的分类地位段找到出处**（或为由学名直接推出的规范属名），无推测性/污染性内容 |
| genus字段纯净度 | 无伪genus（终验：全部286物种的genus要么为空，要么是规范单形属名且与上级阶元不冲突）|
| 门-文件夹-文件名一致性 | 100%（8个门各有独立文件夹）|

结构示例：
```json
{
  "seq": 176,
  "name_cn": "羽参属未定种",
  "name_latin": "Paroriza sp.",
  "page": 150,
  "taxonomy": {
    "phylum": "Echinodermata", "class": "Holothuroidea",
    "order": "Persiculida", "family": "Gephyrothuriidae",
    "subfamily": "", "genus": "Paroriza", ...
  },
  "images": [{"filename": "...", "path": "...", "page": 150}]
}
```

---

## 八、经验教训（复用价值）

1. **不要信任何单一来源**：PDF的OCR层错误率超乎想象，书末索引本身同样有错；而WoRMS能证明一个名字不存在，却不能告诉你正确的名字是什么——必须回PDF原文+分类学文献做三角验证。

2. **API端点会过期**：WoRMS REST的经典端点`AphiaRecordsByMatchNames`和`AphiaClassificationByID`现已404，网上大量代码不能用了。当前可用端点是`AphiaRecordsByName`（精确）+ GBIF match（模糊与谱系）。

3. **图片归属要用几何信息而非逻辑页码**：跨页排版下按页码分配图片必然出错，xref+bbox坐标+阅读顺序才是可靠方案。

4. **PDF切片是普遍现象**：排版引擎切大图为条带非常常见，逐对象提取前先做"同x范围+无缝相接"聚类合并。

5. **限速礼貌访问公共API**：0.5–1s的请求间隔避免被封IP，全程乐观重试。

6. **中间产出JSON做备份**：每次大批量修改前留`.bak`副本（本次修过一次把正确名改错的教训——那个备份救了场）。

7. **JSON文件可能被IDE意外编辑引入语法错误**（尾随逗号）：读取前用容错逻辑剥离。

8. **外部数据库的新分类系统不能自动覆盖源文献**：GBIF/WoRMS近年把传统目科级单元合并拆分（如Alcyonacea→Scleralcyonacea、Paragorgiidae→Coralliidae），初版照搬后被全部回滚。原则：**谱系骨架可用外部库补全，阶元名称本身以图集原文为准**。

9. **同页多物种是串块污染的温床**：按"当前页+下一页"宽松扫描提取阶元，会把邻居物种的分类信息塞进当前物种（如扇贝科被塞进同页海牛亚目的Doridina）。必须以物种标题为严格块边界，且**PDF没写的阶元留空才是对的**。

10. **阶元名做未定种时，学名必须取同级名称**：中文名是"XX亚目/总科/亚科未定种"的，拉丁名必须用同级名（如全轴柳珊瑚亚目→`Holaxonia sp.`而非上级的`Alcyonacea sp.`；海牛亚目→`Doridina sp.`），且同样要带`sp.`后缀。

11. **"疑似OCR错误"要先查命名年份**：PDF里的怪名可能是图集出版后新发表的分类单元（如Helenmonae，2019年发表），不能见怪名就"纠正"成常见名。

12. **乱码判读要三方互证**：中文名残片、拉丁名残片、分类地位段三处证据交叉确认才能定名（seq49的`'izOlln硝us`孤看像Olindias，结合"群体海葵目+Epizoanthidae科"才能定为Epizoanthus）。

---

## 九、可复用的脚本资产

以下脚本可独立复跑或移植到其他图集（位于/tmp，建议迁移到项目scripts/pipeline/保存）：

| 脚本 | 功能 |
|------|------|
| `fix_image_species_mapping.py` | 位置感知的图片→物种映射 |
| `extract_missing_images.py` | 检测+补提取遗漏图片 |
| `merge_split_images.py` | 检测并合并切片图片 |
| `worms_validate2.py` | WoRMS精确匹配+GBIF模糊建议两级校验 |
| `apply_name_fixes.py` | 学名/中文名批量修复（含修改清单数据）|
| `tax_phase1.py` / `tax_phase3_v2.py` / `tax_phase4.py` | taxonomy四阶段重建 |

---

*本文档由整理过程的真实操作日志提炼，所有修复均已通过最终校验脚本确认。*
