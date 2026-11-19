# 可视化控制台（Web Console）

深海生物知识系统的可视化控制台：知识图谱浏览、检索证据图库、盲测管线回放、VLM 终判接口状态。纯 Python 标准库实现（无任何第三方依赖），单文件后端 + 单页前端。

## 功能

| 页面 | 说明 | 依赖 |
|---|---|---|
| 总览 | 图谱统计、管线五阶段流程、门分布、证据库构成 | Neo4j |
| 知识库 | 2,641 个形态种条目检索与详情（分类阶元、WoRMS、知识描述、证据图集） | Neo4j + 图像目录 |
| 检索库 | 30,629 张证据图网格浏览，按来源筛选 | Neo4j + 图像目录 |
| 管线回放 | 盲测对象逐条结果（检测框 / 检索候选 / GT 对照 / 原图标注叠加） | Neo4j + 盲测数据（可选） |
| VLM 终判 | 接口状态与在线试跑（11 图选择题协议） | 配置 `vlm_api_url` 后启用 |

盲测数据缺失时管线回放自动降级为提示，知识库与检索库不受影响。

## 部署步骤

### 1. 准备 Neo4j

安装 Neo4j 5.x（Docker 或本机均可），恢复知识图谱数据：

```bash
# Docker 示例
docker run -d --name depth-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<你的密码> -v <数据卷>:/data neo4j:5.20.0

# 从魔搭下载 neo4j.dump 恢复（容器停机状态下）
docker run --rm -v <数据卷>:/data -v <dump目录>:/out neo4j:5.20.0 \
  neo4j-admin database load neo4j --from-stdin < /out/neo4j.dump
```

### 2. 准备图像目录

证据图（框切图）位于 `data/processed/` 下与图谱中 `image_id` 一致的相对路径，需与主仓数据布局一致。管线回放还需盲测数据（可选，见下方环境变量）。

### 3. 启动

```bash
cd Ocean-DePTH-VLM
python3 webapp/console_api.py --host 0.0.0.0 --port 8089
```

浏览器访问 `http://<服务器IP>:8089`。无第三方依赖，Python 3.8+ 即可。

## 配置（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEPTH_NEO4J_HTTP` | `http://localhost:7474/db/neo4j/tx/commit` | Neo4j HTTP API 地址 |
| `DEPTH_NEO4J_AUTH` | `neo4j:password` | Neo4j 认证（`用户:密码`，**务必改成你的密码**） |
| `DEPTH_BLIND_REPORT` | `<repo>/data/experiments/blindtest_e2e/e2e_report_postjedi.json` | 盲测指标报告 |
| `DEPTH_BLIND_OBJECTS` | `<repo>/data/experiments/blindtest_e2e/per_object_postjedi.jsonl` | 盲测逐对象结果 |
| `DEPTH_BLIND_MANIFEST` | `/tmp/exp/blindtest/manifest.json` | 盲测场景清单 |
| `DEPTH_DET_CACHE` | `/tmp/exp/blindtest/det_cache.pkl` | 盲测检测缓存 |

示例：

```bash
DEPTH_NEO4J_AUTH='neo4j:你的密码' DEPTH_BLIND_MANIFEST=/data/blindtest/manifest.json \
  python3 webapp/console_api.py --host 0.0.0.0 --port 8089
```

### VLM 终判接入

在 `config/graphrag_inference.yaml` 填入微调后模型的推理端点即可启用（OpenAI 兼容 API）：

```yaml
vlm_api_url: http://<VLM主机>:8001/v1
vlm_model: <模型名>
```

填入后控制台"VLM 终判"页自动显示"在线"，可对任意盲测对象在线试跑。
