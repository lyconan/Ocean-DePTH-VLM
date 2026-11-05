"""APSMA搜索API — 独立可交付的REST API服务 (v3.1.3)

可独立部署，供科考人员查询APSMA知识库:
- 文本搜索: /search?query=sponge
- 图像搜索: /search/image (上传图像→BioCLIP→向量检索)
- 分类信息: /taxonomy/{morphotype_id}

v3.1.3: 改用 utils.bioclip_embed 共享工具加载 BioCLIP (via open_clip),
        避免 bioclip 包未安装时的导入失败。
"""
import sys
from pathlib import Path

# 添加 scripts 目录到 path 以便导入 utils
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from typing import List, Optional
import os
import torch
import io
from PIL import Image

from utils.bioclip_embed import get_bioclip_model, extract_embedding as _extract_emb

app = FastAPI(title="APSMA Search API", version="1.0.0",
              description="深海底栖生物APSMA形态型知识检索API")

# 全局连接
_driver = None
_clip_model = None
_clip_processor = None

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.17.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


@app.on_event("startup")
async def startup():
    global _driver, _clip_model, _clip_processor
    _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # 加载 BioCLIP (via open_clip, 委托给共享工具)
    # 注意: 本 API 查询 morphotype_exsitu_embeddings 索引, 该索引仍是基座嵌入空间
    #       (2026-11-03 v2 重提只更新了 EntryImage.embedding), 必须显式 lora=False,
    #       否则查询/库嵌入失配。待该索引用 v2 重提后再移除。
    _clip_model, _clip_processor = get_bioclip_model(lora=False)


@app.on_event("shutdown")
async def shutdown():
    if _driver:
        _driver.close()


class TaxonomyInfo(BaseModel):
    morphotype_id: str
    phylum: Optional[str] = None
    class_name: Optional[str] = None
    order_name: Optional[str] = None
    morphotype: Optional[str] = None
    description: Optional[str] = None
    size_range_cm: Optional[List[Optional[float]]] = None
    substrate: Optional[str] = None
    depth_range_m: Optional[List[Optional[float]]] = None
    ccz_areas: Optional[List[str]] = None
    image_path: Optional[str] = None
    score: Optional[float] = None  # 相关性分数


class SearchResult(BaseModel):
    results: List[TaxonomyInfo]
    total: int


@app.get("/search", response_model=SearchResult)
def search_text(
    query: str = Query(..., description="形态型名称、描述或分类的文本搜索查询"),
    limit: int = Query(20, ge=1, le=100),
):
    """文本搜索APSMA形态型(名称、描述、分类)。"""
    if _driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialized")
    with _driver.session() as session:
        result = session.run("""
            CALL db.index.fulltext.queryNodes('morphotype_text_index', $q)
            YIELD node, score
            OPTIONAL MATCH (node)-[:OCCURS_IN]->(a:CCZArea)
            RETURN node.morphotype_id AS morphotype_id,
                   node.phylum_name AS phylum,
                   node.class_name AS class_name,
                   node.order_name AS order_name,
                   node.name AS morphotype,
                   node.description AS description,
                   [node.size_min_cm, node.size_max_cm] AS size_range_cm,
                   node.substrate AS substrate,
                   [node.depth_min_m, node.depth_max_m] AS depth_range_m,
                   collect(a.name) AS ccz_areas,
                   node.exsitu_image_path AS image_path,
                   score
            ORDER BY score DESC
            LIMIT $limit
        """, q=query, limit=limit)
        results = [dict(r) for r in result]
    return SearchResult(results=results, total=len(results))


@app.post("/search/image", response_model=SearchResult)
async def search_image(
    file: UploadFile = File(...),
    limit: int = Query(10, ge=1, le=50),
):
    """图像相似性搜索(上传图像→BioCLIP嵌入→向量检索)。"""
    if _driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialized")
    if _clip_model is None or _clip_processor is None:
        raise HTTPException(status_code=503, detail="CLIP model not initialized")
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # BioCLIP 嵌入 (使用共享工具, 兼容 open_clip 和 transformers)
    embedding = _extract_emb(image, _clip_model, _clip_processor)
    embedding_list = embedding.tolist()

    with _driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('morphotype_exsitu_embeddings', $k, $embedding)
            YIELD node, score
            OPTIONAL MATCH (node)-[:OCCURS_IN]->(a:CCZArea)
            RETURN node.morphotype_id AS morphotype_id,
                   node.phylum_name AS phylum,
                   node.class_name AS class_name,
                   node.order_name AS order_name,
                   node.name AS morphotype,
                   node.description AS description,
                   [node.size_min_cm, node.size_max_cm] AS size_range_cm,
                   node.substrate AS substrate,
                   [node.depth_min_m, node.depth_max_m] AS depth_range_m,
                   collect(a.name) AS ccz_areas,
                   node.exsitu_image_path AS image_path,
                   score
            ORDER BY score DESC
            LIMIT $limit
        """, k=limit, embedding=embedding_list, limit=limit)
        results = [dict(r) for r in result]
    return SearchResult(results=results, total=len(results))


@app.get("/taxonomy/{morphotype_id}", response_model=TaxonomyInfo)
def get_taxonomy(morphotype_id: str):
    """获取特定APSMA形态型的完整分类信息。"""
    if _driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialized")
    with _driver.session() as session:
        result = session.run("""
            MATCH (m:Morphotype {morphotype_id: $mid})
            OPTIONAL MATCH (m)-[:OCCURS_IN]->(a:CCZArea)
            RETURN m.morphotype_id AS morphotype_id,
                   m.phylum_name AS phylum,
                   m.class_name AS class_name,
                   m.order_name AS order_name,
                   m.name AS morphotype,
                   m.description AS description,
                   [m.size_min_cm, m.size_max_cm] AS size_range_cm,
                   m.substrate AS substrate,
                   [m.depth_min_m, m.depth_max_m] AS depth_range_m,
                   collect(a.name) AS ccz_areas,
                   m.exsitu_image_path AS image_path
        """, mid=morphotype_id)
        record = result.single()
        if not record:
            raise HTTPException(status_code=404, detail=f"Morphotype {morphotype_id} not found")
    return TaxonomyInfo(**dict(record))


@app.get("/health")
def health():
    """健康检查。"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # APSMA 搜索 API 服务
    # 容器内启动: python scripts/14_apsma_search_api.py
    # 宿主机访问: http://<容器IP>:8088/docs (Swagger UI)
    port = int(os.environ.get("APSMA_API_PORT", 8088))
    host = os.environ.get("APSMA_API_HOST", "0.0.0.0")
    print(f"[APSMA Search API] 启动服务: http://{host}:{port}")
    print(f"[APSMA Search API] Neo4j: {NEO4J_URI}")
    print(f"[APSMA Search API] 文档: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port, log_level="info")
