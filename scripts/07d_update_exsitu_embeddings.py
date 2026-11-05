#!/usr/bin/env python3
"""更新 Neo4j 中 Morphotype 节点的 exsitu_embedding (用 BioCLIP 重新提取的嵌入)

用法:
    python scripts/07d_update_exsitu_embeddings.py \\
        --embeddings models/clip_morphotype_embeddings_bioclip.pt \\
        --neo4j-uri bolt://172.17.0.1:7687
"""
import argparse
import sys
from pathlib import Path

import torch
from neo4j import GraphDatabase


def update_embeddings(embeddings_path: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    """加载 .pt 文件并更新 Neo4j 中所有 morphotype 的 exsitu_embedding。"""
    print(f"[1/3] 加载嵌入: {embeddings_path}")
    embeddings = torch.load(embeddings_path, weights_only=False)
    print(f"  加载 {len(embeddings)} 个形态型嵌入")

    # 验证维度
    sample_id = next(iter(embeddings))
    sample_emb = embeddings[sample_id]
    if hasattr(sample_emb, 'shape'):
        print(f"  嵌入维度: {sample_emb.shape}")
        assert sample_emb.shape[0] == 512, f"期望 512 维, 实际 {sample_emb.shape[0]}"
    else:
        print(f"  嵌入类型: {type(sample_emb)}")

    print(f"\n[2/3] 连接 Neo4j: {neo4j_uri}")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    updated = 0
    failed = 0
    try:
        with driver.session() as session:
            # 先检查当前状态
            before = session.run(
                "MATCH (m:Morphotype) WHERE m.exsitu_embedding IS NOT NULL "
                "RETURN count(m) AS c"
            ).single()["c"]
            print(f"  更新前有 exsitu_embedding 的节点: {before}")

            for mid, emb in embeddings.items():
                emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                # 归一化 (确保是单位向量)
                import math
                norm = math.sqrt(sum(x * x for x in emb_list))
                if norm > 0:
                    emb_list = [x / norm for x in emb_list]

                result = session.run("""
                    MATCH (m:Morphotype {morphotype_id: $mid})
                    SET m.exsitu_embedding = $emb
                    RETURN m.morphotype_id AS id
                """, mid=mid, emb=emb_list)

                if result.single():
                    updated += 1
                else:
                    failed += 1
                    if failed <= 3:
                        print(f"  [警告] 未找到 morphotype_id={mid}")

        # 验证更新后状态
        with driver.session() as session:
            after = session.run(
                "MATCH (m:Morphotype) WHERE m.exsitu_embedding IS NOT NULL "
                "RETURN count(m) AS c"
            ).single()["c"]
            # 检查向量索引状态
            try:
                idx_state = session.run(
                    "SHOW INDEXES WHERE name='morphotype_exsitu_embeddings' "
                    "RETURN state, populationPercent"
                ).single()
                idx_info = f"state={idx_state['state']}, pop={idx_state['populationPercent']}%" if idx_state else "N/A"
            except Exception:
                idx_info = "查询失败"

    finally:
        driver.close()

    print(f"\n[3/3] 更新结果:")
    print(f"  成功更新: {updated} 个 morphotype")
    print(f"  失败: {failed} 个")
    print(f"  Neo4j 中现有 exsitu_embedding: {after} (更新前 {before})")
    print(f"  向量索引状态: {idx_info}")
    print(f"\n✓ BioCLIP exsitu 嵌入已写入 Neo4j!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="更新 Neo4j exsitu 嵌入 (BioCLIP)")
    parser.add_argument("--embeddings", default="models/clip_morphotype_embeddings_bioclip.pt")
    parser.add_argument("--neo4j-uri", default="bolt://172.17.0.1:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    args = parser.parse_args()

    update_embeddings(args.embeddings, args.neo4j_uri, args.neo4j_user, args.neo4j_password)
