"""DePTH-VLM 完整推理管线 — YOLO检测 + GraphRAG增强 + Set-of-Mark批量VLM推理 (v3.1.2)

管线流程:
1. YOLO11检测(目标提议网络) → 过滤噪声(conf < 0.2)
2. 双裁剪策略: tight_crop(BioCLIP) + contextual_crop(VLM)
3. BioCLIP嵌入 → Neo4j多视图向量检索 + 分类体系图谱遍历
4. Set-of-Mark标注 → 单次Qwen2.5-VL批量推理
5. 容错JSON解析 + 缺失目标并发重试 + VLM输出优雅降级
"""
import os
import re
import json
import base64
import asyncio
import threading
from io import BytesIO
from typing import List, Dict, Optional

import torch
from PIL import Image
from neo4j import GraphDatabase
from ultralytics import YOLO
from openai import OpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils.contextual_crop import contextual_crop, tight_crop
from utils.som_annotator import draw_som_annotations
from utils.schemas import normalize_vlm_result
from utils.taxonomy_mapper import infer_phylum_from_concept


class DePTHGraphRAGPipeline:
    """深海底栖生物识别管线 — v3.1.2 Set-of-Mark批量VLM + 并发重试 + 优雅降级"""

    def __init__(self, config: Dict):
        # Layer 1: YOLO检测(目标提议网络)
        # v3.3: dual_path=true 启用自训34纲双路推理 (960+TTA ∪ 1920, conf0.05/nms0.55;
        #       11c2 端到端实测 recall 87.4% / 剔Other纲acc 94.1%), 宁多勿漏,
        #       误检由检索层排序与VLM判读兜底
        self.dual_path = bool(config.get("dual_path", False))
        if self.dual_path:
            from utils.yolo_infer import DualPathDetector
            self.yolo = DualPathDetector(
                config.get("yolo_weights",
                           "runs/detect/yolo11m_class34-2/weights/best.pt"),
                device=config.get("yolo_device", "cuda:1"),
                paths=tuple(tuple(p) for p in
                            config.get("dual_paths", [[960, True], [1920, False]])),
            )
        else:
            self.yolo = YOLO(config.get("yolo_weights", "models/yolo_fathomnet.pt"))
        self.conf_filter = config.get("conf_filter", 0.2)
        self.iou_nms = config.get("iou_nms", 0.55)

        # BioCLIP嵌入(主要)，CLIP回退
        self._init_clip_model(config)

        # Neo4j连接
        self.neo4j_driver = GraphDatabase.driver(
            config.get("neo4j_uri", "bolt://localhost:7687"),
            auth=(config.get("neo4j_user", "neo4j"),
                  config.get("neo4j_password", "password"))
        )

        # VLM客户端(Qwen2.5-VL)
        # 72B AWQ INT4 量化模型单次推理需 ~800s, OpenAI默认 600s 会超时
        # 设置 timeout=1800s (30min) 以容纳长推理
        vlm_timeout = config.get("vlm_timeout", 1800.0)
        self.vlm = OpenAI(
            base_url=config.get("vlm_api_url", "http://localhost:8001/v1"),
            api_key=config.get("vlm_api_key", "EMPTY"),
            timeout=vlm_timeout,
            max_retries=2,
        )
        self.vlm_model = config.get("vlm_model", "oceanexus-72b-awq")
        self.vlm_temperature = config.get("vlm_temperature", 0.1)
        self.vlm_max_tokens = config.get("vlm_max_tokens", 4096)

        # SoM + Token预算
        self.max_vision_tokens = config.get("max_vision_tokens", 16000)
        self.tokens_per_image = config.get("tokens_per_image", 1600)
        self.max_concurrent_retries = config.get("max_concurrent_retries", 5)

        # GraphRAG检索参数
        self.top_k = config.get("top_k", 5)
        self.soft_boost = config.get("soft_boost", 1.05)
        # VLM输出Top-N形态型候选(默认3)
        self.vlm_top_n = config.get("vlm_top_n", 3)

        # YOLO分类提示控制
        # YOLO precision=0.149 时分类不可靠, 禁用提示让GraphRAG完全依赖BioCLIP检索
        self.use_yolo_hint = config.get("use_yolo_hint", True)

        # 消融实验控制标志
        self.skip_rag = config.get("skip_rag", False)
        self.skip_graph_enrichment = config.get("skip_graph_enrichment", False)
        self.skip_insitu_channel = config.get("skip_insitu_channel", False)

        # 管线状态(用于重试)
        self._last_detections = []
        self._last_references = {}
        self._async_vlm = None

    def close(self):
        """关闭Neo4j driver，释放连接资源。"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()

    def __del__(self):
        """析构时安全关闭driver。"""
        try:
            self.close()
        except Exception:
            pass

    def _init_clip_model(self, config: Dict):
        """初始化BioCLIP 2.5 模型 (via open_clip, 委托给共享封装)。
        v4.1: get_bioclip_model 默认加载 LoRA v2 适配器 — 与 Neo4j 库
              (entryimage_embeddings, 2026-11-02 v2 重提) 同一嵌入空间。
        """
        from utils.bioclip_embed import get_bioclip_model
        self.clip_model, self.clip_processor = get_bioclip_model()
        print("使用BioCLIP 2.5 嵌入模型 (via open_clip)")
        # v4.1: mid 质心缓存 (融合重排用), 懒加载
        self._centroids = None
        self._centroid_loaded = False

    # ═══════════════════════════════════════════════════════════
    # STEP 1: 检测(目标提议网络)
    # ═══════════════════════════════════════════════════════════
    def detect_organisms(self, seafloor_image_path: str) -> List[Dict]:
        """YOLO检测生物体 → 过滤噪声 → 返回提议。

        v3.3: dual_path=true 时走自训34纲双路 (960+TTA ∪ 1920) 合并检测,
              宁多勿漏 — 低conf多检出, 误检交由检索层与VLM兜底。
        v3.2: 支持细粒度YOLO (622类物种/属级), 自动映射到门级用于检索增强。
        YOLO是目标提议网络 + 分类提示器。
        所有conf >= conf_filter的检测都进入GraphRAG + VLM。
        """
        if self.dual_path:
            # DualPathDetector.predict 返回 List[List[dict]] (box/conf/cls/name)
            merged = self.yolo.predict([seafloor_image_path],
                                       conf=self.conf_filter,
                                       iou_nms=self.iou_nms)
            records = [
                {"class_name": d["name"],
                 "confidence": float(d["conf"]),
                 "bbox": [float(v) for v in d["box"]],
                 "class_id": int(d["cls"])}
                for d in (merged[0] if merged else [])
            ]
        else:
            results = self.yolo(seafloor_image_path, conf=self.conf_filter)
            records = [
                {"class_name": self.yolo.names[int(box.cls)],
                 "confidence": float(box.conf),
                 "bbox": box.xyxy.cpu().numpy()[0].tolist(),
                 "class_id": int(box.cls)}
                for r in results for box in r.boxes
            ]

        records = [rec for rec in records if rec["confidence"] >= self.conf_filter]

        detections = []
        for target_id, rec in enumerate(records, 1):
            class_name = rec["class_name"]
            # 细粒度YOLO: class_name 是物种/属名, 映射到门级用于检索增强
            # 34纲YOLO: class_name 即纲名, _CLASS_TO_PHYLUM_34 映射到门
            phylum = infer_phylum_from_concept(class_name)
            if phylum == "unknown":
                phylum = None  # 无法映射时不做门级约束

            det = {
                "bbox": rec["bbox"],
                "class_id": rec["class_id"],
                "class_name": class_name,  # 细粒度类别名/纲名
                "phylum": phylum,          # 门级分类 (用于检索soft_boost)
                "confidence": rec["confidence"],
                "target_id": target_id,
            }
            detections.append(det)
            print(f"[YOLO] Target #{target_id}: class={class_name} "
                  f"conf={rec['confidence']:.3f} bbox={rec['bbox']} "
                  f"phylum_hint={phylum or 'None'}")

        return detections

    # ═══════════════════════════════════════════════════════════
    # STEP 2: 上下文裁剪 + 多视图向量检索
    # ═══════════════════════════════════════════════════════════
    def retrieve_with_graph_enrichment(
        self,
        image_patch_pil: Image.Image,
        coarse_class: str = None,
        top_k: int = None,
    ) -> List[Dict]:
        """BioCLIP嵌入 → 多视图向量检索 → GraphRAG分类体系丰富。
        
        v3.1特性:
        - 多视图向量检索(insitu + exsitu通道)
        - 可选粗分类裁剪: YOLO高置信度时限制到门子图
        """
        # 消融实验: 跳过RAG检索
        if self.skip_rag:
            return []

        if top_k is None:
            top_k = self.top_k

        # 2a: BioCLIP嵌入 (使用共享工具, 兼容 open_clip 和 transformers)
        # v4.1: 翻转TTA — 原图+水平镜像嵌入平均归一化 (消除朝向敏感, 自检 +0.2pp)
        from utils.bioclip_embed import extract_embedding as _extract_emb
        embedding = _extract_emb(image_patch_pil, self.clip_model, self.clip_processor)
        emb_flip = _extract_emb(image_patch_pil.transpose(Image.FLIP_LEFT_RIGHT),
                                self.clip_model, self.clip_processor)
        embedding = embedding + emb_flip
        embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        embedding_list = embedding.tolist()

        with self.neo4j_driver.session() as session:
            # 2b: 双路径并行向量检索 + Soft重排序(v3.1.1)
            candidates = self._multi_view_vector_search(
                session, embedding_list, top_k=top_k,
                yolo_hint_phylum=coarse_class
            )

            # 消融实验: 跳过图谱遍历
            if self.skip_graph_enrichment:
                return candidates

            # 2c: 图谱丰富 — 遍历分类体系树
            for candidate in candidates:
                # v3.1.2: 从反范式化属性读取分类信息
                lineage_result = session.run("""
                    MATCH (ms:Morphospecies {morphospecies_id: $mid})
                    RETURN ms.phylum AS phylum,
                           ms.class_name AS class,
                           ms.order_name AS order_name
                """, mid=candidate["id"])

                lineage = lineage_result.single()
                if lineage:
                    candidate["phylum"] = lineage["phylum"]
                    candidate["class"] = lineage["class"]
                    candidate["order_name"] = lineage["order_name"]

                # 获取同属形态型
                siblings_result = session.run("""
                    MATCH (ms:Morphospecies {morphospecies_id: $mid})
                    MATCH (s:Morphospecies)
                    WHERE s.order_name = ms.order_name
                      AND s.morphospecies_id <> $mid
                    RETURN s.display_name AS sibling_name,
                           s.latin AS sibling_desc
                    LIMIT 5
                """, mid=candidate["id"])
                candidate["siblings"] = [dict(s) for s in siblings_result]

                # 地理分布(CCZArea/OCCURS_IN尚未导入,跳过查询避免WARNING)
                # geo_result = session.run("""
                #     OPTIONAL MATCH (m:Morphotype {morphotype_id: $mid})-[:OCCURS_IN]->(a:CCZArea)
                #     RETURN collect(a.name) AS areas
                # """, mid=candidate["id"])
                # geo = geo_result.single()
                # if geo and geo["areas"]:
                #     candidate["ccz_areas"] = geo["areas"]

                # WoRMS对齐
                worms_result = session.run("""
                    MATCH (ms:Morphospecies {morphospecies_id: $mid})-[w:WORMS_MATCHED]->(wt:WoRMSTaxon)
                    RETURN coalesce(w.mapping_type, 'WORMS_EXACT') AS mapping_type,
                           coalesce(w.matched_rank, 'accepted') AS matched_rank,
                           wt.aphia_id AS aphia_id
                """, mid=candidate["id"])
                worms = worms_result.single()
                if worms:
                    candidate["worms_mapping"] = dict(worms)

        return candidates

    def _load_centroids(self):
        """v4.1: 懒加载 mid 质心矩阵 (scripts/35_build_centroids.py 产出)。
        文件缺失时退化为纯图相似排序 (不阻断服务)。
        库变更(增删EntryImage/换嵌入模型)后需重跑 35 号脚本刷新。
        """
        if self._centroid_loaded:
            return
        self._centroid_loaded = True
        path = (Path(__file__).parent.parent
                / "data" / "experiments" / "ft_20261102" / "centroids_v2.npz")
        if not path.exists():
            print(f"[GraphRAG] 警告: 质心文件缺失 ({path}), 融合重排降级为纯图相似")
            return
        import numpy as np
        z = np.load(str(path))
        mids = [str(m) for m in z["mids"]]
        self._centroids = {
            "C": z["C"],                     # (n_mid, 1024) f32, L2归一化
            "idx": {m: i for i, m in enumerate(mids)},
        }
        print(f"[GraphRAG] 质心已加载: {len(mids)} mids (融合重排就绪)")

    def _apply_centroid_fusion(self, merged: Dict[str, Dict], embedding: list):
        """v4.1: 融合重排 — score = 0.5×最高图相似 + 0.5×(查询·质心)。
        就地修改 merged[mid]["score"], 并保留 image_score/centroid_score 供调试。
        """
        self._load_centroids()
        if not self._centroids or not merged:
            return
        import numpy as np
        q = np.asarray(embedding, dtype=np.float32)
        C, idx = self._centroids["C"], self._centroids["idx"]
        hit = [m for m in merged if m in idx]
        if not hit:
            return
        rows = np.array([idx[m] for m in hit])
        csims = (q[None, :] @ C[rows].T)[0]  # (k,)
        for j, mid in enumerate(hit):
            c = merged[mid]
            c["image_score"] = c["score"]
            c["centroid_score"] = float(csims[j])
            c["score"] = 0.5 * c["score"] + 0.5 * c["centroid_score"]

    def _multi_view_vector_search(
        self,
        session,
        embedding: list,
        top_k: int = 5,
        yolo_hint_phylum: str = None,
    ) -> List[Dict]:
        """双路径并行检索 + Soft重排序(v3.1.1)。
        
        Path A: 全局多视图检索(无门约束)
        Path B: 子图检索(YOLO提示门约束，B-tree索引加速)
        合并: YOLO提示门获得1.05×分数提升(非硬过滤)
        """
        # Path A: 全局检索
        # v4.0: BioCLIP2.5 每图一嵌入 — 检索 EntryImage 节点, 聚合到 Morphospecies (每条目保留最高分)
        # insitu/exsitu 区分不再需要: 新图像库的 FathomNet/JODD 图本身就是 in-situ 域
        img_results = session.run("""
            CALL db.index.vector.queryNodes('entryimage_embeddings', $k, $embedding)
            YIELD node, score
            MATCH (ms:Morphospecies {morphospecies_id: node.msp_id})
            RETURN ms.morphospecies_id AS id,
                   ms.display_name AS name,
                   '' AS description,
                   '' AS substrate,
                   ms.phylum AS phylum,
                   ms.class_name AS class,
                   ms.order_name AS order_name,
                   node.image_id AS exsitu_image_path,
                   score, 'entryimage' AS source
            ORDER BY score DESC
        """, k=max(top_k * 4, 50), embedding=embedding)  # v4.1: 候选池≥50 (重排需要)

        # 聚合: 每个条目保留最高分
        global_merged = {}
        for r in img_results:
            mid = r["id"]
            if mid not in global_merged or r["score"] > global_merged[mid]["score"]:
                global_merged[mid] = dict(r)

        # v4.1: 质心融合重排 (检索v2方案)
        #   score(mid) = 0.5 × max图相似 + 0.5 × (查询·质心)
        #   FN层自检 top-5: 96.2% → 98.8%
        self._apply_centroid_fusion(global_merged, embedding)

        # Path B: 子图检索(YOLO提示)
        pruned_candidates = []
        if yolo_hint_phylum:
            try:
                pruned_results = session.run("""
                    CALL db.index.vector.queryNodes('entryimage_embeddings', $k2, $embedding)
                    YIELD node, score
                    MATCH (ms:Morphospecies {morphospecies_id: node.msp_id})
                    WHERE ms.phylum = $phylum_hint
                    RETURN ms.morphospecies_id AS id,
                           ms.display_name AS name,
                           coalesce(ms.description, '') AS description,
                           '' AS substrate,
                           ms.phylum AS phylum,
                           ms.class_name AS class,
                           ms.order_name AS order_name,
                           node.image_id AS exsitu_image_path,
                           score
                    ORDER BY score DESC
                    LIMIT $k
                """, k=top_k, k2=top_k*4, embedding=embedding, phylum_hint=yolo_hint_phylum)
                pruned_candidates = [dict(r) for r in pruned_results]
            except Exception as e:
                print(f"  [Path B 子图查询失败]: {e}")

        # Soft重排序(v3.3: 门内boost + 跨门降权)
        all_candidates = list(global_merged.values()) + pruned_candidates
        print(f"[GraphRAG] 全局候选={len(global_merged)} 子图候选={len(pruned_candidates)} "
              f"YOLO门提示={yolo_hint_phylum or 'None'}")
        for c in all_candidates:
            if yolo_hint_phylum and c.get("phylum") == yolo_hint_phylum:
                c["adjusted_score"] = c["score"] * self.soft_boost
                c["hint_boosted"] = True
            elif yolo_hint_phylum and c.get("phylum") and c.get("phylum") != yolo_hint_phylum:
                # 跨门降权: 将跨门候选的分数降低50%,减少跨门干扰
                c["adjusted_score"] = c["score"] * 0.5
                c["hint_boosted"] = False
                c["cross_phylum_penalty"] = True
                print(f"[GraphRAG] 跨门降权: {c['id']} phylum={c.get('phylum')} "
                      f"raw={c['score']:.3f} adj={c['adjusted_score']:.3f}")
            else:
                c["adjusted_score"] = c["score"]
                c["hint_boosted"] = False

        # 去重 + 排序
        seen = set()
        final = []
        for c in sorted(all_candidates, key=lambda x: x.get("adjusted_score", x["score"]), reverse=True):
            if c["id"] not in seen:
                seen.add(c["id"])
                final.append(c)

        for i, c in enumerate(final[:top_k]):
            flag = ""
            if c.get("hint_boosted"): flag = " [BOOSTED]"
            if c.get("cross_phylum_penalty"): flag = " [CROSS-PENALTY]"
            print(f"[GraphRAG] Top-{i+1}: {c['id']} score={c['score']:.3f} "
                  f"adj={c.get('adjusted_score',c['score']):.3f} "
                  f"phylum={c.get('phylum','?')}{flag}")

        return final[:top_k]

    # ═══════════════════════════════════════════════════════════
    # STEP 3: SET-OF-MARK批量VLM推理
    # ═══════════════════════════════════════════════════════════
    def batch_vlm_reason(
        self,
        annotated_image_pil: Image.Image,
        detections: List[Dict],
        all_references: Dict,
    ) -> List[Dict]:
        """批量VLM推理 — Token预算控制 + 交错组装(v3.1.1)。
        
        三级保护:
        Level 1: Token预算硬上限(max_vision_tokens)
        Level 2: 交错文本-图像组装
        Level 3: 目标分桶(<=3→1调用, 4-8→2调用, >8→3+调用)
        """
        buckets = self._bucket_detections(detections)

        all_results = []
        for bucket in buckets:
            bucket_refs = {det["target_id"]: all_references.get(det["target_id"], [])
                          for det in bucket}
            results = self._single_batch_call(
                annotated_image_pil, bucket, bucket_refs
            )
            all_results.extend(results)

        return all_results

    def _bucket_detections(self, detections: List[Dict]) -> List[List[Dict]]:
        """按空间位置分桶(v3.1.1)。"""
        if len(detections) <= 3:
            return [detections]

        sorted_dets = sorted(detections, key=lambda d: d["bbox"][1])

        if len(detections) <= 8:
            mid = len(sorted_dets) // 2
            return [sorted_dets[:mid], sorted_dets[mid:]]
        else:
            chunk_size = (len(sorted_dets) + 2) // 3
            return [sorted_dets[i:i+chunk_size]
                    for i in range(0, len(sorted_dets), chunk_size)]

    def _single_batch_call(
        self,
        annotated_image_pil: Image.Image,
        detections: List[Dict],
        all_references: Dict,
    ) -> List[Dict]:
        """执行单次VLM API调用 + Token预算限制。"""
        # Level 1: Token预算
        ref_budget = self.max_vision_tokens - self.tokens_per_image

        ref_images_for_target = {}
        for det in detections:
            tid = det["target_id"]
            refs = all_references.get(tid, [])
            target_ref_imgs = []

            for ref in refs:
                if ref_budget <= 0:
                    break

                image_path = ref.get("exsitu_image_path") or ref.get("image_path")
                if image_path and os.path.exists(image_path):
                    try:
                        ref_img = Image.open(image_path).convert("RGB")
                        ref_img.thumbnail((384, 384), Image.Resampling.LANCZOS)
                        ref_b64 = self._encode_image_to_base64(ref_img)
                        target_ref_imgs.append(ref_b64)
                        ref_budget -= self.tokens_per_image
                    except Exception:
                        continue

            ref_images_for_target[tid] = target_ref_imgs

        # Level 2: 交错文本-图像组装
        img_b64 = self._encode_image_to_base64(annotated_image_pil)

        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]

        for det in detections:
            tid = det["target_id"]
            refs = all_references.get(tid, [])

            target_text = self._build_single_target_text(det, refs)
            content.append({"type": "text", "text": target_text})

            for ref_b64 in ref_images_for_target.get(tid, []):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{ref_b64}"}
                })

            num_text_only = len(refs) - len(ref_images_for_target.get(tid, []))
            if num_text_only > 0:
                content.append({"type": "text", "text":
                    f"\n[注意: {num_text_only} 个参考因视觉Token预算限制仅以文本提供]\n"})

        instructions = self._build_vlm_instructions(
            len(detections),
            target_ids=[det["target_id"] for det in detections]
        )
        content.append({"type": "text", "text": instructions})

        # VLM调用
        response = self.vlm.chat.completions.create(
            model=self.vlm_model,
            messages=[{"role": "user", "content": content}],
            temperature=self.vlm_temperature,
            max_tokens=self.vlm_max_tokens
        )

        return self._parse_batch_response(
            response.choices[0].message.content,
            expected_target_ids=[det["target_id"] for det in detections]
        )

    def _build_single_target_text(self, det: Dict, refs: List[Dict]) -> str:
        """构建单个目标的交错文本块。"""
        tid = det["target_id"]
        conf = det["confidence"]
        # use_yolo_hint=False 时不向VLM传递YOLO分类提示, 避免误导
        yolo_hint = det["class_name"] if self.use_yolo_hint else "N/A"

        ref_text = ""
        for i, ref in enumerate(refs, 1):
            if ref is None:
                continue
            lineage = f"{ref.get('phylum', '?')} > {ref.get('class', '?')} > {ref.get('order_name', '?')}"
            ref_text += f"  Ref {i} (score {ref.get('score', 0):.3f}): {ref.get('name', 'N/A')} | Taxonomy: {lineage}\n"
            desc = ref.get('description') or 'N/A'
            ref_text += f"    Description: {desc[:200]}\n"

        return f"""---
### Target #{tid} (YOLO hint: {yolo_hint}, conf: {conf:.2f})
Reference images for Target #{tid} are attached immediately below.
{ref_text}
---"""

    def _build_vlm_instructions(self, num_targets: int, target_ids: List[int] = None) -> str:
        """构建VLM指令提示。"""
        target_id_str = ", ".join(f"#{tid}" for tid in (target_ids or range(1, num_targets + 1)))
        top_n = self.vlm_top_n

        return f"""You are an expert deep-sea benthic biologist identifying megafauna from CCZ nodule fields using the APSMA atlas.

# Task
Identify {num_targets} target(s) marked with numbered bounding boxes in the seafloor image above.
For each target, reference images from the APSMA atlas are interleaved above — compare the target with its reference images.

**CRITICAL RULES**:
1. You are ONLY evaluated on targets {target_id_str}. Do NOT describe, mention, or list any other Target IDs visible in the image that are not in this list. Ignore them completely.
2. Contextual padding may be applied around low-confidence boxes. Focus on the CENTER organism; ignore background nodules.

# Output Format
Return a JSON array with exactly {num_targets} entries (one per target listed above):
```json
[
  {{
    "target_id": 1,
    "phylum": "...",
    "class": "...",
    "order": "...",
    "morphotype": "Best candidate ID from references",
    "top{top_n}_morphotypes": ["1st_choice_ID", "2nd_choice_ID", "3rd_choice_ID"],
    "confidence": "LOW/MEDIUM/HIGH",
    "reasoning": "3-5 sentences comparing with reference images",
    "diagnostics": "image quality notes"
  }}
]
```

**top{top_n}_morphotypes**: **MANDATORY — NO "unknown" ALLOWED!** You MUST list exactly {top_n} morphotype IDs chosen from the reference candidates provided above for this target. Rank them by likelihood (most likely first). The first element MUST match "morphotype". Even if you are uncertain, pick the {top_n} closest references — guessing from references is always better than "unknown". Example: if references are [Asteroidea_Mo12, Holothuroidea_Mo96, Asteroidea_Mo20, Echinoidea_Mo05], output ["Asteroidea_Mo12", "Asteroidea_Mo20", "Holothuroidea_Mo96"]."""

    def _parse_batch_response(
        self,
        response_text: str,
        expected_target_ids: List[int] = None,
    ) -> List[Dict]:
        """容错解析VLM批量JSON响应(v3.1.1)。
        
        三级韧性:
        1. json_repair库恢复半结构化JSON
        2. 正则表达式逐目标提取回退
        3. 逐目标重试缺失目标
        """
        # Level 1: json_repair
        try:
            import json_repair
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            results = json_repair.loads(json_str.strip())
            if isinstance(results, dict):
                results = [results]
            if isinstance(results, list):
                print(f"[VLM解析] Level-1 json_repair成功, 解析出 {len(results)} 个目标")
                for r in results:
                    if isinstance(r, dict):
                        print(f"[VLM解析]   Target #{r.get('target_id','?')}: "
                              f"phylum={r.get('phylum','?')} class={r.get('class','?')} "
                              f"morphotype={r.get('morphotype','?')} "
                              f"top3={r.get('top3_morphotypes','?')} "
                              f"conf={r.get('confidence','?')}")
                return self._validate_and_fill_results(results, expected_target_ids)
        except ImportError:
            pass
        except Exception as e:
            print(f"[VLM解析] Level-1 json_repair失败: {e}")

        # Level 2: 正则提取
        results = self._regex_extract_targets(response_text)
        if results:
            print(f"[VLM解析] Level-2 正则提取成功, 解析出 {len(results)} 个目标")
            return self._validate_and_fill_results(results, expected_target_ids)

        # Level 3: 完全失败 — 触发全部期望目标重试
        print(f"[PARSE FAILED] VLM响应无法解析，触发全部目标重试")
        return self._validate_and_fill_results([], expected_target_ids)

    def _pad_top_n(self, candidates: List[str], n: int = None,
                   fallback: List[str] = None) -> List[str]:
        """确保top-N列表长度为N,不足时从fallback(检索候选)补齐,去重保序。

        去重逻辑: 同一形态型的不同ID格式视为重复,如:
          "Asteroidea_Mo12" 和 "APSMA_V1_Echinodermata_Asteroidea_Mo12" 是同一形态型
        规则: 提取 MoXX 编号或物种名作为归一化key,相同key只保留一个(优先保留短ID)
        """
        if n is None:
            n = self.vlm_top_n
        if fallback is None:
            fallback = []

        def _normalize_id(mid: str) -> str:
            """将形态型ID归一化用于去重比较。"""
            # APSMA ID: APSMA_V1_{Phylum}_{Class}_{MoXX} -> 提取 Class_MoXX
            import re
            m = re.match(r'APSMA_V\d+_\w+_(\w+_(Mo\d+))', mid)
            if m:
                return m.group(1).lower()  # e.g. "asteroidea_mo12"
            # WORMS ID: WORMS_{Phylum}_{Species} -> 提取 Species
            if mid.startswith("WORMS_"):
                parts = mid.split("_", 2)
                if len(parts) >= 3:
                    return parts[2].lower()  # e.g. "pecten_hemiradiatus"
            # 短 ID: Class_MoXX 或 SpeciesName
            return mid.lower()

        # 去重保序(基于归一化key)
        seen_keys = set()
        unique = []
        for c in candidates:
            if c == "unknown":
                continue
            key = _normalize_id(c)
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(c)
        # 从fallback(检索候选ID)补齐,跳过已存在和unknown
        for fb in fallback:
            if len(unique) >= n:
                break
            if fb == "unknown":
                continue
            key = _normalize_id(fb)
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(fb)
        # 极端情况:仍不足则补unknown(不应发生)
        while len(unique) < n:
            unique.append("unknown")
        return unique[:n]

    def _build_top3_details(self, top3_ids: List[str],
                            references: List[Dict]) -> List[Dict]:
        """为top3候选构建详细信息列表(含描述、分类、体型等)。"""
        NO_DESC = "无详细描述"
        # 建立 id -> ref 的快速查找表
        ref_map = {r["id"]: r for r in references if r.get("id")}
        details = []
        for mid in top3_ids:
            if mid == "unknown":
                details.append({"id": "unknown", "name": "unknown", "description": NO_DESC})
            elif mid in ref_map:
                ref = ref_map[mid]
                desc = ref.get("description") or NO_DESC
                details.append({
                    "id": mid,
                    "name": ref.get("name", mid),
                    "description": desc,
                    "phylum": ref.get("phylum"),
                    "class": ref.get("class"),
                    "order": ref.get("order"),
                    "size_min_cm": ref.get("size_min_cm"),
                    "size_max_cm": ref.get("size_max_cm"),
                    "substrate": ref.get("substrate"),
                })
            else:
                # VLM给的ID可能格式不完全匹配(如不带前缀),尝试模糊匹配
                matched = None
                for rid, ref in ref_map.items():
                    if mid in rid or rid in mid:
                        matched = ref
                        break
                if matched:
                    desc = matched.get("description") or NO_DESC
                    details.append({
                        "id": mid,
                        "name": matched.get("name", mid),
                        "description": desc,
                        "phylum": matched.get("phylum"),
                        "class": matched.get("class"),
                        "order": matched.get("order"),
                        "size_min_cm": matched.get("size_min_cm"),
                        "size_max_cm": matched.get("size_max_cm"),
                        "substrate": matched.get("substrate"),
                    })
                else:
                    details.append({"id": mid, "name": mid, "description": NO_DESC})
        return details

    def _regex_extract_targets(self, response_text: str) -> List[Dict]:
        """正则表达式回退提取(兼容top3_morphotypes字段)。"""
        results = []
        # 主模式: 提取各字段(含可选的top3_morphotypes)
        pattern = re.compile(
            r'"target_id"\s*:\s*(\d+).*?'
            r'"phylum"\s*:\s*"([^"]*?)".*?'
            r'"class"\s*:\s*"([^"]*?)".*?'
            r'"order"\s*:\s*"([^"]*?)".*?'
            r'"morphotype"\s*:\s*"([^"]*?)".*?'
            r'(?:'
            r'"top3_morphotypes"\s*:\s*\[([^\]]*?)\].*?'
            r')?'
            r'"confidence"\s*:\s*"([^"]*?)"',
            re.DOTALL
        )
        for match in pattern.finditer(response_text):
            top3_str = match.group(6)  # 可能为None
            top3_list = []
            if top3_str:
                top3_list = [s.strip().strip('"').strip("'")
                            for s in top3_str.split(",") if s.strip().strip('"').strip("'")]
            results.append({
                "target_id": int(match.group(1)),
                "phylum": match.group(2),
                "class": match.group(3),
                "order": match.group(4),
                "morphotype": match.group(5),
                "top3_morphotypes": top3_list if top3_list else [match.group(5)],
                "confidence": match.group(7),
                "reasoning": "extracted via regex fallback",
                "diagnostics": "partial parse"
            })
        return results

    def _validate_and_fill_results(
        self,
        results: List[Dict],
        expected_target_ids: List[int] = None,
    ) -> List[Dict]:
        """验证结果 + 重试缺失目标(v3.1.2并发重试)。"""
        if expected_target_ids is None:
            return results

        results_by_id = {}
        for r in results:
            if isinstance(r, dict) and "target_id" in r:
                # 强制target_id为int，防止VLM返回字符串"2"导致与整数2不匹配
                try:
                    tid = int(r["target_id"])
                except (TypeError, ValueError):
                    continue
                results_by_id[tid] = normalize_vlm_result(r)

        missing_ids = [tid for tid in expected_target_ids if tid not in results_by_id]

        if missing_ids:
            print(f"[WARNING] VLM批量遗漏 {len(missing_ids)} 个目标: {missing_ids}")
            retry_results = self._retry_missing_concurrently(missing_ids)
            for tid, retry_result in zip(missing_ids, retry_results):
                if retry_result:
                    results_by_id[tid] = normalize_vlm_result(retry_result)
                else:
                    results_by_id[tid] = {
                        "target_id": tid, "phylum": "unknown",
                        "class": "unknown", "order": "unknown",
                        "morphotype": "unknown", "confidence": "LOW",
                        "reasoning": "VLM dropped; fallback to YOLO",
                        "diagnostics": "batch_parse_retry_failed"
                    }

        return list(results_by_id.values())

    # ── v3.1.2: 异步重试基础设施 ──
    def _get_async_vlm_client(self):
        """懒初始化AsyncOpenAI客户端。"""
        if not hasattr(self, "_async_vlm") or self._async_vlm is None:
            try:
                from openai import AsyncOpenAI
                self._async_vlm = AsyncOpenAI(
                    base_url=getattr(self.vlm, "base_url", None),
                    api_key=getattr(self.vlm, "api_key", None),
                    max_retries=2, timeout=1800.0,
                )
            except ImportError:
                raise ImportError("AsyncOpenAI requires openai>=1.0.0")
        return self._async_vlm

    def _run_async(self, coro):
        """从同步上下文安全运行协程。"""
        result_box = {}

        def _runner():
            loop = asyncio.new_event_loop()
            try:
                result_box["value"] = loop.run_until_complete(coro)
            except Exception as e:
                result_box["error"] = e
            finally:
                loop.close()

        t = threading.Thread(target=_runner)
        t.start()
        t.join()

        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("value")

    def _retry_missing_concurrently(self, missing_ids: List[int]) -> List:
        """并发重试所有缺失目标(v3.1.2)。"""
        if not missing_ids:
            return []

        try:
            return self._run_async(
                self._retry_missing_async(missing_ids, self.max_concurrent_retries)
            )
        except Exception as e:
            print(f"[FALLBACK] 异步重试不可用 ({e}); 串行重试。")
            return [self._retry_single_target_sync(tid) for tid in missing_ids]

    async def _retry_missing_async(self, missing_ids: List[int], max_concurrency: int):
        """asyncio.gather + 信号量并发重试。"""
        sem = asyncio.Semaphore(max_concurrency)
        client = self._get_async_vlm_client()

        async def _guarded(tid):
            async with sem:
                return await self._retry_single_target_async(client, tid)

        return await asyncio.gather(*[_guarded(tid) for tid in missing_ids])

    async def _retry_single_target_async(self, client, target_id: int):
        """异步重试单个缺失目标。"""
        det = next((d for d in self._last_detections if d["target_id"] == target_id), None)
        refs = self._last_references.get(target_id, [])

        if not det or not refs or det.get("padded_crop") is None:
            return None

        top_ref = refs[0] if refs else {}
        prompt = f"""Identify this single target organism (Target #{target_id}).
Best reference match: {top_ref.get('name', 'N/A')} ({top_ref.get('phylum', '?')} > {top_ref.get('class', '?')} > {top_ref.get('order_name', '?')})

Return a single JSON object:
{{"target_id": {target_id}, "phylum": "...", "class": "...", "order": "...", "morphotype": "...", "confidence": "LOW/MEDIUM/HIGH", "reasoning": "..."}}"""

        try:
            img_b64 = self._encode_image_to_base64(det["padded_crop"])
            response = await client.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]}],
                temperature=self.vlm_temperature,
                max_tokens=512
            )
            try:
                import json_repair
                result = json_repair.loads(response.choices[0].message.content)
            except ImportError:
                result = json.loads(response.choices[0].message.content)
            if isinstance(result, list) and result:
                result = result[0]
            elif isinstance(result, list) and not result:
                return None
            return result
        except Exception as e:
            print(f"[RETRY FAILED] Target #{target_id}: {e}")
            return None

    def _retry_single_target_sync(self, target_id: int):
        """同步回退重试。"""
        det = next((d for d in self._last_detections if d["target_id"] == target_id), None)
        refs = self._last_references.get(target_id, [])

        if not det or not refs or det.get("padded_crop") is None:
            return None

        top_ref = refs[0] if refs else {}
        prompt = f"""Identify this single target organism (Target #{target_id}).
Best reference match: {top_ref.get('name', 'N/A')} ({top_ref.get('phylum', '?')} > {top_ref.get('class', '?')} > {top_ref.get('order_name', '?')})

Return a single JSON object:
{{"target_id": {target_id}, "phylum": "...", "class": "...", "order": "...", "morphotype": "...", "confidence": "LOW/MEDIUM/HIGH", "reasoning": "..."}}"""

        try:
            img_b64 = self._encode_image_to_base64(det["padded_crop"])
            response = self.vlm.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]}],
                temperature=self.vlm_temperature,
                max_tokens=512
            )
            try:
                import json_repair
                result = json_repair.loads(response.choices[0].message.content)
            except ImportError:
                result = json.loads(response.choices[0].message.content)
            if isinstance(result, list) and result:
                result = result[0]
            elif isinstance(result, list) and not result:
                return None
            return result
        except Exception as e:
            print(f"[RETRY FAILED] Target #{target_id}: {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # STEP 4: 完整管线执行
    # ═══════════════════════════════════════════════════════════
    def run(self, seafloor_image_path: str) -> List[Dict]:
        """执行完整GraphRAG管线。
        
        v3.1流程:
        1. YOLO检测 → 过滤conf < 0.2噪声
        2. 双裁剪策略: tight_crop(BioCLIP) + contextual_crop(VLM)
        3. BioCLIP嵌入 + 多视图GraphRAG检索
        4. SoM标注 → 单次VLM批量推理
        """
        image = Image.open(seafloor_image_path).convert("RGB")

        # Phase 1: 检测
        detections = self.detect_organisms(seafloor_image_path)
        if not detections:
            return []

        # Phase 2: 双裁剪 + GraphRAG检索
        all_references = {}

        for det in detections:
            # 双裁剪策略(v3.1.1)
            clip_crop = tight_crop(image, det["bbox"], shrink_ratio=0.05)
            padded_crop = contextual_crop(
                image, det["bbox"], det["confidence"],
                padding_ratio=0.25
            )
            det["padded_crop"] = padded_crop

            # v3.2: 细粒度YOLO的 class_name 是物种/属名, 用映射后的 phylum 做检索约束
            # use_yolo_hint=False 时禁用YOLO分类提示, 让GraphRAG完全依赖BioCLIP检索
            yolo_hint_phylum = det.get("phylum") if self.use_yolo_hint else None

            # 双路径并行GraphRAG检索
            enriched_refs = self.retrieve_with_graph_enrichment(
                clip_crop, coarse_class=yolo_hint_phylum, top_k=self.top_k
            )

            all_references[det["target_id"]] = enriched_refs
            det["retrieved_references"] = [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "score": r.get("score", 0),
                    "description": r.get("description"),
                    "phylum": r.get("phylum"),
                    "class": r.get("class"),
                    "order": r.get("order_name"),
                    "size_min_cm": r.get("size_min_cm"),
                    "size_max_cm": r.get("size_max_cm"),
                    "substrate": r.get("substrate"),
                }
                for r in enriched_refs
            ]

        # Phase 3: SoM标注 + 批量VLM
        annotated_image = draw_som_annotations(image, detections)

        self._last_detections = detections
        self._last_references = all_references

        batch_results = self.batch_vlm_reason(annotated_image, detections, all_references)

        # 合并VLM结果
        results_by_id = {r["target_id"]: r for r in batch_results if "target_id" in r}
        for det in detections:
            tid = det["target_id"]
            # 获取该目标的检索候选ID列表(用作后备填充)
            ref_ids = [r["id"] for r in det.get("retrieved_references", []) if r.get("id")]
            if tid in results_by_id:
                vlm_result = results_by_id[tid]
                # top3 去重 + fallback 补齐
                raw_top3 = vlm_result.get("top3_morphotypes",
                            [vlm_result.get("morphotype", "unknown")])
                padded_top3 = self._pad_top_n(raw_top3, fallback=ref_ids)
                if raw_top3 != padded_top3:
                    print(f"[Top3] Target #{tid}: VLM原始={raw_top3} → 补齐后={padded_top3}")
                det["vlm_classification"] = {
                    "phylum": vlm_result.get("phylum", "unknown"),
                    "class": vlm_result.get("class", "unknown"),
                    "order": vlm_result.get("order", "unknown"),
                    "morphotype": vlm_result.get("morphotype", "unknown"),
                    "top3_morphotypes": padded_top3,
                    "top3_details": self._build_top3_details(
                        padded_top3,
                        det.get("retrieved_references", [])),
                    "confidence": vlm_result.get("confidence", "LOW"),
                }
                det["vlm_reasoning"] = vlm_result.get("reasoning", "")
                det["diagnostics"] = vlm_result.get("diagnostics", "")
            else:
                # VLM未返回结果时的回退: 用YOLO门级映射或细粒度类名
                if self.use_yolo_hint:
                    fallback_phylum = det.get("phylum") or det.get("class_name", "unknown")
                else:
                    fallback_phylum = "unknown"
                print(f"[Top3] Target #{tid}: VLM未返回,使用YOLO回退 phylum={fallback_phylum}")
                fb_top3 = self._pad_top_n([det.get("class_name", "unknown")], fallback=ref_ids)
                det["vlm_classification"] = {
                    "phylum": fallback_phylum,
                    "class": "unknown", "order": "unknown",
                    "morphotype": det.get("class_name", "unknown"),  # 细粒度类名作为morphotype提示
                    "top3_morphotypes": fb_top3,
                    "top3_details": self._build_top3_details(
                        fb_top3,
                        det.get("retrieved_references", [])),
                    "confidence": "LOW"
                }

        return detections

    # ═══════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════
    def _encode_image_to_base64(self, image_pil: Image.Image, max_size: int = 768) -> str:
        """PIL Image → base64字符串。

        Args:
            image_pil: 输入图像
            max_size: 最长边像素上限 (默认 768, 避免 72B AWQ 模型 OOM)
        """
        # 限制图像大小, 避免 VLM 72B AWQ 模型 OOM (FathomNet 真实照片可达 1920x1080)
        if max_size and max(image_pil.size) > max_size:
            image_pil = image_pil.copy()
            image_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def load_config(config_path: str) -> Dict:
    """加载YAML配置文件。"""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DePTH-VLM GraphRAG推理管线")
    parser.add_argument("--image", required=True, help="输入海底图像路径")
    parser.add_argument("--config", default="config/graphrag_inference.yaml", help="管线配置文件")
    parser.add_argument("--output", default=None, help="结果输出JSON路径")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = DePTHGraphRAGPipeline(config)

    try:
        results = pipeline.run(args.image)

        print(f"\n=== 推理结果 ===")
        for det in results:
            vlm_cls = det.get("vlm_classification", {})
            top3 = vlm_cls.get("top3_morphotypes", [])
            print(f"  Target #{det['target_id']}: {vlm_cls.get('phylum', '?')} > "
                  f"{vlm_cls.get('class', '?')} > {vlm_cls.get('order', '?')} | "
                  f"morphotype={vlm_cls.get('morphotype', '?')} | "
                  f"top3={top3} | "
                  f"conf={vlm_cls.get('confidence', '?')}")

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            # 保存裁剪图到文件
            output_dir = os.path.dirname(args.output) or "."
            crops_dir = os.path.join(output_dir, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            for det in results:
                padded_crop = det.pop("padded_crop", None)
                if padded_crop is not None:
                    crop_path = os.path.join(crops_dir, f"target_{det['target_id']}.png")
                    try:
                        padded_crop.save(crop_path)
                        det["crop_path"] = crop_path
                    except Exception:
                        det["crop_path"] = None
                else:
                    det["crop_path"] = None
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"结果保存: {args.output}")
            print(f"裁剪图保存: {crops_dir}/")
    finally:
        pipeline.close()
