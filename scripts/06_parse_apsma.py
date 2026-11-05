"""APSMA PDF解析 — MinerU布局分析 + LLM辅助分类提取"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional


# PIL图像格式名映射(jpg→JPEG等)
_PIL_FORMAT_MAP = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "tif": "TIFF",
    "tiff": "TIFF",
    "bmp": "BMP",
    "webp": "WEBP",
}


def _normalize_pil_format(fmt: str) -> str:
    """将用户输入的图像格式名标准化为PIL接受的格式名。"""
    return _PIL_FORMAT_MAP.get(fmt.lower(), fmt.upper())


def parse_apsma_atlases(
    input_dir: str,
    output_dir: str,
    extract_images: bool = True,
    image_format: str = "png",
    image_dpi: int = 300,
    llm_api_url: str = None,
    llm_api_key: str = None,
):
    """解析APSMA图集PDF → 结构化JSONL记录。
    
    处理管线:
    1. MinerU检测布局块(文本、图表、表格、标题)
    2. 提取图表区域 → 裁剪标本图像
    3. OCR相邻文本块 → 提取分类体系 + 形态描述
    4. LLM辅助解析模糊案例
    5. Pydantic验证(06b脚本)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "apsma_images"
    if extract_images:
        images_dir.mkdir(exist_ok=True)
    
    all_records = []
    pdf_files = sorted(input_dir.glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    for pdf_path in pdf_files:
        print(f"\n解析: {pdf_path.name}")
        volume = _infer_volume(pdf_path.name)
        
        try:
            records = _parse_single_pdf(
                pdf_path, output_dir, images_dir,
                volume, extract_images, image_format, image_dpi,
                llm_api_url, llm_api_key,
            )
            all_records.extend(records)
            print(f"  提取 {len(records)} 条形态型记录")
        except Exception as e:
            print(f"  解析失败: {e}")
            continue
    
    # 保存原始JSONL
    jsonl_path = output_dir / "apsma_knowledge.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\n总计: {len(all_records)} 条APSMA形态型记录")
    print(f"输出: {jsonl_path}")
    return all_records


def _infer_volume(filename: str) -> int:
    """从文件名推断APSMA卷号。"""
    import re
    match = re.search(r'[Vv](\d+)', filename)
    if match:
        return int(match.group(1))
    return 1


def _parse_single_pdf(
    pdf_path, output_dir, images_dir, volume,
    extract_images, image_format, image_dpi,
    llm_api_url, llm_api_key,
):
    """解析单个APSMA PDF文件。"""
    # 尝试使用MinerU
    try:
        from magic_pdf.pipe.UNIPipe import UNIPipe
        records = _parse_with_mineru(
            pdf_path, images_dir, volume,
            extract_images, image_format, image_dpi,
            llm_api_url, llm_api_key,
        )
        return records
    except ImportError:
        print("  MinerU未安装，使用基础PDF解析...")
        return _parse_with_basic_pdf(pdf_path, images_dir, volume, extract_images)


def _parse_with_mineru(
    pdf_path, images_dir, volume,
    extract_images, image_format, image_dpi,
    llm_api_url, llm_api_key,
):
    """使用MinerU解析PDF。"""
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.libs.config_reader import get_local_config
    
    records = []
    
    # MinerU处理
    model_config = get_local_config()
    pipe = UNIPipe(str(pdf_path), model_config)
    pipe.pipe_classify()
    pipe.pipe_parse()
    
    content_list = pipe.pipe_mk()
    
    for page_idx, page_content in enumerate(content_list):
        # 提取页面中的文本和图像块
        text_blocks = []
        image_blocks = []
        
        for block in page_content.get("blocks", []):
            if block.get("type") == "text":
                text_blocks.append(block.get("content", ""))
            elif block.get("type") == "image":
                image_blocks.append(block)
        
        page_text = "\n".join(text_blocks)
        
        # 如果有图像块，尝试提取形态型信息
        for img_idx, img_block in enumerate(image_blocks):
            # 提取图像
            img_path = None
            if extract_images and "img" in img_block:
                img_path = images_dir / f"V{volume}_{pdf_path.stem}_p{page_idx+1}_img{img_idx}.{image_format}"
                img_block["img"].save(str(img_path), format=_normalize_pil_format(image_format), dpi=(image_dpi, image_dpi))
            
            # LLM辅助提取分类信息
            if llm_api_url and page_text:
                taxonomy_info = _llm_assisted_parse(
                    img_path, page_text, llm_api_url, llm_api_key
                )
                if taxonomy_info:
                    record = _build_apsma_record(
                        taxonomy_info, img_path, volume, page_idx + 1
                    )
                    if record:
                        records.append(record)
    
    return records


def _parse_with_basic_pdf(pdf_path, images_dir, volume, extract_images):
    """基础PDF解析回退(无需MinerU)。"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  PyMuPDF未安装，跳过PDF解析")
        return []
    
    records = []
    doc = fitz.open(str(pdf_path))
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text()
        images = page.get_images(full=True)
        
        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            
            img_path = None
            if extract_images:
                img_path = images_dir / f"V{volume}_{pdf_path.stem}_p{page_idx+1}_img{img_idx}.png"
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])
            
            # 简单文本模式匹配提取分类信息
            taxonomy_info = _simple_taxonomy_extract(text)
            if taxonomy_info:
                record = _build_apsma_record(
                    taxonomy_info, img_path, volume, page_idx + 1
                )
                if record:
                    records.append(record)
    
    doc.close()
    return records


def _llm_assisted_parse(image_path, extracted_text, api_url, api_key):
    """LLM辅助从非结构化文本+图像提取结构化分类信息。"""
    from openai import OpenAI
    
    client = OpenAI(base_url=api_url, api_key=api_key or "EMPTY")
    
    system_prompt = """You are a deep-sea taxonomy expert. Given text from a taxonomic atlas,
extract the following fields if present:

1. Full taxonomic classification (Phylum, Class, Order, Morphotype ID)
2. Morphological description (detailed, in bullet points)
3. Size measurements (format: "X-Y cm")
4. Substrate/habitat information
5. Geographic distribution (which CCZ license areas)

If any field is not found, mark as "NOT_FOUND".

Return as JSON:
{"phylum": "...", "class": "...", "order": "...", "morphotype": "...",
 "description": "...", "size_range_cm": [min, max], "substrate": "...",
 "depth_range_m": [min, max], "ccz_area": [...]}"""

    content = [{"type": "text", "text": f"OCR text:\n\n{extracted_text[:3000]}"}]
    
    if image_path and os.path.exists(image_path):
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        content.insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
    
    try:
        response = client.chat.completions.create(
            model="oceanexus-72b-awq",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  LLM解析失败: {e}")
        return None


def _simple_taxonomy_extract(text: str) -> Optional[Dict]:
    """简单文本模式匹配提取分类信息(无LLM回退)。"""
    import re
    
    info = {}
    
    # 查找门级分类
    phyla = ["Porifera", "Echinodermata", "Cnidaria", "Annelida",
             "Arthropoda", "Xenophyophorea", "Mollusca", "Brachiopoda", "Foraminifera"]
    for phylum in phyla:
        if phylum in text:
            info["phylum"] = phylum
            break
    
    # 查找尺寸信息
    size_match = re.search(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*cm', text)
    if size_match:
        info["size_range_cm"] = [float(size_match.group(1)), float(size_match.group(2))]
    
    # 查找深度信息
    depth_match = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*m', text)
    if depth_match:
        info["depth_range_m"] = [int(depth_match.group(1)), int(depth_match.group(2))]
    
    return info if info.get("phylum") else None


def _build_apsma_record(taxonomy_info, img_path, volume, page):
    """构建APSMA形态型记录。"""
    phylum = taxonomy_info.get("phylum", "NOT_FOUND")
    class_name = taxonomy_info.get("class", "NOT_FOUND")
    order = taxonomy_info.get("order", "NOT_FOUND")
    morphotype = taxonomy_info.get("morphotype", "NOT_FOUND")
    
    if phylum == "NOT_FOUND":
        return None
    
    # 生成形态型ID
    morphotype_id = f"APSMA_V{volume}_{phylum}_{class_name}_Mo{page:02d}" if class_name != "NOT_FOUND" else f"APSMA_V{volume}_{phylum}_Mo{page:02d}"
    
    return {
        "morphotype_id": morphotype_id,
        "taxonomy": {
            "phylum": phylum,
            "class": class_name,
            "order": order,
            "morphotype": morphotype,
        },
        "features": {
            "description": taxonomy_info.get("description", ""),
            "size_range_cm": taxonomy_info.get("size_range_cm", [None, None]),
            "substrate": taxonomy_info.get("substrate", ""),
            "depth_range_m": taxonomy_info.get("depth_range_m", [None, None]),
        },
        "image": {
            "path": str(img_path) if img_path else "",
            "dimensions": {"width": 0, "height": 0},
            "type": "ex_situ_specimen_photograph",
        },
        "occurrence": {
            "ccz_area": taxonomy_info.get("ccz_area", []),
            "latitude_range": [None, None],
            "longitude_range": [None, None],
        },
        "source": {
            "volume": volume,
            "page": page,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析APSMA图集PDF")
    parser.add_argument("--input-dir", default="data/raw/apsma_pdfs")
    parser.add_argument("--output-dir", default="data/processed/apsma_parsed")
    parser.add_argument("--extract-images", action=argparse.BooleanOptionalAction, default=True,
                       help="是否提取图像(默认启用，使用--no-extract-images禁用)")
    parser.add_argument("--image-format", default="png")
    parser.add_argument("--image-dpi", type=int, default=300)
    parser.add_argument("--llm-api-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    args = parser.parse_args()
    
    parse_apsma_atlases(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        extract_images=args.extract_images,
        image_format=args.image_format,
        image_dpi=args.image_dpi,
        llm_api_url=args.llm_api_url,
        llm_api_key=args.llm_api_key,
    )
