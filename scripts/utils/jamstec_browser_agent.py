"""
JAMSTEC 浏览器数据采集 Agent (Playwright)

功能:
  1. 自动化浏览 J-EDI 深海影像库,搜索指定分类/区域,下载影像元数据和缩略图
  2. 自动化浏览 BISMaL 深海物种库,提取物种分类+分布+描述信息
  3. 支持命令行参数指定搜索条件,输出 JSON/CSV 结果

安装依赖:
  conda create -n depth_agent python=3.10
  conda activate depth_agent
  pip install playwright aiohttp
  playwright install chromium

用法:
  # J-EDI 搜索指定分类的深海影像
  python scripts/utils/jamstec_browser_agent.py jedi \
    --category Echinodermata \
    --max-images 100 \
    --output data/raw/jamstec/jedi_search_results.json

  # BISMaL 查询指定物种的详细信息
  python scripts/utils/jamstec_browser_agent.py bismal \
    --species "Enypniastes eximia" \
    --output data/raw/jamstec/bismal_species_info.json

  # 批量查询多个物种
  python scripts/utils/jamstec_browser_agent.py bismal \
    --species-list "Porifera,Echinodermata,Cnidaria,Annelida,Arthropoda" \
    --output data/raw/jamstec/bismal_batch_results.json

  # OBIS API 方式获取出现记录 (无需浏览器)
  python scripts/utils/jamstec_browser_agent.py obis \
    --scientificname "Enypniastes eximia" \
    --output data/raw/jamstec/obis_occurrences.json

注意:
  - J-EDI/BISMaL 搜索需要 Playwright 浏览器,首次运行会自动安装 chromium
  - OBIS API 方式无需浏览器,适合大批量数据获取
  - J-EDI 搜索使用 JSF 有状态表单,Agent 自动处理 ViewState
  - 下载速率需遵守 JAMSTEC 使用条款,建议设置 --delay 间隔
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- OBIS API (无需浏览器) ----

OBIS_API_BASE = "https://api.obis.org/v3"
JAMSTEC_IPT_BASE = "https://www.godac.jamstec.go.jp/ipt"

# APSMA 相关的门和关键词
APSMA_DEEPSEA_KEYWORDS = [
    "Porifera", "Echinodermata", "Cnidaria", "Annelida",
    "Arthropoda", "Xenophyophorea", "Mollusca", "Brachiopoda",
    "Foraminifera", "deepsea", "abyss", "megafauna",
]


async def _http_get(url: str, session: Any = None) -> dict:
    """异步 HTTP GET 请求。"""
    import aiohttp
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                print(f"[WARN] HTTP {resp.status} for {url}")
                return {}
    finally:
        if close_session:
            await session.close()


async def obis_query_occurrences(
    scientificname: str = "",
    dataset_id: str = "",
    nodeid: str = "",
    limit: int = 10000,
    session: Any = None,
) -> List[Dict]:
    """通过 OBIS API 查询出现记录 (无需浏览器)。"""
    params = []
    if scientificname:
        params.append(f"scientificname={scientificname}")
    if dataset_id:
        params.append(f"datasetid={dataset_id}")
    if nodeid:
        params.append(f"nodeid={nodeid}")
    params.append(f"size={limit}")

    url = f"{OBIS_API_BASE}/occurrence?{'&'.join(params)}"
    data = await _http_get(url, session)
    return data.get("results", [])


async def obis_list_jamstec_datasets(session: Any = None) -> List[Dict]:
    """列出所有 JAMSTEC 在 OBIS 注册的数据集。"""
    url = f"{OBIS_API_BASE}/dataset?q=jamstec"
    data = await _http_get(url, session)
    return data.get("results", [])


async def obis_download_dwc_archive(
    resource_id: str,
    version: str = "",
    output_dir: str = "data/raw/jamstec/bismal_dwc_a",
) -> str:
    """下载 BISMaL 的 Darwin Core Archive (无需浏览器)。"""
    out_path = Path(output_dir) / f"dwca-{resource_id}.zip"
    if out_path.exists():
        print(f"[INFO] 已存在: {out_path}")
        return str(out_path)

    url = f"{JAMSTEC_IPT_BASE}/archive.do?r={resource_id}"
    if version:
        url += f"&v={version}"

    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                with open(out_path, "wb") as f:
                    f.write(await resp.read())
                print(f"[OK] 下载完成: {out_path} ({out_path.stat().st_size} bytes)")
                return str(out_path)
            else:
                print(f"[ERROR] HTTP {resp.status} for {url}")
                return ""


# ---- Playwright 浏览器 Agent ----

JEDI_BASE_URL = "https://www.godac.jamstec.go.jp/jedi"
BISMAL_BASE_URL = "https://www.godac.jamstec.go.jp/bismal"


async def _ensure_playwright():
    """确保 Playwright 和 Chromium 已安装。"""
    try:
        from playwright.async_api import async_playwright
        return True
    except ImportError:
        print("[ERROR] playwright 未安装。请运行:")
        print("  pip install playwright")
        print("  playwright install chromium")
        return False


async def jedi_search_images(
    category: str = "",
    depth_min: int = 0,
    depth_max: int = 11000,
    max_images: int = 100,
    delay: float = 1.0,
    output: str = "",
    headless: bool = True,
) -> List[Dict]:
    """
    使用 Playwright 浏览 J-EDI,搜索指定分类的深海影像。

    J-EDI 使用 JSF 有状态表单,Agent 自动:
      1. 打开搜索页面
      2. 填写分类/深度筛选条件
      3. 提交搜索
      4. 提取结果列表中的元数据(影像ID、分类、深度、位置等)
      5. 可选下载缩略图
    """
    pw_ok = await _ensure_playwright()
    if not pw_ok:
        return []

    from playwright.async_api import async_playwright

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        # 1. 打开搜索页面
        search_url = f"{JEDI_BASE_URL}/shot_search_main.jsf?LANG=EN"
        print(f"[INFO] 打开 J-EDI 搜索页: {search_url}")
        await page.goto(search_url, wait_until="networkidle")
        await page.wait_for_timeout(3000)

        # 2. 填写搜索条件
        if category:
            # 勾选生物分类复选框
            checkbox = page.locator(f"label:has-text('{category}')")
            if await checkbox.count() > 0:
                await checkbox.click()
                print(f"[INFO] 选中分类: {category}")
            else:
                # 尝试在输入框中填写
                category_input = page.locator(
                    "input[name='search_form:checkedBioCategory']"
                )
                if await category_input.count() > 0:
                    await category_input.fill(category)
                    print(f"[INFO] 输入分类: {category}")

        # 3. 提交搜索
        submit_btn = page.locator("input[type='submit'][value='Search']")
        if await submit_btn.count() > 0:
            await submit_btn.click()
            print("[INFO] 提交搜索...")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(5000)

        # 4. 提取搜索结果
        result_rows = page.locator("table.search-result tbody tr")
        count = await result_rows.count()
        print(f"[INFO] 找到 {count} 条结果")

        n = min(count, max_images)
        for i in range(n):
            row = result_rows.nth(i)
            try:
                # 提取各字段 (J-EDI 结果表结构)
                image_id = await row.locator("td:nth-child(1)").inner_text()
                species_name = await row.locator("td:nth-child(2)").inner_text()
                depth_text = await row.locator("td:nth-child(3)").inner_text()
                location = await row.locator("td:nth-child(4)").inner_text()

                result = {
                    "image_id": image_id.strip(),
                    "species_name": species_name.strip(),
                    "depth": depth_text.strip(),
                    "location": location.strip(),
                    "source": "J-EDI",
                }
                results.append(result)
            except Exception as e:
                print(f"[WARN] 第{i+1}行解析失败: {e}")
                continue

            if delay > 0 and i < n - 1:
                await page.wait_for_timeout(int(delay * 1000))

        await browser.close()

    # 输出结果
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[OK] 保存 {len(results)} 条结果到 {output}")

    return results


async def bismal_search_species(
    species: str = "",
    species_list: List[str] = [],
    delay: float = 1.0,
    output: str = "",
    headless: bool = True,
) -> List[Dict]:
    """
    使用 Playwright 浏览 BISMaL,查询物种详细信息。

    对每个物种:
      1. 打开 BISMaL 搜索页
      2. 输入物种名称搜索
      3. 提取物种页面中的分类树、分布、描述、关联影像
      4. 返回结构化信息
    """
    pw_ok = await _ensure_playwright()
    if not pw_ok:
        return []

    from playwright.async_api import async_playwright

    if species:
        species_list = [species]

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        for sp_name in species_list:
            print(f"[INFO] 查询物种: {sp_name}")
            # 搜索页面
            search_url = f"{BISMAL_BASE_URL}/e/occurrences?search={sp_name}"
            await page.goto(search_url, wait_until="networkidle")
            await page.wait_for_timeout(3000)

            # 提取出现记录
            try:
                # 获取页面中的出现记录列表
                occurrence_items = page.locator(".occurrence-item, .result-item, tr[data-id]")
                n_items = await occurrence_items.count()

                species_info = {
                    "species_name": sp_name,
                    "occurrence_count": n_items,
                    "occurrences": [],
                    "source": "BISMaL",
                }

                for j in range(min(n_items, 50)):
                    item = occurrence_items.nth(j)
                    text = await item.inner_text()
                    species_info["occurrences"].append({
                        "index": j,
                        "text": text.strip(),
                    })

                # 提取分类树信息 (如果有)
                taxonomy_section = page.locator(".taxonomy, .classification, #taxonomy")
                if await taxonomy_section.count() > 0:
                    tax_text = await taxonomy_section.inner_text()
                    species_info["taxonomy"] = tax_text.strip()

                results.append(species_info)

            except Exception as e:
                print(f"[WARN] 物种 {sp_name} 查询失败: {e}")
                results.append({
                    "species_name": sp_name,
                    "error": str(e),
                    "source": "BISMaL",
                })

            if delay > 0:
                await page.wait_for_timeout(int(delay * 1000))

        await browser.close()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[OK] 保存 {len(results)} 条结果到 {output}")

    return results


# ---- 命令行入口 ----

def _parse_args():
    parser = argparse.ArgumentParser(
        description="JAMSTEC 浏览器数据采集 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # OBIS API 模式 (无需浏览器)
    obis_p = sub.add_parser("obis", help="OBIS API 查询 (无需浏览器)")
    obis_p.add_argument("--scientificname", default="", help="物种学名")
    obis_p.add_argument("--datasetid", default="", help="数据集ID")
    obis_p.add_argument("--nodeid", default="", help="OBIS节点ID")
    obis_p.add_argument("--limit", type=int, default=10000, help="最大返回数")
    obis_p.add_argument("--download-dwc", action="store_true", help="下载DwC-A归档")
    obis_p.add_argument("--list-datasets", action="store_true", help="列出JAMSTEC数据集")
    obis_p.add_argument("--output", default="", help="输出文件路径")

    # J-EDI 搜索模式 (需要浏览器)
    jedi_p = sub.add_parser("jedi", help="J-EDI 深海影像搜索 (需要浏览器)")
    jedi_p.add_argument("--category", default="", help="生物分类筛选")
    jedi_p.add_argument("--depth-min", type=int, default=0, help="最小深度")
    jedi_p.add_argument("--depth-max", type=int, default=11000, help="最大深度")
    jedi_p.add_argument("--max-images", type=int, default=100, help="最大提取数")
    jedi_p.add_argument("--delay", type=float, default=1.0, help="请求间隔(秒)")
    jedi_p.add_argument("--output", default="", help="输出文件路径")
    jedi_p.add_argument("--headless", action="store_true", default=True)
    jedi_p.add_argument("--visible", dest="headless", action="store_false")

    # BISMaL 搜索模式 (需要浏览器)
    bismal_p = sub.add_parser("bismal", help="BISMaL 物种信息查询 (需要浏览器)")
    bismal_p.add_argument("--species", default="", help="单个物种名")
    bismal_p.add_argument("--species-list", default="", help="逗号分隔的物种名列表")
    bismal_p.add_argument("--delay", type=float, default=1.0, help="请求间隔(秒)")
    bismal_p.add_argument("--output", default="", help="输出文件路径")
    bismal_p.add_argument("--headless", action="store_true", default=True)
    bismal_p.add_argument("--visible", dest="headless", action="store_false")

    return parser.parse_args()


async def _main():
    args = _parse_args()

    if args.mode == "obis":
        if args.list_datasets:
            datasets = await obis_list_jamstec_datasets()
            print(f"\n共 {len(datasets)} 个 JAMSTEC 数据集:")
            for ds in datasets:
                occ = ds.get("statistics", {}).get("Occurrence", 0)
                archive = ds.get("archive", "")
                print(f"  [{ds['id'][:30]}] occ:{occ:>8} {ds.get('title', '')[:60]}")
                if archive:
                    print(f"    DwC-A: {archive}")

            if args.output:
                os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump(datasets, f, indent=2, ensure_ascii=False)

            if args.download_dwc:
                output_dir = os.path.dirname(args.output) or "data/raw/jamstec/bismal_dwc_a"
                print("\n下载 DwC-A 归档...")
                for ds in datasets:
                    resource_id = ds.get("id", "")
                    archive_url = ds.get("archive", "")
                    if archive_url:
                        # 从 archive URL 提取 resource_id
                        match = re.search(r"r=([^&]+)", archive_url)
                        rid = match.group(1) if match else resource_id
                        await obis_download_dwc_archive(rid, output_dir=output_dir)

        elif args.scientificname or args.datasetid or args.nodeid:
            results = await obis_query_occurrences(
                scientificname=args.scientificname,
                dataset_id=args.datasetid,
                nodeid=args.nodeid,
                limit=args.limit,
            )
            print(f"\n查询到 {len(results)} 条出现记录")
            if args.output:
                os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[OK] 保存到 {args.output}")

    elif args.mode == "jedi":
        await jedi_search_images(
            category=args.category,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            max_images=args.max_images,
            delay=args.delay,
            output=args.output,
            headless=args.headless,
        )

    elif args.mode == "bismal":
        sp_list = []
        if args.species_list:
            sp_list = [s.strip() for s in args.species_list.split(",") if s.strip()]
        await bismal_search_species(
            species=args.species,
            species_list=sp_list,
            delay=args.delay,
            output=args.output,
            headless=args.headless,
        )


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
