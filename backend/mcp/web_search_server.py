#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: web_search_server.py
作者: ZZS
项目: 2_代码
创建日期: 2026/7/20
描述: 
"""
# backend/mcp/web_search_server.py

import asyncio
import os
import sys

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="EduAgent-WebSearch",
    stateless_http=True,
    json_response=True,
)


# ── 搜索后端 ──────────────────────────────────────────────────────
async def _search_tavily(query: str, max_results: int, api_key: str) -> list[dict]:
    """Tavily 搜索（结构化结果，需 API key，https://tavily.com）"""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key":             api_key,
                "query":               query,
                "max_results":         max_results,
                "include_answer":      False,
                "include_raw_content": False,
            },
        )
        resp.raise_for_status()
    return [
        {
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("content", "")[:500],
            "content": r.get("content", ""),
        }
        for r in resp.json().get("results", [])
    ]

# todo  duckduckgo 需要魔法。 改成博查（需要key）
async def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo 搜索（免费，无需 API key）

    duckduckgo-search 6.x+ 移除了 AsyncDDGS，统一使用同步 DDGS。
    用 asyncio.to_thread 包装避免阻塞事件循环（与 run_in_executor 效果相同，写法更简洁）。
    """
    # from duckduckgo_search import DDGS
    from ddgs import DDGS


    def _sync_search() -> list[dict]:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", "")[:500],
                    "content": r.get("body", ""),
                })
        return results

    return await asyncio.to_thread(_sync_search)

# todo 新增 博查 联网搜索，需要注册key
async def _search_bocha(query: str, max_results: int) -> list[dict]:
    """博查 Bocha 搜索（需 API Key）

    官方文档：https://open.bochaai.com/
    API 端点：https://api.bochaai.com/v1/web-search
    """
    api_key = os.environ.get("BOCHA_API_KEY", "")
    if not api_key:
        raise RuntimeError("BOCHA_API_KEY 未配置，请在环境变量中设置")

    endpoint = "https://api.bochaai.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "summary": True,      # 开启详细摘要
        "freshness": "noLimit",
        "count": max_results,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        raw = response.json()

    # 兼容两种可能的返回结构
    results = []
    data = raw.get("data", {})
    # Bocha Web Search 返回格式兼容 Bing API，结果在 data.webPages.value
    web_pages = data.get("webPages", {}).get("value", [])

    for r in web_pages:
        results.append({
            "title":   r.get("name", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("snippet", ""),
            "content": r.get("summary", "") or r.get("snippet", ""),  # 优先用长摘要
        })
    return results


# ── MCP 工具 ──────────────────────────────────────────────────────

@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Search the web for current information not available in the knowledge base.

    Automatically selects the best available backend:
    - Tavily (preferred): structured results; requires TAVILY_API_KEY in .env.local
    - DuckDuckGo (fallback): free, no API key required

    Args:
        query:       Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of search results, each with title / url / snippet / content
    """
    from backend.config import get_settings
    from backend.core.logger import get_logger

    logger = get_logger(__name__)
    settings = get_settings()

    # 优先 Tavily，失败后降级 DuckDuckGo
    if settings.tavily_api_key:
        try:
            results = await _search_tavily(query, max_results, settings.tavily_api_key)
            logger.info("web_search_mcp.tavily_done", hits=len(results))
            return results
        except Exception as e:
            logger.warning("web_search_mcp.tavily_failed", error=str(e))

    try:
        # todo 将原来的 duckduckgo 改成 bocha， duckduckgo 需要魔法。
        # results = await _search_duckduckgo(query, max_results)
        results = await _search_bocha(query, max_results)
        logger.info("web_search_mcp.ddgs_done", hits=len(results))
        return results
    except Exception as e:
        logger.error("web_search_mcp.ddgs_failed", error=str(e))
        return []


# ── 独立运行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    sys.path.insert(0, str(__file__).split("/backend/")[0])
    load_dotenv(".env.local")

    port = int(os.getenv("WEB_SEARCH_MCP_SERVER_PORT", "8002"))
    print(f"Web Search MCP Server → http://localhost:{port}/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)
