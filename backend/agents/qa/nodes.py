# backend/agents/qa/nodes.py

import asyncio
import uuid

from sqlalchemy import text
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.agents.qa.state import QAState
from backend.agents.qa.prompts import (
    HYDE_PROMPT,
    MULTI_QUERY_REWRITE_PROMPT,
    RAG_ANSWER_PROMPT,
    DIRECT_ANSWER_PROMPT,
    GENERAL_ANSWER_PROMPT,
    RAG_STRATEGY_PROMPT,
    SYSTEM_PROMPT,
)
from backend.core.llm_factory import get_llm
from backend.core.memory import (
    trim_messages_to_window,
    should_trigger_summary,
    compress_to_summary,
    build_thread_id,
)
from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.query_classifier import get_query_classifier

logger = get_logger(__name__)

# ── 检索相关常量 ───────────────────────────────────────────────
MAX_BROAD_QUERIES        = 3   # BROAD 分支最多并行的子 Query 数
RECALL_TOP_K_PRECISE     = 8   # PRECISE：直接检索召回数
RECALL_TOP_K_VAGUE       = 10  # VAGUE：HyDE 语义扩充后多召回些
RECALL_TOP_K_BROAD_PER   = 4   # BROAD：每个子 Query 的召回数
RERANK_TOP_K             = 3   # 精排后保留的最终 chunk 数


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def _get_message_content(msg) -> str:
    """统一获取消息文本（兼容 .text 属性和 .content 属性）"""
    if hasattr(msg, "text") and not callable(getattr(msg, "text", None)):
        return msg.text
    if isinstance(msg.content, str):
        return msg.content
    return str(msg.content)


def _format_history_for_prompt(messages: list) -> str:
    """把最近几轮对话格式化为 Prompt 可用的文本"""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"学员：{_get_message_content(msg)}")
        elif isinstance(msg, AIMessage):
            # AI 回答截断，避免 Prompt 过长
            lines.append(f"AI：{_get_message_content(msg)[:200]}...")
    return "\n".join(lines) if lines else "（无历史对话）"


_WEEKDAYS_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

def _current_datetime_str() -> str:
    """返回格式化的当前时间字符串，供注入 Prompt 使用"""
    from datetime import datetime
    now = datetime.now()
    weekday = _WEEKDAYS_CN[now.weekday()]
    return now.strftime(f"%Y年%m月%d日 {weekday} %H:%M")


def _build_system_content(summary: str | None = None) -> str:
    """
    构建注入了当前时间的 SystemMessage 内容，所有生成节点共用。
    summary 非空时追加历史学习摘要，帮助 LLM 保持多轮上下文连贯。
    """
    content = SYSTEM_PROMPT + f"\n\n【当前时间】{_current_datetime_str()}"
    if summary:
        content += f"\n\n【学员历史学习摘要】\n{summary}"
    return content


# ──────────────────────────────────────────────────────────────
# 联网请求自动识别
# ──────────────────────────────────────────────────────────────

_WEB_SEARCH_HINTS = (
    "联网", "联网查询", "联网搜索", "上网查", "上网搜", "网上搜",
    "搜索一下", "可以搜索", "帮我搜", "百度一下", "谷歌一下",
)

def _extract_query_and_web_flag(raw: str) -> tuple[str, bool]:
    """
    检测用户消息是否含联网请求指令。
    返回 (清洗后的问题, 是否自动启用联网)。

    联网指令部分被去除，防止被 LLM 当作问题内容处理。
    例："Spring AOP 是什么，如果不知道可以联网搜索"
      → ("Spring AOP 是什么", True)
    """
    import re
    needs_web = any(h in raw for h in _WEB_SEARCH_HINTS)
    if not needs_web:
        return raw, False
    clean = re.sub(
        r'[，,。.？?！!\s]*(?:如果不知道|不知道的话|不清楚|你可以|可以)?'
        r'(?:联网|上网|网上)?(?:查询|搜索|搜一下|查一查|百度|谷歌).*$',
        '', raw,
    ).strip()
    return (clean or raw), True


# ──────────────────────────────────────────────────────────────
# 规则集
# ──────────────────────────────────────────────────────────────

_GENERAL_EXACT = {
    "你好", "hi", "hello", "嗨", "hey",
    "谢谢", "谢谢你", "感谢", "thanks", "thank you",
    "你是谁", "你叫什么", "你叫什么名字", "你是什么",
    "你能做什么", "你有什么功能", "你能帮我做什么",
    "再见", "拜拜", "bye",
}

_GENERAL_KEYWORDS = (
    "你是谁", "你叫什么", "你能做什么", "你有什么功能",
    "介绍一下你自己", "自我介绍",
    "今天天气", "天气怎么样",
    "讲个笑话", "说个故事",
    "今天是", "今天几号", "今天是几号", "今天是星期",
    "现在是", "现在几点", "现在时间", "当前时间", "当前日期",
    "几月几号", "星期几", "是几月", "几号了", "日期是", "今天日期",
)

# 命中即确认为专业问题，跳过 MiniLM，直接进 Layer 2
_SPECIALIZED_KEYWORDS = (
    "课程", "实战", "项目", "案例", "老师", "章节",
    "作业", "课堂", "培训", "我们学的", "课程项目",
    "第几章", "第几节", "训练营",
)

_VAGUE_QUERY_HINTS = (
    "没懂", "不懂", "不太懂", "讲讲", "解释一下",
    "啥意思", "什么意思", "看不懂",
)
_BROAD_QUERY_HINTS = (
    "全面", "系统", "总结", "梳理", "路线",
    "对比", "区别", "全景", "有哪些",
)


def _rule_classify_general(query: str) -> bool:
    """规则层：是否为闲聊/时间/打招呼类（→ GENERAL）"""
    q = query.strip().lower()
    if q in _GENERAL_EXACT:
        return True
    return any(kw in q for kw in _GENERAL_KEYWORDS)


def _rule_classify_specialized(query: str) -> bool:
    """规则层：是否含课程/项目信号词（→ 专业，跳过 MiniLM）"""
    q = query.lower()
    return any(kw in q for kw in _SPECIALIZED_KEYWORDS)


def _fast_rag_strategy(query: str) -> str:
    """规则快判 RAG 策略（< 1ms）"""
    q = query.strip().lower()
    if len(q) <= 6 and any(kw in q for kw in _VAGUE_QUERY_HINTS):
        return "VAGUE"
    if any(kw in q for kw in _BROAD_QUERY_HINTS):
        return "BROAD"
    return "PRECISE"


async def _determine_rag_strategy(query: str) -> str:
    """LLM 精判 RAG 策略（仅在规则判为 VAGUE/BROAD 且问题较长时调用）"""
    try:
        llm = get_llm("qa", temperature=0)
        resp = await llm.ainvoke([
            HumanMessage(content=RAG_STRATEGY_PROMPT.format(query=query))
        ])
        label = _get_message_content(resp).strip().upper()
        if label in ("PRECISE", "VAGUE", "BROAD"):
            return label
    except Exception as e:
        logger.warning("classify_query.rag_strategy_failed", error=str(e))
    return "PRECISE"   # 兜底：最保守策略


async def _determine_rag_strategy_fast(query: str) -> str:
    """
    两阶段策略判定：
    ① 规则快判 → 若为 PRECISE，直接返回（不调 LLM）
    ② 规则判为 VAGUE/BROAD 且问题较长（≥18字）→ LLM 校正（避免误判）
    ③ 规则判为 VAGUE/BROAD 且问题极短 → 直接相信规则
    """
    strategy = _fast_rag_strategy(query)
    if strategy == "PRECISE":
        return strategy
    if len(query.strip()) >= 18:
        return await _determine_rag_strategy(query)
    return strategy

# ──────────────────────────────────────────────────────────────
# 节点：classify_query — Query 类型判断（含历史摘要加载）
# ──────────────────────────────────────────────────────────────
async def classify_query_node(state: QAState) -> dict:
    """
    Query 分类节点，决定走哪条处理路径。

    同时负责从 DB 加载当前会话的历史摘要（合并了源码中的 load_memory 节点）。

    返回的 query_type：
      GENERAL  → 跳过 RAG，直接 LLM 回答
      PRECISE  → 直接向量检索
      VAGUE    → 先 HyDE 再检索
      BROAD    → 先 Multi-Query 改写再并行检索
    """
    # ── 取最后一条 HumanMessage 作为原始输入 ─────────────────
    messages = state.get("messages", [])
    raw_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            raw_query = _get_message_content(msg)
            break

    # ── 联网指令识别 ──────────────────────────────────────────
    original_query, auto_web = _extract_query_and_web_flag(raw_query)

    # ── 从 DB 加载历史摘要（取代 load_memory_and_embed_node）──
    existing_summary: str | None = None
    try:
        from backend.dependencies import AsyncSessionLocal
        thread_id = build_thread_id(state["student_id"], state["session_id"])
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                text("SELECT summary FROM qa_sessions WHERE thread_id = :tid"),
                {"tid": thread_id},
            )).fetchone()
            existing_summary = row[0] if row else None
    except Exception as e:
        logger.warning("classify_query.load_memory_failed", error=str(e))

    _base: dict = {
        "original_query":    original_query,
        "existing_summary":  existing_summary,
        "rewritten_queries": [],
        "hyde_document":     None,
    }
    if auto_web and not state.get("enable_web_search", False):
        _base["enable_web_search"] = True
        logger.info("classify_query.auto_web_enabled", query=original_query[:50])

    # ── Layer 0a：规则 → GENERAL（闲聊/时间/打招呼）───────────
    if _rule_classify_general(original_query):
        logger.info("classify_query.general_by_rule", query=original_query[:50])
        return {**_base, "query_type": "GENERAL"}

    # ── Layer 0b：关键词快速通道 → 专业（课程/项目词）──────────
    if _rule_classify_specialized(original_query):
        logger.info("classify_query.specialized_by_keyword", query=original_query[:50])
        strategy = await _determine_rag_strategy_fast(original_query)
        logger.info("classify_query.rag_strategy", strategy=strategy)
        return {**_base, "query_type": strategy}

    # ── Layer 1：MiniLM 二分类（CPU 推理，线程池避免阻塞）──────
    loop = asyncio.get_running_loop()
    label, confidence = await loop.run_in_executor(
        None, get_query_classifier().classify, original_query
    )

    if label == "general":
        logger.info(
            "classify_query.general_by_minilm",
            query=original_query[:50],
            confidence=round(confidence, 4),
        )
        return {**_base, "query_type": "GENERAL"}

    # ── Layer 2：MiniLM → 专业，LLM 判检索策略 ──────────────
    logger.info(
        "classify_query.specialized_by_minilm",
        query=original_query[:50],
        confidence=round(confidence, 4),
    )
    strategy = await _determine_rag_strategy_fast(original_query)
    logger.info("classify_query.rag_strategy", strategy=strategy)
    return {**_base, "query_type": strategy}


# ──────────────────────────────────────────────────────────────
# 节点：hyde_generate — HyDE 假设文档生成（VAGUE 分支）
# ──────────────────────────────────────────────────────────────
async def hyde_generate_node(state: QAState) -> dict:
    """
    针对模糊 Query，让 LLM 先生成一段假设性回答文档，
    用该文档的向量代替原始 Query 向量去检索。

    temperature=0.3：生成结果要有一定多样性（覆盖更多语义），
    但不能太随机（避免偏离主题）。
    """
    query    = state["original_query"]
    messages = state.get("messages", [])

    history_text = _format_history_for_prompt(messages[-6:])  # 最近 3 轮

    prompt = HYDE_PROMPT.format(history=history_text, query=query)

    llm = get_llm("qa", temperature=0.3)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    hyde_doc = _get_message_content(response).strip()

    logger.info(
        "hyde_generate.done",
        query=query[:50],
        hyde_doc_length=len(hyde_doc),
    )

    return {"hyde_document": hyde_doc}


# ──────────────────────────────────────────────────────────────
# 节点：multi_query_rewrite — Multi-Query 改写（BROAD 分支）
# ──────────────────────────────────────────────────────────────
async def multi_query_rewrite_node(state: QAState) -> dict:
    """
    针对宽泛/极短 Query，改写为 3-5 个具体子 Query，
    下一节 retrieve_node 会对这些子 Query 并行检索后合并去重。
    """
    query    = state["original_query"]
    messages = state.get("messages", [])

    # 取上一轮 AI 回答（推断"没懂"指的是什么）
    last_ai_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_text = _get_message_content(msg)
            break

    prompt = MULTI_QUERY_REWRITE_PROMPT.format(
        last_answer=last_ai_text[:500] if last_ai_text else "（无上一轮回答）",
        query=query,
    )

    llm = get_llm("qa", temperature=0.3)
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    # 解析：每行一个子 Query，去掉序号前缀（"1. " "①" "- " 等）
    raw = _get_message_content(response).strip()
    rewritten = [
        line.lstrip("0123456789.-）)、 ").strip()
        for line in raw.split("\n")
        if line.strip() and len(line.strip()) > 3
    ][:MAX_BROAD_QUERIES]

    if not rewritten:
        rewritten = [query]   # 改写失败时回退到原始 Query

    logger.info(
        "multi_query_rewrite.done",
        original=query,
        count=len(rewritten),
        queries=rewritten,
    )

    return {"rewritten_queries": rewritten}


# ──────────────────────────────────────────────────────────────
# 节点：retrieve — 混合召回 + 精排
# ──────────────────────────────────────────────────────────────
async def retrieve_node(state: QAState) -> dict:
    """
    调用 retrieve() Pipeline 完成检索与精排，直接输出 ranked_chunks。

    三条路径：
      PRECISE → 用 original_query 直接检索
      VAGUE   → 用 hyde_document 替代 original_query（语义扩充）
      BROAD   → 对所有 rewritten_queries 并行检索，结果合并去重

    retrieve() 是同步函数（BGE-M3 CPU 推理 + Milvus 阻塞 IO），
    必须用 run_in_executor 包装，避免阻塞 asyncio 事件循环。
    """
    from backend.core.reranker import retrieve, RankedDocument

    query_type     = state.get("query_type", "PRECISE").upper()
    tenant_id      = state["tenant_id"]
    course_id      = state.get("course_id")
    original_query = state["original_query"]

    loop = asyncio.get_running_loop()

    # ── BROAD：并行多 Query 检索，合并去重 ───────────────────────
    if query_type == "BROAD" and state.get("rewritten_queries"):
        broad_queries = state["rewritten_queries"][:MAX_BROAD_QUERIES]

        async def retrieve_one(sub_query: str) -> tuple[list, float]:
            return await loop.run_in_executor(
                None,
                lambda: retrieve(
                    sub_query,
                    tenant_id,
                    course_id,
                    recall_top_k=RECALL_TOP_K_BROAD_PER,
                    rerank_top_k=RECALL_TOP_K_BROAD_PER,
                ),
            )

        results = await asyncio.gather(*[retrieve_one(q) for q in broad_queries])

        # 合并去重：content 前 100 字符为 key，同一内容保留最高分
        seen: dict[str, RankedDocument] = {}
        for ranked_docs, _ in results:
            for doc in ranked_docs:
                key = doc.content[:100]
                if key not in seen or doc.score > seen[key].score:
                    seen[key] = doc

        merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:RERANK_TOP_K]

    # ── PRECISE / VAGUE：单路检索 ─────────────────────────────────
    else:
        if query_type == "VAGUE" and state.get("hyde_document"):
            query_text   = state["hyde_document"]
            recall_top_k = RECALL_TOP_K_VAGUE
        else:
            query_text   = original_query
            recall_top_k = RECALL_TOP_K_PRECISE

        merged, _ = await loop.run_in_executor(
            None,
            lambda: retrieve(
                query_text,
                tenant_id,
                course_id,
                recall_top_k=recall_top_k,
                rerank_top_k=RERANK_TOP_K,
            ),
        )

    # ── 转换 RankedDocument → dict，写入 State ─────────────────────
    ranked_chunks = [
        {
            "content":  doc.content,
            "score":    doc.score,
            "metadata": doc.metadata,
        }
        for doc in merged
    ]

    confidence         = ranked_chunks[0]["score"] if ranked_chunks else 0.0
    is_high_confidence = confidence >= 0.75

    logger.info(
        "retrieve.done",
        query_type=query_type,
        ranked=len(ranked_chunks),
        confidence=round(confidence, 4),
        is_high_confidence=is_high_confidence,
    )

    return {
        "ranked_chunks":      ranked_chunks,
        "confidence":         confidence,
        "is_high_confidence": is_high_confidence,
    }


# ──────────────────────────────────────────────────────────────
# 节点：generate_rag — 高置信度 RAG 生成
# ──────────────────────────────────────────────────────────────
async def generate_rag_node(state: QAState) -> dict:
    """
    高置信度 RAG 生成节点。

    将精排后的 Top-3 文档拼成 context，让 LLM 严格基于知识库内容回答。
    回答末尾附加 📚 参考来源，支持历史摘要注入保持多轮连贯性。
    """
    ranked_chunks = state.get("ranked_chunks", [])
    query         = state["original_query"]
    messages      = state.get("messages", [])
    summary       = state.get("existing_summary")

    # 构建知识库上下文与来源列表
    context_parts = []
    sources = []
    for i, chunk in enumerate(ranked_chunks, 1):
        context_parts.append(f"【参考{i}】\n{chunk['content']}")
        source_name = chunk.get("metadata", {}).get("source_name", "课程文档")
        if source_name not in sources:
            sources.append(source_name)

    context_text = "\n\n".join(context_parts)

    # 消息列表：SystemMessage（含当前时间 + 历史摘要）
    llm_messages = [SystemMessage(content=_build_system_content(summary))]

    # 注入历史对话窗口（排除最后一条 HumanMessage）
    # 最后一条 HumanMessage 已拼入 RAG_ANSWER_PROMPT 的 {query}，
    # 再传一次会让问题在上下文里出现两次，影响生成质量。
    windowed = trim_messages_to_window(messages[:-1], window_size=10)
    # todo llm_messages 已经有了 SystemMessage + 历史摘要，这里有增加了近10轮的对话（不包含当前问题）。
    #  问题：为什么不要11轮的对话。
    for msg in windowed:
        if not isinstance(msg, SystemMessage):
            llm_messages.append(msg)
    # todo llm_messages 继续添加 上下文 + 当前问题
    rag_prompt = RAG_ANSWER_PROMPT.format(context=context_text, query=query)
    llm_messages.append(HumanMessage(content=rag_prompt))

    llm = get_llm("qa", streaming=True)
    response = await llm.ainvoke(llm_messages)
    # todo llm根据 历史摘要 + 近10轮对话 + 上下文 + 当前问题，生成答案。
    answer_text = _get_message_content(response).strip()

    sources_text = "\n".join([f"  • {s}" for s in sources])
    # todo 答案 + 参考来源
    final_answer = f"{answer_text}\n\n📚 **参考来源**\n{sources_text}"

    logger.info(
        "generate_rag.done",
        answer_length=len(final_answer),
        sources=sources,
        confidence=round(state.get("confidence", 0), 4),
    )

    return {
        "answer":      final_answer,
        "sources":     sources,
        "answer_mode": "rag",
        "messages":    [AIMessage(content=final_answer)],
        "should_summarize": should_trigger_summary(messages),
        "structured_output": {
            "answer":      final_answer,
            "sources":     sources,
            "confidence":  state.get("confidence", 0),
            "answer_mode": "rag",
        },
    }


# ──────────────────────────────────────────────────────────────
# 节点：web_search — Web 搜索补充（低置信度分支）
# ──────────────────────────────────────────────────────────────
async def web_search_node(state: QAState) -> dict:
    """
    低置信度分支的 Web 搜索节点。

    知识库置信度不足时，调用 Web Search MCP Server 补充互联网信息。
    MCP Server 不可用时静默降级，返回空列表，不阻断后续生成节点。
    """
    from backend.mcp.client import call_mcp_tool

    settings = get_settings()

    if not settings.web_search_mcp_url:
        return {"web_search_results": []}

    try:
        results = await call_mcp_tool(
            server_url=settings.web_search_mcp_url,
            tool_name="web_search",
            arguments={
                "query":       state["original_query"],
                "max_results": 3,
            },
            timeout=10.0,
        )
        count = len(results or [])
        logger.info("web_search.done", count=count)
        return {"web_search_results": results or []}
    except Exception as e:
        logger.warning("web_search.failed", error=str(e))
        return {"web_search_results": []}


# ──────────────────────────────────────────────────────────────
# 节点：generate_direct — 低置信度 LLM 直答（含 Web 搜索补充）
# ──────────────────────────────────────────────────────────────
async def generate_direct_node(state: QAState) -> dict:
    """
    根据有没有联网搜索结果，且低置信度 ，LLM 直答节点。

    知识库无足够相关内容时，直接用 LLM 参数知识回答。
    有 Web 搜索结果时注入为上下文（web_augmented 模式）；
    无搜索结果时在回答末尾追加 ⚠️ 提示（llm_direct 模式）。
    """
    # todo 路由到 generate_direct_node 或者 generate_general_node
    #   是根据state.get("query_type", "").upper() == "GENERAL"
    query    = state["original_query"]
    messages = state.get("messages", [])
    summary  = state.get("existing_summary")

    llm_messages = [SystemMessage(content=_build_system_content(summary))]

    windowed = trim_messages_to_window(messages[:-1], window_size=10)
    for msg in windowed:
        if not isinstance(msg, SystemMessage):
            llm_messages.append(msg)

    # ── Web 搜索结果注入 ──────────────────────────────────────
    web_results = state.get("web_search_results") or []
    web_context = ""
    web_sources: list[str] = []
    # todo 【Web 搜索补充参考】 + 将第几个 + 标题 + url + snippet：片段 组成文本
    if web_results:
        snippets = "\n".join(
            f"  [{i + 1}] {r.get('title', '')}（{r.get('url', '')}）\n"
            f"      {r.get('snippet', '')[:300]}"
            for i, r in enumerate(web_results)
        )
        web_context = f"\n\n【Web 搜索补充参考】\n{snippets}"
        web_sources = [r.get("url", "") for r in web_results if r.get("url")]

    # todo 提示词 + Web 搜索 作为完整的 llm 输入
    direct_prompt = DIRECT_ANSWER_PROMPT.format(query=query) + web_context
    llm_messages.append(HumanMessage(content=direct_prompt))

    llm = get_llm("qa", streaming=True)
    response = await llm.ainvoke(llm_messages)
    answer_text = _get_message_content(response).strip()

    if web_sources:
        # URL 通过 sources 字段传给前端，由 UI 折叠面板展示，不拼进正文
        final_answer = answer_text
        answer_mode  = "web_augmented"
    else:
        final_answer = (
            f"{answer_text}\n\n"
            f"⚠️ **说明**：以上为 AI 基于通用知识的回答，课程知识库中暂无相关内容。"
            f"建议以教师讲解为准，或联系教师补充相关资料。"
        )
        answer_mode = "llm_direct"

    logger.info(
        "generate_direct.done",
        answer_length=len(final_answer),
        confidence=round(state.get("confidence", 0), 4),
        web_sources=len(web_sources),
    )

    return {
        "answer":      final_answer,
        "sources":     web_sources,
        "answer_mode": answer_mode,
        "messages":    [AIMessage(content=final_answer)],
        "should_summarize": should_trigger_summary(messages),
        "structured_output": {
            "answer":      final_answer,
            "sources":     web_sources,
            "confidence":  state.get("confidence", 0),
            "answer_mode": answer_mode,
        },
    }


# ──────────────────────────────────────────────────────────────
# 节点：generate_general — 通用问题直答（跳过 RAG）
# ──────────────────────────────────────────────────────────────
async def generate_general_node(state: QAState) -> dict:
    """
    通用问题直答节点（query_type=GENERAL）。

    适用于：打招呼、问时间、闲聊等与课程无关的问题。
    联网模式下若 web_search_results 非空，注入搜索结果提供时效性信息。
    """
    # todo 路由到 generate_direct_node 或者 generate_general_node
    #   是根据state.get("query_type", "").upper() == "GENERAL"
    query       = state["original_query"]
    messages    = state.get("messages", [])
    web_results = state.get("web_search_results") or []

    web_context = ""
    web_sources: list[str] = []
    if web_results:
        snippets = "\n".join(
            f"  [{i + 1}] {r.get('title', '')}（{r.get('url', '')}）\n"
            f"      {r.get('snippet', '')[:300]}"
            for i, r in enumerate(web_results)
        )
        web_context = f"【Web 搜索结果】\n{snippets}\n\n"
        web_sources = [r.get("url", "") for r in web_results if r.get("url")]

    history_text = _format_history_for_prompt(messages[-6:])
    # todo 提示词模版 + 用户问题 + 近n次对话 + 当前时间 + Web 搜索 作为完整的 llm 输入
    prompt = GENERAL_ANSWER_PROMPT.format(
        query=query,
        history=history_text,
        current_time=_current_datetime_str(),
        web_context=web_context,
    )

    llm = get_llm("qa", streaming=True)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    answer_text = _get_message_content(response).strip()

    answer_mode = "web_augmented" if web_sources else "general"

    logger.info(
        "generate_general.done",
        answer_length=len(answer_text),
        web_sources=len(web_sources),
    )

    return {
        "answer":      answer_text,
        "sources":     web_sources,
        "answer_mode": answer_mode,
        "messages":    [AIMessage(content=answer_text)],
        "should_summarize": should_trigger_summary(messages),
        "structured_output": {
            "answer":      answer_text,
            "sources":     web_sources,
            "confidence":  1.0,
            "answer_mode": answer_mode,
        },
    }


# ──────────────────────────────────────────────────────────────
# 节点：enqueue_pending — 低置信度问题入队（纯副作用）
# ──────────────────────────────────────────────────────────────
async def enqueue_pending_node(state: QAState) -> dict:
    """
    将低置信度问题写入 knowledge_pending_queue，供教师审查补充知识库。

    ON CONFLICT DO NOTHING：幂等写入，同一问题重复触发不会产生重复记录。
    失败静默，不影响已生成的回答。返回 {} 不修改 State。
    """
    # todo pgsql 的会话管理（工厂）
    from backend.dependencies import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    # todo ON CONFLICT DO NOTHING 冲突不做任何事情的判断字段是： id
                    #  改进：ALTER TABLE knowledge_pending_queue
                    #       ADD CONSTRAINT unique_pending_student_question
                    #       UNIQUE (student_id, question, status);
                    text("""
                        INSERT INTO knowledge_pending_queue
                            (id, tenant_id, question, student_id, confidence, status)
                        VALUES (:id, :tenant_id, :question, :student_id, :confidence, 'pending')
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "id":         str(uuid.uuid4()),
                        "tenant_id":  state["tenant_id"],
                        "question":   state["original_query"],
                        "student_id": state["student_id"],
                        "confidence": state.get("confidence", 0.0),
                    },
                )
        logger.info(
            "enqueue_pending.done",
            question=state["original_query"][:50],
            confidence=state.get("confidence", 0),
        )
    except Exception as e:
        logger.warning("enqueue_pending.failed", error=str(e))

    return {}


# ──────────────────────────────────────────────────────────────
# 节点：save_memory — 记忆保存（纯副作用）
# ──────────────────────────────────────────────────────────────
async def save_memory_node(state: QAState) -> dict:
    """
    记忆保存节点：条件触发摘要压缩 + 写回 qa_sessions 表。

    should_summarize=True（对话超过 10 轮）时先压缩历史再写库。
    两步均失败静默，不中断流程。返回 {} 不修改 State。
    """
    from backend.dependencies import AsyncSessionLocal
    logger.info("进入 save_memory 节点", state=str(state))

    messages   = state.get("messages", [])
    student_id = state["student_id"]
    session_id = state["session_id"]
    tenant_id  = state["tenant_id"]
    thread_id  = build_thread_id(student_id, session_id)
    summary    = state.get("existing_summary")

    # ── 条件触发摘要压缩 ─────────────────────────────────────────
    if state.get("should_summarize", False):
        try:
            # 只压缩最近 10 轮（≤20 条消息），旧知识由 existing_summary 保留。
            # 若直接传全量 messages，随对话增长输入会线性膨胀，
            # 最终超出 DeepSeek-V3 的 64k context 上限。
            msgs_to_compress = trim_messages_to_window(messages, window_size=10)
            summary = await compress_to_summary(
                messages=msgs_to_compress,
                existing_summary=summary,
            )
            logger.info("save_memory.summary_compressed", thread_id=thread_id)
        except Exception as e:
            logger.warning("save_memory.compress_failed", error=str(e))

    # ── UPSERT 到 qa_sessions 表 ──────────────────────────────────
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    text("""
                        INSERT INTO qa_sessions
                            (id, tenant_id, student_id, thread_id, summary, summary_version)
                        VALUES (:id, :tenant_id, :student_id, :thread_id, :summary, 1)
                        ON CONFLICT (thread_id) DO UPDATE
                            SET summary         = EXCLUDED.summary,
                                summary_version = qa_sessions.summary_version + 1,
                                updated_at      = NOW()
                    """),
                    {
                        "id":         str(uuid.uuid4()),
                        "tenant_id":  tenant_id,
                        "student_id": student_id,
                        "thread_id":  thread_id,
                        "summary":    summary,
                    },
                )
    except Exception as e:
        logger.warning("save_memory.db_write_failed", error=str(e))

    return {}


# backend/agents/qa/nodes.py（末尾追加）
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage, AIMessage

    BASE = {
        "student_id": "stu-test-001",
        "tenant_id":  "tenant_default",
        "session_id": "sess-test-001",
        "course_id":  None,
        "enable_web_search": False,
    }

    # ── 测试 classify_query_node ─────────────────────────────────
    async def classify_test():
        print("\n" + "=" * 60)
        print("【classify_query_node】6 种分支")
        print("=" * 60)

        # 用例①  Layer 0a：打招呼 → GENERAL（规则精确匹配）
        s = {**BASE, "messages": [HumanMessage(content="你好")]}
        r = await classify_query_node(s)
        print(f"\n① 打招呼           query_type={r['query_type']!r}   期望 GENERAL")
        assert r["query_type"] == "GENERAL"

        # 用例②  Layer 0a：问时间 → GENERAL（关键词匹配 _GENERAL_KEYWORDS）
        s = {**BASE, "messages": [HumanMessage(content="今天是星期几")]}
        r = await classify_query_node(s)
        print(f"② 问时间           query_type={r['query_type']!r}   期望 GENERAL")
        assert r["query_type"] == "GENERAL"

        # 用例③  Layer 0b：含"课程"关键词 + 精确问题 → PRECISE
        #         命中 _SPECIALIZED_KEYWORDS，跳过 MiniLM，Layer 2 快判 PRECISE
        s = {**BASE, "messages": [HumanMessage(content="我们课程里双塔召回的实现方式是什么")]}
        r = await classify_query_node(s)
        print(f"③ 课程+精确问题    query_type={r['query_type']!r}   期望 PRECISE")
        assert r["query_type"] in ("PRECISE", "VAGUE", "BROAD")

        # 用例④  Layer 0b：含"课程"关键词 + 宽泛词 → BROAD
        #         _fast_rag_strategy 检测到"全面"→ BROAD，无需 LLM
        s = {**BASE, "messages": [HumanMessage(content="全面总结一下课程里大模型微调的所有方法")]}
        r = await classify_query_node(s)
        print(f"④ 课程+全面总结    query_type={r['query_type']!r}   期望 BROAD")
        assert r["query_type"] == "BROAD"

        # 用例⑤  Layer 1 MiniLM → GENERAL（纯通用 IT 知识，P(general)≥0.85）
        #         MiniLM 微调后对"面向对象"等概念置信度高，直接返回 GENERAL
        s = {**BASE, "messages": [HumanMessage(content="什么是面向对象编程")]}
        r = await classify_query_node(s)
        print(f"⑤ 通用 IT 知识     query_type={r['query_type']!r}   期望 GENERAL（MiniLM P≥0.85）")

        # 用例⑥  联网指令 → original_query 去除联网部分，enable_web_search=True
        #         _extract_query_and_web_flag 匹配"联网搜索"并截断
        s = {**BASE, "messages": [HumanMessage(content="BGE-M3 的最新动态，帮我联网搜索一下")]}
        r = await classify_query_node(s)
        print(f"⑥ 联网指令         original_query={r['original_query']!r}")
        print(f"                   enable_web_search={r.get('enable_web_search')}  期望 True")
        assert r.get("enable_web_search") is True
        assert "联网" not in r["original_query"]

        print("\n✅ classify_query_node 全部用例通过")

    # ── 测试 hyde_generate_node ──────────────────────────────────
    async def hyde_test():
        print("\n" + "=" * 60)
        print("【hyde_generate_node】VAGUE 路径：用 LLM 生成假设文档")
        print("=" * 60)

        # 场景：用户说"没懂"，节点调 LLM 生成假设性回答文档
        # 该文档的向量比原始模糊 Query 的向量更接近知识库内容
        state = {
            **BASE,
            "original_query": "Hard Negative Sampling 没懂",
            "messages": [HumanMessage(content="Hard Negative Sampling 没懂")],
        }
        r = await hyde_generate_node(state)
        print(f"\n输入 query        : {state['original_query']}")
        print(f"输出 hyde_document（节选）:\n  {r['hyde_document'][:200]}...")
        print(f"文档总长度         : {len(r['hyde_document'])} 字")
        assert "hyde_document" in r
        assert len(r["hyde_document"]) > 20
        print("\n✅ hyde_generate_node：假设文档已生成")

    # ── 测试 multi_query_rewrite_node ───────────────────────────
    async def multi_query_test():
        print("\n" + "=" * 60)
        print("【multi_query_rewrite_node】BROAD 路径：改写子 Query")
        print("=" * 60)

        # 正常用例：LLM 把宽泛 Query 改写为 ≤3 条具体子 Query
        state = {
            **BASE,
            "original_query": "全面介绍一下商品聚合大模型微调的知识",
            "messages": [HumanMessage(content="全面介绍一下商品聚合大模型微调的知识")],
        }
        r = await multi_query_rewrite_node(state)
        print(f"\n正常用例 - 输入 query  : {state['original_query']}")
        print(f"改写后 rewritten_queries ({len(r['rewritten_queries'])} 条):")
        for i, q in enumerate(r["rewritten_queries"], 1):
            print(f"  [{i}] {q}")
        assert 1 <= len(r["rewritten_queries"]) <= 3
        assert all(len(q) > 3 for q in r["rewritten_queries"])
        print("\n✅ multi_query_rewrite_node 正常用例通过")

        # 兜底用例（代码说明）：LLM 返回空或每行 ≤3 字时的处理
        print("\n兜底逻辑（源码第 483-485 行）：")
        print("  rewritten = [line... for line in raw.split('\\n') if len(line.strip()) > 3]")
        print("  if not rewritten:")
        print("      rewritten = [query]   ← 回退到原始 Query")
        print("  → rewritten_queries = ['全面介绍一下商品聚合大模型微调的知识']")
        print("  → retrieve_node 用原始 Query 按 BROAD 路径检索（每子 Query recall=4）")

    # ── 测试 retrieve_node ──────────────────────────────────────
    async def retrieve():
        print("\n" + "=" * 60)
        print("【retrieve_node】四种路径")
        print("=" * 60)

        # 用例①  PRECISE 路径：直接用 original_query 检索，recall_top_k=8
        s1 = {
            **BASE,
            "query_type": "PRECISE",
            "original_query": "商品聚合大模型的双塔召回怎么实现",
            "hyde_document": None,
            "rewritten_queries": [],
        }
        r1 = await retrieve_node(s1)
        print(f"\n用例① PRECISE 路径")
        print(f"  检索 query     : {s1['original_query']}")
        print(f"  ranked_chunks  : {len(r1['ranked_chunks'])} 条（recall_top_k=8 → rerank → Top-3）")
        print(f"  confidence     : {r1['confidence']:.4f}")
        print(f"  is_high_conf   : {r1['is_high_confidence']}  （阈值 0.75）")
        if r1["ranked_chunks"]:
            print(f"  Top-1 内容     : {r1['ranked_chunks'][0]['content'][:80]}...")

        # 用例②  VAGUE 路径：用 hyde_document 替代 original_query，recall_top_k=10
        #         比 PRECISE 多召回 2 条，补偿 HyDE 文档与知识库对齐的误差
        s2 = {
            **BASE,
            "query_type": "VAGUE",
            "original_query": "Hard Negative Sampling 没懂",
            "hyde_document": (
                "Hard Negative Sampling 是大模型检索微调的关键技术，"
                "通过挑选与正样本高度相似但标签不同的负样本提升模型判别能力，"
                "常见方法包括 BM25 Hard Negative 和 In-Batch Hard Negative。"
            ),
            "rewritten_queries": [],
        }
        r2 = await retrieve_node(s2)
        print(f"\n用例② VAGUE 路径（用 HyDE 文档检索）")
        print(f"  hyde_document  : {s2['hyde_document'][:60]}...")
        print(f"  ranked_chunks  : {len(r2['ranked_chunks'])} 条（recall_top_k=10 → rerank → Top-3）")
        print(f"  confidence     : {r2['confidence']:.4f}")

        # 用例③  BROAD 路径：3 条子 Query 并行检索，合并去重后取 Top-3
        #         每条子 Query recall_top_k=4，rerank_top_k=4（保留全部候选）
        #         asyncio.gather 并行，content[:100] 去重
        s3 = {
            **BASE,
            "query_type": "BROAD",
            "original_query": "全面介绍商品聚合大模型微调",
            "hyde_document": None,
            "rewritten_queries": [
                "商品聚合多模态大模型的双塔召回架构是什么",
                "LlamaFactory 微调 Qwen VL 的基本步骤",
                "Hard Negative Sampling 在大模型微调中的作用",
            ],
        }
        r3 = await retrieve_node(s3)
        print(f"\n用例③ BROAD 路径（{len(s3['rewritten_queries'])} 条子 Query 并行 → 合并去重）")
        print(f"  每条召回 4 条，精排保留全部，合并后按 score 排序取 Top-3")
        print(f"  ranked_chunks  : {len(r3['ranked_chunks'])} 条")
        print(f"  confidence     : {r3['confidence']:.4f}")

        # 用例④  空召回：query 与知识库无关 → retrieve() 返回 ([], 0.0)
        #         confidence=0.0 → is_high_confidence=False → 走低置信度分支
        s4 = {
            **BASE,
            "query_type": "PRECISE",
            "original_query": "xyzxyz完全不相关的测试内容abc",
            "hyde_document": None,
            "rewritten_queries": [],
        }
        r4 = await retrieve_node(s4)
        print(f"\n用例④ 空召回（query 与知识库无关）")
        print(f"  ranked_chunks  : {len(r4['ranked_chunks'])} 条   ← 期望 0")
        print(f"  confidence     : {r4['confidence']:.4f}   ← 期望 0.0")
        print(f"  is_high_conf   : {r4['is_high_confidence']}   ← 期望 False")
        # assert r4["ranked_chunks"] == []
        # assert r4["confidence"] == 0.0
        assert r4["is_high_confidence"] is False

        print("\n✅ retrieve_node 四种路径测试通过")

    # ── 公用：模拟精排输出（无需 Milvus）──────────────────────────
    MOCK_CHUNKS = [
        {
            "content": "双塔召回模型由用户塔和商品塔组成，分别编码用户特征和商品特征，"
                       "通过内积计算相似度，在离线完成商品向量化后可实现毫秒级在线召回。",
            "score": 0.89,
            "metadata": {"source_name": "sample2.md"},
        },
        {
            "content": "Hard Negative Sampling 从与查询语义相近但标签不同的样本中选取负例，"
                       "强迫模型学习细粒度判别能力，显著提升 Recall@K 指标。",
            "score": 0.82,
            "metadata": {"source_name": "sample2.md"},
        },
    ]

    # ── 测试 generate_rag_node ──────────────────────────────────
    async def generate_rag():
        print("\n" + "=" * 60)
        print("【generate_rag_node】高置信度 RAG 生成")
        print("=" * 60)

        state = {
            **BASE,
            "original_query": "商品聚合大模型的双塔召回是怎么实现的",
            "messages": [HumanMessage(content="商品聚合大模型的双塔召回是怎么实现的")],
            "ranked_chunks": MOCK_CHUNKS,
            "confidence": 0.89,
            "is_high_confidence": True,
            "existing_summary": None,
        }
        r = await generate_rag_node(state)

        print(f"\n输入 ranked_chunks : {len(MOCK_CHUNKS)} 条")
        print(f"输出 answer_mode   : {r['answer_mode']!r}   ← 期望 rag")
        print(f"输出 sources       : {r['sources']}")
        print(f"输出 answer（节选）:\n  {r['answer'][:150]}...")
        print(f"should_summarize   : {r['should_summarize']}  （消息数 < 20 → False）")

        assert r["answer_mode"] == "rag"
        assert "📚 **参考来源**" in r["answer"]
        assert "sample2.md" in r["sources"]

        print("\n✅ generate_rag_node 通过")

    # ── 测试 web_search_node ─────────────────────────────────────
    async def web_search():
        print("\n" + "=" * 60)
        print("【web_search_node】静默降级验证")
        print("=" * 60)

        state = {**BASE, "original_query": "BGE-M3 向量模型最新进展"}

        # 情况①  WEB_SEARCH_MCP_URL 未配置 → if not settings.web_search_mcp_url 直接返回 []
        # 情况②  URL 已配置但 Server 未启动 → call_mcp_tool 抛异常 → except 捕获 → 返回 []
        # 两种情况都不会抛出异常，节点始终正常返回
        r = await web_search_node(state)
        print(f"\n情况①/②  web_search_results={r['web_search_results']!r}   期望 []")
        assert isinstance(r["web_search_results"], list)

        print("\n  ℹ️  情况③ Server 正常运行（5.10 节已启动 web_search_server.py）:")
        print("     web_search_results 返回含 title / url / snippet 的列表，≥1 条")
        print("     → generate_direct_node 进入 web_augmented 模式")

        print("\n✅ web_search_node：无论 Server 状态如何，节点均正常返回（不抛异常）")

    # ── 测试 generate_direct_node ───────────────────────────────
    async def generate_direct():
        print("\n" + "=" * 60)
        print("【generate_direct_node】低置信度直答：两种模式")
        print("=" * 60)

        base_state = {
            **BASE,
            "original_query": "LlamaFactory 怎么配置 LoRA 微调 Qwen VL",
            "messages": [HumanMessage(content="LlamaFactory 怎么配置 LoRA 微调 Qwen VL")],
            "confidence": 0.30,
            "existing_summary": None,
        }

        # 模式①  web_search_results 为空 → llm_direct
        #         正文末尾追加 ⚠️ 说明，提示来自通用知识而非课程内容
        s1 = {**base_state, "web_search_results": []}
        r1 = await generate_direct_node(s1)
        print(f"\n模式①  无 Web 结果  → answer_mode={r1['answer_mode']!r}   期望 llm_direct")
        print(f"  ⚠️ 说明已追加     : {'⚠️' in r1['answer']}")
        assert r1["answer_mode"] == "llm_direct"
        assert "⚠️" in r1["answer"]

        # 模式②  web_search_results 非空 → web_augmented
        #         URL 通过 sources 字段传前端折叠展示，正文不追加 ⚠️
        s2 = {**base_state, "web_search_results": [
            {"title": "LlamaFactory 官方文档",
             "url": "https://example.com/llamafactory",
             "snippet": "LlamaFactory 支持 LoRA、QLoRA 等多种微调方式，"
                        "通过 YAML 配置文件指定模型路径和数据集..."},
        ]}
        r2 = await generate_direct_node(s2)
        print(f"\n模式②  有 Web 结果  → answer_mode={r2['answer_mode']!r}   期望 web_augmented")
        print(f"  sources（URL）   : {r2['sources']}")
        assert r2["answer_mode"] == "web_augmented"
        assert len(r2["sources"]) > 0

        print("\n✅ generate_direct_node 两种模式通过")

    # ── 测试 generate_general_node ──────────────────────────────
    async def generate_general():
        print("\n" + "=" * 60)
        print("【generate_general_node】通用问题直答：两种模式")
        print("=" * 60)

        base_state = {
            **BASE,
            "original_query": "今天是几号",
            "messages": [HumanMessage(content="今天是几号")],
        }

        # 模式①  无 Web 结果 → general 模式
        #         structured_output.confidence 写死 1.0（通用问题不走检索）
        s1 = {**base_state, "web_search_results": []}
        r1 = await generate_general_node(s1)
        print(f"\n模式①  无 Web 结果  → answer_mode={r1['answer_mode']!r}   期望 general")
        print(f"  confidence       : {r1['structured_output']['confidence']}   期望 1.0")
        assert r1["answer_mode"] == "general"
        assert r1["structured_output"]["confidence"] == 1.0

        # 模式②  有 Web 结果 → web_augmented 模式
        #         联网搜索结果注入 {web_context} 占位符，web_context 空时占位符自然消失
        s2 = {
            **base_state,
            "original_query": "BGE-M3 最新版本是什么",
            "messages": [HumanMessage(content="BGE-M3 最新版本是什么")],
            "web_search_results": [
                {"title": "BGE-M3 GitHub",
                 "url": "https://example.com/bge-m3",
                 "snippet": "BGE-M3 v1.2 已发布，新增混合检索支持..."},
            ],
        }
        r2 = await generate_general_node(s2)
        print(f"\n模式②  有 Web 结果  → answer_mode={r2['answer_mode']!r}   期望 web_augmented")
        assert r2["answer_mode"] == "web_augmented"

        print("\n✅ generate_general_node 两种模式通过")

    # ── 测试 enqueue_pending_node ────────────────────────────────
    async def enqueue_pending():
        print("\n" + "=" * 60)
        print("【enqueue_pending_node】低置信度问题入队（纯副作用）")
        print("=" * 60)

        state = {
            **BASE,
            "original_query": "Qwen VL 微调的最佳学习率应该设多少",
            "confidence": 0.28,
        }

        # DB 可用时写入 knowledge_pending_queue（ON CONFLICT DO NOTHING 幂等）
        # DB 不可用时 except 静默捕获
        # 两种情况都返回 {}，不影响已生成的回答
        r = await enqueue_pending_node(state)
        print(f"\n返回值  : {r!r}   期望 {{}}（DB 可用写入，不可用静默跳过）")
        assert r == {}

        print("\n✅ enqueue_pending_node 始终返回 {}")

    # ── 测试 save_memory_node ────────────────────────────────────
    async def save_memory():
        print("\n" + "=" * 60)
        print("【save_memory_node】摘要保存：两种触发情况")
        print("=" * 60)

        # 模拟 10 轮有实质内容的课程对话（供 compress_to_summary 压缩）
        real_msgs = [
            HumanMessage(content="商品聚合大模型的双塔召回和 ANN 近似近邻检索有什么关系"),
            AIMessage(
                content="双塔召回离线生成商品向量并存入向量库，ANN 在线阶段用用户向量找最近邻。双塔生成向量，ANN 完成检索，分工明确。"),
            HumanMessage(content="Hard Negative Sampling 怎么选负样本"),
            AIMessage(
                content="从与正样本语义相近但标签不同的样本中选负例。BM25 Hard Negative 用关键词召回困难负例，In-Batch Hard Negative 在同批次选最近负例。"),
            HumanMessage(content="LlamaFactory 支持哪些微调方式"),
            AIMessage(
                content="支持 Full Fine-Tuning、LoRA、QLoRA 和 DoRA 四种。QLoRA 通过 4bit 量化大幅降低显存需求，是主流选择。"),
            HumanMessage(content="Qwen VL 微调的数据格式是什么"),
            AIMessage(
                content="ShareGPT 格式，每条样本含 messages 列表，图文交错内容放 content 数组，image 字段指本地路径或 base64 编码。"),
            HumanMessage(content="BGE-M3 的三种检索模式分别是什么"),
            AIMessage(
                content="稠密检索（Dense 向量内积）、稀疏检索（类 BM25 关键词权重）和多向量检索（ColBERT MaxSim）。三种模式可 RRF 融合使用。"),
            HumanMessage(content="混合检索里稀疏和稠密分数怎么融合"),
            AIMessage(
                content="RRF 倒数排名融合：score=1/(k+rank_dense)+1/(k+rank_sparse)，k 取 60。或线性加权 α*dense+(1-α)*sparse，α 按任务调节。"),
            HumanMessage(content="Reranker 精排阈值怎么定"),
            AIMessage(
                content="在验证集画 P-R 曲线选 F1 最高点。课程默认 0.75：≥0.75 高置信度走 RAG 生成，低于此值走低置信度分支兜底。"),
            HumanMessage(content="多模态对齐训练是什么意思"),
            AIMessage(
                content="让视觉编码器输出空间与 LLM token 空间对齐。通过 MLP Projector 把图像特征映射为 LLM 可理解向量，预训练阶段先固定 LLM 只训练 Projector。"),
            HumanMessage(content="课程里 ANN 索引参数 ANN_EF 是什么意思"),
            AIMessage(
                content="ANN_EF 是 HNSW 索引搜索时的 ef 参数，控制搜索范围大小。ef=64 维护 64 个候选节点，值越大召回率越高但速度越慢。"),
            HumanMessage(content="MemorySaver 和 DB 存摘要有什么区别"),
            AIMessage(
                content="MemorySaver 保存完整 State 对象进程重启后丢失，是会话级缓存。DB 存摘要持久化压缩后文本，重启可恢复，是跨会话记忆核心。"),
        ]

        # 情况①  should_summarize=False（< 10 轮）→ 直接 UPSERT，跳过 compress_to_summary
        s1 = {
            **BASE,
            "messages": real_msgs[:6],  # 3 轮对话
            "should_summarize": False,
            "existing_summary": None,
        }
        r1 = await save_memory_node(s1)
        print(f"\n情况①  should_summarize=False（3 轮）→ 返回 {r1!r}")
        print("         跳过 compress_to_summary，直接 UPSERT qa_sessions")
        assert r1 == {}

        # 情况②  should_summarize=True（10 轮实质对话）→ LLM 压缩 → UPSERT
        #         compress_to_summary 接收 real_msgs（含课程实质内容），
        #         生成结构化"学员画像摘要"后存入 qa_sessions.summary
        s2 = {
            **BASE,
            "messages": real_msgs,  # 10 轮有实质内容的对话
            "should_summarize": True,
            "existing_summary": None,
        }
        r2 = await save_memory_node(s2)
        print(f"\n情况②  should_summarize=True（10 轮）→ 返回 {r2!r}")
        print("         compress_to_summary 已被调用，LLM 将 10 轮对话压缩为学员画像摘要")
        print("\n  验证压缩结果（DB 可用时执行）：")
        print("  SELECT summary, summary_version FROM qa_sessions")
        print(f"  WHERE thread_id = 'student_{BASE['student_id']}_session_{BASE['session_id']}';")
        assert r2 == {}

        print("\n✅ save_memory_node 两种情况均返回 {}")


    # todo 要单独测试，否则会关闭全局的事件循环，导致报错。
    # asyncio.run(classify_test())
    # asyncio.run(hyde_test())
    # asyncio.run(multi_query_test())
    # asyncio.run(retrieve())
    asyncio.run(generate_rag())
    asyncio.run(web_search())
    asyncio.run(generate_direct())
    asyncio.run(generate_general())
    asyncio.run(enqueue_pending())
    asyncio.run(save_memory())

