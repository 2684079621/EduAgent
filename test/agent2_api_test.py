#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: agent2_api_test.py
作者: ZZS
项目: 2_代码
创建日期: 2026/10/14
描述:
"""
# scripts/manual_tests/test_qa.py

import sys
import json
import httpx

sys.path.insert(0, ".")

BASE_URL  = "http://localhost:8000/api/v1"
USERNAME  = "student01"          # 或 "student01@eduagent.local"
PASSWORD  = "Student@123456"
SESSION_A = "qa-test-session-001"   # 普通测试会话
SESSION_B = "qa-test-session-002"   # 多轮记忆测试会话


def login() -> str:
    resp = httpx.post(
        f"{BASE_URL}/auth/login",
        json={"username": USERNAME, "password": PASSWORD},
        trust_env=False,   # 禁止读取系统代理，避免 macOS 代理导致 502
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print(f"[login] token 获取成功（前20字符）: {token[:20]}...")
    return token


def chat(token: str, message: str, session_id: str = SESSION_A,
         enable_web_search: bool = False) -> dict:
    resp = httpx.post(
        f"{BASE_URL}/qa/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "session_id":        session_id,
            "message":           message,
            "enable_web_search": enable_web_search,
        },
        timeout=60.0,
        trust_env=False,
    )
    resp.raise_for_status()
    return resp.json()


def stream_chat(token: str, message: str, session_id: str = SESSION_A) -> None:
    """打印 SSE 流式事件，每行一条"""
    with httpx.stream(
        "POST",
        f"{BASE_URL}/qa/chat/stream",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept":        "text/event-stream",
        },
        json={"session_id": session_id, "message": message},
        timeout=60.0,
        trust_env=False,
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                data = line[5:].strip()
                try:
                    print(json.loads(data))
                except json.JSONDecodeError:
                    print(data)


def print_result(label: str, result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  answer_mode : {result['answer_mode']}")
    print(f"  confidence  : {result['confidence']:.4f}")
    print(f"  sources     : {result['sources']}")
    print(f"  answer      : {result['answer'][:200]}...")
    print(f"{'='*60}")


# scripts/manual_tests/test_qa.py（续）

if __name__ == "__main__":
    token = login()

    # ──────────────────────────────────────────────────────────
    # 场景①  GENERAL 路径：打招呼
    # 预期：answer_mode="general"，无📚来源，不走检索
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景① GENERAL 路径")
    r = chat(token, "你好，请介绍一下你自己")
    print_result("GENERAL", r)
    assert r["answer_mode"] == "general", f"预期 general，实际 {r['answer_mode']}"
    # todo 置信度是精排后有的，没有精排就是0
    # assert r["confidence"] == 1.0
    assert "📚" not in r["answer"]
    print("✅ GENERAL 路径通过")

    # ──────────────────────────────────────────────────────────
    # 场景②  PRECISE 路径：具体技术问题
    # 预期：answer_mode="rag"，confidence≥0.75，sources 非空
    # 日志关注：classify_query.rag_strategy=PRECISE → retrieve.done
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景② PRECISE 路径")
    r = chat(token, "商品聚合大模型的双塔召回架构是什么？")
    print_result("PRECISE", r)
    assert r["answer_mode"] == "rag", f"预期 rag，实际 {r['answer_mode']}"
    assert r["confidence"] >= 0.75, f"置信度过低：{r['confidence']}"
    assert len(r["sources"]) > 0, "RAG 回答应有来源"
    assert "📚" in r["answer"]
    print("✅ PRECISE 路径通过")

    # ──────────────────────────────────────────────────────────
    # 场景③  VAGUE 路径：模糊追问（需在 PRECISE 之后同一会话）
    # 预期：日志出现 hyde_generate.done，比模糊 Query 直接检索效果更好
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景③ VAGUE 路径（模糊追问）")
    r = chat(token, "解释一下")   # 上一轮聊了双塔召回，"解释一下"指代它
    print_result("VAGUE", r)
    # VAGUE 可能高置信度（rag）也可能低置信度（llm_direct），关键看日志
    print("  ⚠️  请在后端日志确认：hyde_generate.done 被触发")
    print("✅ VAGUE 路径已发送，请核对日志")

    # ──────────────────────────────────────────────────────────
    # 场景④  BROAD 路径：宽泛问题
    # 预期：日志出现 multi_query_rewrite.done，rewritten_queries 有多条子 Query
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景④ BROAD 路径（宽泛问题）")
    r = chat(token, "微服务架构有哪些核心概念，全面总结一下")
    print_result("BROAD", r)
    print("  ⚠️  请在后端日志确认：multi_query_rewrite.done，queries 有 2-3 条")
    print("✅ BROAD 路径已发送，请核对日志")

