#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: agent_编排8.5测试.py
作者: ZZS
项目: 2_代码
创建日期: 2026/10/18
描述: 
"""
# scripts/manual_tests/uc_08_05_dispatch.py
# 仅离线验证纯工具。运行：python scripts/manual_tests/uc_08_05_dispatch.py

import sys
sys.path.insert(0, ".")

from backend.api.v1.unified_chat import _sse, UnifiedChatRequest


def section(t):
    print("\n" + "=" * 66 + f"\n  {t}\n" + "=" * 66)


def sse():
    section("① _sse：dict → SSE 数据帧（中文不转义）")
    print(_sse({"type": "routing_decision", "agent_type": "qa", "reason": "技术提问"}))
    print(_sse({"type": "done"}))


def request():
    section("② UnifiedChatRequest：非空 + 限长校验")
    ok = UnifiedChatRequest(session_id="job-prep", message="帮我准备求职")
    print("合法请求:", ok.session_id, "|", ok.message)
    try:
        UnifiedChatRequest(session_id="x", message="")   # 空 message
    except Exception as e:
        print("空 message 被拦截:", type(e).__name__)


if __name__ == "__main__":
    sse()
    request()
    section("✅ 08-05 SSE 工具 / 请求模型 离线验证通过")
