#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: agent2_state_prompt_test.py
作者: ZZS
项目: 03_代码
创建日期: 2026/10/13
描述:
"""
from backend.agents.qa.state import QAState
from backend.agents.qa.prompts import (
    SYSTEM_PROMPT, RAG_STRATEGY_PROMPT, HYDE_PROMPT,
    MULTI_QUERY_REWRITE_PROMPT, RAG_ANSWER_PROMPT,
    DIRECT_ANSWER_PROMPT, GENERAL_ANSWER_PROMPT,
)
print('QAState 字段数：', len(QAState.__annotations__))
print('RAG_STRATEGY_PROMPT 占位符：', [s for s in ['query'] if '{'+s+'}' in RAG_STRATEGY_PROMPT])
print('GENERAL_ANSWER_PROMPT 占位符：', [s for s in ['current_time','query','web_context','history'] if '{'+s+'}' in GENERAL_ANSWER_PROMPT])
print('全部 import 成功')