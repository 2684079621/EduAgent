#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: agent1_issues_test.py
作者: ZZS
项目: 03_代码
创建日期: 2026/10/10
描述:
"""
import asyncio
import sys
from types import SimpleNamespace
import backend.agents.resume.nodes as N
from backend.agents.resume.state import IssueItem, IssueList, ResumeSummary

class FakeThink:                       # 假 Think 模型
    async def ainvoke(self, messages):
        return SimpleNamespace(content="宏观分析：项目深度不足是核心短板",
                               text="宏观分析：项目深度不足是核心短板")

class FakeStructured:                  # 假结构化模型（可返回对象或抛错）
    def __init__(self, obj=None, fail=False): self.obj, self.fail = obj, fail
    async def ainvoke(self, messages):
        if self.fail: raise RuntimeError("模拟LLM失败")
        return self.obj

state = {
    "raw_text": "简历原文 " * 50,
    "structured": {"name": "张三", "target_position": "后端", "projects": [{"name": "电商"}]},
    "dimension_scores": [
        {"key": "project_depth", "dimension": "项目深度", "score": 60, "weight": 0.3, "issues": ["项目描述空洞"]},
        {"key": "tech_match", "dimension": "技术匹配度", "score": 70, "weight": 0.25, "issues": ["技能无层次"]},
    ],
    "weighted_score": 64.0,
}


# todo

r = asyncio.run(N.diagnose_issues_node(state))
print(r)
state["issues"] = r["issues"]
r3 = asyncio.run(N.generate_summary_node(state))
print(r3)

sys.exit(1111)

# ① diagnose 成功：故意乱序优先级，验证排序 high→medium→low
N.get_llm = lambda *a, **k: FakeThink()
mixed = IssueList(items=[
    IssueItem(priority="low",    dimension="结构",     description="证书未列",   location="末尾",       suggestion="补充"),
    IssueItem(priority="high",   dimension="项目深度", description="无量化",     location="项目-电商",  suggestion="加QPS"),
    IssueItem(priority="medium", dimension="表达",     description="非动词开头", location="项目第1句",  suggestion="改写"),
])
N.get_structured_llm = lambda a, schema: FakeStructured(obj=mixed)
r = asyncio.run(N.diagnose_issues_node(state))
print("① 排序后优先级:", [i["priority"] for i in r["issues"]])

# ② diagnose 降级：结构化 LLM 失败 → 用维度问题兜底
N.get_structured_llm = lambda a, schema: FakeStructured(fail=True)
r2 = asyncio.run(N.diagnose_issues_node(state))
print("② 降级后问题数:", len(r2["issues"]), "| 全 medium:", all(i["priority"] == "medium" for i in r2["issues"]))

# ③ generate_summary 成功
summary_obj = ResumeSummary(highlights=["有项目经历"], core_improvements=["补充量化"],
                            overall_comment="中等水平", fit_assessment="较匹配")
N.get_structured_llm = lambda a, schema: FakeStructured(obj=summary_obj)
state["issues"] = r["issues"]
r3 = asyncio.run(N.generate_summary_node(state))
print("③ summary 字段:", list(r3["summary"].keys()), "| fit:", r3["summary"]["fit_assessment"])
