# scripts/agent4_itv_save_test.py
# 运行：python scripts/agent4_itv_save_test.py

import asyncio
import json
from sqlalchemy import text
from langchain_core.messages import HumanMessage, AIMessage

from itv_fixtures import load_env, section, base_state, setup_db_fixture

load_env()

from backend.agents.interview.nodes import save_report_node, save_memory_node
from backend.dependencies import AsyncSessionLocal

SAMPLE_REPORT = {
    "dimensions": [{"dimension": "技术深度", "score": 72, "comment": "RAG 掌握较好",
                    "highlights": ["自注意力准确"], "weaknesses": ["自回归未答出"]}],
    "overall_score": 76,
    "strengths": ["RAG 项目经验扎实"], "improvements": ["补全大模型基础原理"],
    "overall_comment": "整体达到初级 AI 开发岗要求。",
    "recommended_topics": ["自回归生成", "LoRA 原理"],
    "next_step_advice": "系统梳理基础原理后再次模拟面试。",
}


async def query_session(thread_id: str) -> dict:
    async with AsyncSessionLocal() as s:
        row = (await s.execute(
            text("SELECT status, overall_score, summary, report "
                 "FROM interview_sessions WHERE thread_id = :tid"),
            {"tid": thread_id})).mappings().fetchone()
    return dict(row) if row else {}


async def main():
    section("① 准备 DB fixture（已有一条 in_progress 会话）")
    ctx = await setup_db_fixture()
    before = await query_session(ctx["thread_id"])
    print("写入前:", {"status": before["status"], "overall_score": before["overall_score"]})

    section("② save_report_node：UPDATE 报告，status → finished")
    await save_report_node(base_state(
        student_id=ctx["student_id"], session_id=ctx["session_id"], report=SAMPLE_REPORT))
    after = await query_session(ctx["thread_id"])
    print("写入后 status / overall_score:", after["status"], "/", after["overall_score"])

    section("③ save_memory_node：触发摘要压缩 + UPSERT summary")
    long_msgs = []
    for i in range(11):
        long_msgs += [AIMessage(content=f"面试官第{i}问"), HumanMessage(content=f"学员第{i}答")]
    await save_memory_node(base_state(
        student_id=ctx["student_id"], session_id=ctx["session_id"],
        messages=long_msgs, should_summarize=True))
    final = await query_session(ctx["thread_id"])
    print("UPSERT 后 summary（前 60 字）:", (final["summary"] or "（空）")[:60])

    section("✅ 07-10 save_report / save_memory 落库验证完成")


if __name__ == "__main__":
    asyncio.run(main())
