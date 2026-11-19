# test/agent4_load_context_test.py
# 运行：python test/agent4_load_context_test.py

import asyncio
from langchain_core.messages import HumanMessage

from itv_fixtures import load_env, section, base_state, setup_db_fixture

load_env()

from backend.agents.interview.nodes import load_context_node, _generate_questions_by_llm
from backend.agents.interview.state import InterviewStage


async def main():
    section("① 准备 DB fixture：写入李明简历 + 新建面试会话")
    ctx = await setup_db_fixture()
    print("resume_review_id:", ctx["resume_review_id"])

    section("② 首轮 load_context_node（total_turn_count=0 → 走初始化路径）")
    state = base_state(
        student_id=ctx["student_id"],
        session_id=ctx["session_id"],
        resume_review_id=ctx["resume_review_id"],
        total_turn_count=0,
        messages=[HumanMessage(content="[开始面试]")],
    )
    updates = await load_context_node(state)

    print("初始化阶段:", updates.get("current_stage"))
    print("简历项目:", [p["name"] for p in updates.get("resume_projects", [])])
    print("简历技能:", updates.get("resume_skills", [])[:5], "...")
    bank = updates.get("question_bank", [])
    print(f"\n题库共 {len(bank)} 题，前 5 题：")
    for q in bank[:5]:
        print(f"  [{q['id']:>6}] ({q['difficulty']}) {q['content'][:34]}")
    print("历史摘要:", updates.get("existing_summary"))

    section("③ 单独验证 LLM 动态出题（不依赖 DB）")
    qs = await _generate_questions_by_llm("AI大模型开发工程师", count=4)
    for q in qs:
        print(f"  - ({q['difficulty']}) {q['content'][:38]}  tags={q['tags']}")

    section("✅ 07-04 load_context 首轮初始化验证完成")


if __name__ == "__main__":
    asyncio.run(main())
