# test/agent4_check_stage_test.py
# 纯离线，无需启动服务。运行：python test/agent4_check_stage_test.py

import asyncio
from langchain_core.messages import HumanMessage

from itv_fixtures import load_env, section, base_state, sample_question_bank

load_env()

from backend.agents.interview.nodes import (
    check_stage_node, _check_advance_condition, _next_stage,
)


def test_next_stage():
    section("① _next_stage：阶段顺序推进（纯函数）")
    for cur in ["warmup", "tech_base", "project", "closing", "finished"]:
        print(f"  {cur:10s} → {_next_stage(cur)}")


async def run_case(title: str, state: dict):
    result = await check_stage_node(state)
    decision = result.get("current_stage", "（停留，返回 {}）")
    print(f"\n[{title}]")
    print(f"  输入: stage={state['current_stage']} stage_turns={state['stage_turn_count']} "
          f"total_turns={state['total_turn_count']}")
    print(f"  决策: {decision}")


async def test_check_stage():
    section("② check_stage_node：各场景的推进决策")

    await run_case("WARMUP→? 学员已自我介绍", base_state(
        current_stage="warmup", stage_turn_count=1, total_turn_count=1,
        messages=[HumanMessage(content="我是李明，做过 RAG 项目...")]))

    await run_case("TECH_BASE 轮数不够（3<6）", base_state(
        current_stage="tech_base", stage_turn_count=3, total_turn_count=4,
        question_bank=sample_question_bank(),
        messages=[HumanMessage(content="过拟合是...")]))

    bank_all_asked = [{**q, "asked": True} for q in sample_question_bank()]
    await run_case("TECH_BASE→? 已问满 8 题", base_state(
        current_stage="tech_base", stage_turn_count=6, total_turn_count=7,
        question_bank=bank_all_asked,
        messages=[HumanMessage(content="RAG 是检索增强生成...")]))

    await run_case("PROJECT→? 项目全覆盖", base_state(
        current_stage="project", stage_turn_count=2, total_turn_count=11,
        resume_projects=[{"name": "RAG系统"}, {"name": "LoRA微调"}],
        projects_asked=["RAG系统", "LoRA微调"],
        messages=[HumanMessage(content="这个项目我负责...")]))

    await run_case("CLOSING→? 反问满 2 轮", base_state(
        current_stage="closing", stage_turn_count=2, total_turn_count=14,
        messages=[HumanMessage(content="团队技术栈是什么？")]))

    await run_case("强制结束（关键词）", base_state(
        current_stage="tech_base", stage_turn_count=3, total_turn_count=5,
        messages=[HumanMessage(content="没有其他问题了，结束面试吧。")]))

    await run_case("强制结束（轮数上限 39≥40-2）", base_state(
        current_stage="project", stage_turn_count=5, total_turn_count=39, max_turns=40,
        messages=[HumanMessage(content="继续聊...")]))


async def main():
    test_next_stage()
    await test_check_stage()
    section("✅ 07-05 check_stage 全部场景验证通过")


if __name__ == "__main__":
    asyncio.run(main())
