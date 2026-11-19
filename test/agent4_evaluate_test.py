# test/agent4_evaluate_test.py
# 运行：python test/agent4_evaluate_test.py

import asyncio
from langchain_core.messages import HumanMessage

from itv_fixtures import (
    load_env, section, make_state_eval_excellent, make_state_eval_no_answer,
    base_state, ANSWER_SCRIPT,
)

load_env()

from backend.agents.interview.nodes import evaluate_answer_node


async def run(title: str, state: dict):
    cur_q  = (state.get("current_question") or {}).get("content", "（无）")
    answer = state["messages"][-1].content
    print(f"\n[{title}]")
    print(f"  问题: {cur_q}")
    print(f"  学员回答: {answer[:50]}{'...' if len(answer) > 50 else ''}")
    updates = await evaluate_answer_node(state)
    print(f"  → 质量标签: {updates['last_answer_quality']}")
    print(f"  → 计数更新: total_turns={updates['total_turn_count']} "
          f"stage_turns={updates['stage_turn_count']}")


async def main():
    section("① 优秀回答（Transformer）→ Think Tool 推理 + LLM 评估")
    await run("有技术细节的回答", make_state_eval_excellent())

    section("② 明确不会（'不知道'）→ 快速路径，不调用 LLM")
    await run("no_answer 快速路径", make_state_eval_no_answer())

    section("③ WARMUP 阶段 → 跳过评估，直接 ADEQUATE")
    await run("warmup 跳过", base_state(
        current_stage="warmup", stage_turn_count=1, total_turn_count=1,
        messages=[HumanMessage(content=ANSWER_SCRIPT[0])]))

    section("✅ 07-06 evaluate_answer 三条路径验证完成")


if __name__ == "__main__":
    asyncio.run(main())
