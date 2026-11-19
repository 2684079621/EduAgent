# test/agent4_resp_project_closing_test.py
# 运行：python test/agent4_resp_project_closing_test.py

import asyncio

from itv_fixtures import (
    load_env, section,
    make_state_project_new, make_state_project_followup,
    make_state_closing_opening, make_state_closing_respond,
)

load_env()

from backend.agents.interview.nodes import generate_response_node


async def run(title: str, state: dict):
    print(f"\n[{title}]")
    updates = await generate_response_node(state)
    reply = updates["messages"][0].content
    print("面试官输出：")
    print("  " + reply.replace("\n", "\n  "))
    if "projects_asked" in updates:
        print(f"\n  （已深挖项目: {updates['projects_asked']}）")


async def main():
    section("① PROJECT 新项目第1问（注入简历「RAG 问答系统」+ 过渡语）")
    await run("new_project", make_state_project_new())

    section("② PROJECT 追问（基于学员对项目背景的回答继续深挖）")
    await run("followup_with_feedback", make_state_project_followup())

    section("③ CLOSING 反问邀请")
    await run("closing opening", make_state_closing_opening())

    section("④ CLOSING 回应学员提问")
    await run("closing respond", make_state_closing_respond())

    section("✅ 07-08 PROJECT & CLOSING 回应生成完成")


if __name__ == "__main__":
    asyncio.run(main())
