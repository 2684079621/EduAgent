
# scripts/manual_tests/orch_08_02_single.py
# 纯离线，无需启动服务。运行：python scripts/manual_tests/orch_08_02_single.py

import sys
sys.path.insert(0, ".")

from backend.core.orchestrator import (
    Orchestrator, AgentType, ExecutionMode,
    AgentRequest, AgentResponse, PipelineResult,
)


def section(t):
    print("\n" + "=" * 66 + f"\n  {t}\n" + "=" * 66)


def enums():
    section("① 两个枚举")
    print("ExecutionMode:", [m.value for m in ExecutionMode])
    print("AgentType:    ", [a.value for a in AgentType])


def request():
    section("② AgentRequest：李明的模拟面试请求 + thread_id 拼接")
    req = AgentRequest(
        student_id="liming-001",
        session_id="sess-7",
        agent_type=AgentType.INTERVIEW,
        user_message="我要模拟面试",
        context={"resume_review_id": "d0000001-0000-0000-0000-000000000001"},
    )
    print("agent_type   :", req.agent_type.value)
    print("tenant_id(默认):", req.tenant_id)
    print("thread_id    :", req.thread_id)   # 由 student_id + session_id 拼出
    print("context      :", req.context)


def response_models():
    section("③ AgentResponse / PipelineResult 构造")
    resp = AgentResponse(success=True, agent_type=AgentType.QA, content="RAG 是检索增强生成...")
    print("AgentResponse:", {"success": resp.success, "agent": resp.agent_type.value,
                             "content": resp.content[:12] + "...", "fallback": resp.fallback_used})
    pr = PipelineResult(steps=[resp], combined={"step_1": {"agent_type": "qa"}})
    print("PipelineResult.steps 数:", len(pr.steps), " combined 键:", list(pr.combined.keys()))


def orchestrator_init():
    section("④ Orchestrator 初始化：Pipeline 注册表 + 懒加载缓存")
    orch = Orchestrator()
    print("初始图缓存（懒加载，应为空）:", orch._agent_graphs)
    print("预置 Pipeline:")
    for key, seq in orch._pipelines.items():
        # todo key:18s ： 效果：把 key 输出成至少 18 个字符宽的字符串，默认左对齐，右边用空格补齐。如果 key 本身超过 18 个字符，则原样输出，不会截断。
        print(f"  {key:18s} → {[a.value for a in seq]}")


if __name__ == "__main__":
    enums()
    request()
    response_models()
    orchestrator_init()
    section("✅ 08-02 枚举 / Schema / 初始化 全部离线验证通过")
