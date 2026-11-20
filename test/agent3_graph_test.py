#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: agent3_图模拟测试.py
作者: ZZS
项目: 03_代码
创建日期: 2026/10/16
描述:
"""

# 在项目根目录下运行：
# conda activate edu_agent
# python test/agent3_graph_test.py

import sys, asyncio, uuid
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(".env.local")

from sqlalchemy import text
from langgraph.types import Command
from backend.agents.exam.graph import build_exam_graph
from backend.dependencies import AsyncSessionLocal
from backend.core.logger import get_logger
logger = get_logger(__name__)

graph    = build_exam_graph()
EXAM_ID  = "e0000001-0000-0000-0000-000000000001"


async def main():
    # ── 测试准备：获取真实 student UUID，写入临时提交记录 ────────
    # 原因①：exam_reviews.submission_id 有 FK 约束，必须先在
    #         exam_submissions 中存在对应记录，否则 INSERT 报 ForeignKeyViolationError
    # 原因②：exam_submissions.student_id 有 FK 约束，必须用
    #         users 表中真实存在的 UUID，不能传 "test-student" 这种字符串
    async with AsyncSessionLocal() as session:
        row = await session.execute(
            text("SELECT id FROM users WHERE username = 'student01' LIMIT 1")
        )
        student_uuid = str(row.scalar())

    logger.info("获取真实 student UUID", uuid=student_uuid)

    submission_id = str(uuid.uuid4())
    logger.info("获取真实 submission_id", submission_id=submission_id)

    async with AsyncSessionLocal() as session:
        async with session.begin():
            # 清除旧测试记录，避免 UNIQUE(exam_id, student_id) 重跑冲突
            await session.execute(
                text("""
                    DELETE FROM exam_reviews
                    WHERE submission_id IN (
                        SELECT id FROM exam_submissions
                        WHERE exam_id = :exam_id AND student_id = :student_id
                    )
                """),
                {"exam_id": EXAM_ID, "student_id": student_uuid},
            )
            await session.execute(
                text("DELETE FROM exam_submissions WHERE exam_id=:exam_id AND student_id=:student_id"),
                {"exam_id": EXAM_ID, "student_id": student_uuid},
            )
            # todo  CHECK (status IN ('submitted', 'ai_processing', 'pending_review', 'reviewed', 'published')),
            await session.execute(
                text("""
                    INSERT INTO exam_submissions (id, exam_id, student_id, tenant_id, status)
                    VALUES (:id, :exam_id, :student_id, 'tenant_default', 'ai_processing')
                """),
                {"id": submission_id, "exam_id": EXAM_ID, "student_id": student_uuid},
            )
    logger.info("写入临时提交记录", submission_id=submission_id)

    # ── thread_id：MemorySaver 用它区分不同的批改任务 ─────────────
    # 两次 ainvoke() 必须用同一个 thread_id，图才能在第二次找回冻结的 State
    thread_id = f"test-{uuid.uuid4()}"
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages":        [],
        "word_file_path":  "test_exam_answer.docx",
        "exam_id":         EXAM_ID,
        "submission_id":   submission_id,
        "student_id":      student_uuid,
        "tenant_id":       "tenant_default",
        "session_id":      "test-session",
        "teacher_decision": None,
        "fallback_used":   False,
        "sandbox_skipped": False,
    }

    # ══ 第一次 ainvoke：跑到 teacher_review_node 的 interrupt() 自动暂停 ══
    print("=" * 60)
    print("第一次 ainvoke：启动图，等待 interrupt 暂停...")
    print("=" * 60)
    result1 = await graph.ainvoke(initial_state, config=config)

    # interrupt() 暂停后，ainvoke 提前返回，State 已存入 MemorySaver
    # result1 包含截止到暂停时刻的所有 state 字段
    summary = result1.get("pre_review_summary", {})
    print(f"\n图已暂停（interrupt 触发）")
    print(f"AI 预评总分：{summary.get('total_score')} / {summary.get('full_score')}")
    print(f"需复核题数：{summary.get('needs_review_count')}")
    print(f"薄弱点数量：{len(result1.get('weak_points', []))}")
    print(f"\n>>> 此刻图冻结，等待教师决策 <<<\n")

    # ══ 模拟教师操作：构造 decision，第二次 ainvoke 恢复图 ══════
    # todo 使用真实的 teacher_id
    decision1 = {
        "action":        "approve",   # 或 "modify"，见说明②
        "modifications": [],
        "teacher_id":    "7d5052d2-23c2-423b-9e3b-5a3a7755cc26",  # reviewed_by 列是 UUID 类型，必须合法
        # "teacher_id":    str(uuid.uuid4()),  # reviewed_by 列是 UUID 类型，必须合法
    }

    # todo 改成 "modify"
    decision = {
        # "action": "modify",
        "action": "approve",
        "modifications": [
            {
                "question_id": "从 result1['pre_review_summary']['by_question'][0]['question_id'] 取",
                "new_score": 5,
                "comment": "答案思路正确但表述不完整",
            }
        ],
        "teacher_id": "7d5052d2-23c2-423b-9e3b-5a3a7755cc26",  # 必须是合法 UUID
    }

    print("=" * 60)
    print("第二次 ainvoke：Command(resume=decision)，恢复图执行...")
    print("=" * 60)
    result2 = await graph.ainvoke(Command(resume=decision), config=config)

    final = result2.get("final_results", [])
    print(f"\n图执行完毕")
    print(f"最终批改题数：{len(final)}")
    for q in final:
        print(f"  题{q['question_no']} [{q['question_type']}] "
              f"最终得分 {q['final_score']}/{q['full_score']}")


asyncio.run(main())
