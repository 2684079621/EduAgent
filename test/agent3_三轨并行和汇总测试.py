#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: agent3_三轨并行和汇总测试.py
作者: ZZS
项目: 2_代码
创建日期: 2026/10/16
描述: 
"""
import asyncio
import json
import sys

from backend.agents.exam.nodes import (
    parse_word_node,
    load_questions_meta_node,
    run_three_tracks_node,
    aggregate_results_node,
    analyze_weak_points_node,
)


async def run_pipeline(word_path: str, exam_id: str):
    # ── 初始 state ────────────────────────────────────────────
    state = {
        "word_file_path": word_path,
        "exam_id": exam_id,
        "parsed_questions": [],
        "submission_id": "abc"
    }

    # ── 节点1：解析 Word ──────────────────────────────────────
    print("=" * 60)
    print("节点1 parse_word_node")
    print("=" * 60)
    out1 = await parse_word_node(state)
    state.update(out1)
    print(json.dumps(state["parsed_questions"], ensure_ascii=False, indent=2))

    # ── 节点2：从 DB 补全题目元数据 ───────────────────────────
    print("\n" + "=" * 60)
    print("节点2 load_questions_meta_node")
    print("=" * 60)
    out2 = await load_questions_meta_node(state)
    state.update(out2)
    print(f"merged_questions 数量: {len(state['parsed_questions'])}")
    for q in state["parsed_questions"]:
        print(f"  Q{q['question_no']} [{q['question_type']}] "
              f"student_answer={q['student_answer']!r} "
              f"scoring_points={len(q['scoring_points'])}")

    # ── 节点3：三轨并行批改 ───────────────────────────────────
    print("\n" + "=" * 60)
    print("节点3 run_three_tracks_node")
    print("=" * 60)
    out3 = await run_three_tracks_node(state)
    state.update(out3)
    print(f"objective_results : {len(state['objective_results'])}")
    print(f"subjective_results: {len(state['subjective_results'])}")
    print(f"code_results      : {len(state['code_results'])}")

    # ── 节点4：汇总 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("节点4 aggregate_results_node")
    print("=" * 60)
    out4 = await aggregate_results_node(state)
    state.update(out4)

    summary = state["pre_review_summary"]
    print(f"总分      : {summary['total_score']} / {summary['full_score']}")
    print(f"得分率    : {summary['score_rate']}")
    print(f"需复核题数: {summary['needs_review_count']}")
    print("\n逐题明细:")
    for r in summary["by_question"]:
        print(f"  Q{r['question_no']} [{r['question_type']}] "
              f"score={r.get('score')}/{r.get('full_score')} "
              f"needs_review={r.get('needs_review')} "
              f"feedback={r.get('ai_feedback', '')[:40]!r}")


    # ── 节点5：薄弱点 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("节点5 analyze_weak_points_node")
    print("=" * 60)
    out5 = await analyze_weak_points_node(state)
    state.update(out5)
    print("节点5的输出：",state)

    return state


if __name__ == "__main__":
    word_path = "test_exam_answer.docx"
    exam_id   = "e0000001-0000-0000-0000-000000000001"

    final_state = asyncio.run(run_pipeline(word_path, exam_id))