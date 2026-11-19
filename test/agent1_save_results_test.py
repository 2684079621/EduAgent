import asyncio, uuid
from sqlalchemy import text
from backend.dependencies import AsyncSessionLocal
from backend.agents.resume.nodes import save_results_node

rid = str(uuid.uuid4())

async def main():
    async with AsyncSessionLocal() as s:        # 先插一条 processing 记录
        sid = (await s.execute(text("SELECT id FROM users WHERE username='student01'"))).scalar()
        await s.execute(text("""INSERT INTO resume_reviews (id,tenant_id,student_id,pdf_minio_path,status)
            VALUES (:id,'tenant_default',:sid,'resumes/x.pdf','processing')"""), {"id": rid, "sid": sid})
        await s.commit()

    await save_results_node({                    # 运行节点
        "review_id": rid, "student_id": str(sid), "pdf_local_path": "/tmp/none.pdf",
        "structured": {"name": "张三", "skills_list": ["Java"]}, "weighted_score": 78.5,
        "dimension_scores": [{"key": "project_depth", "dimension": "项目深度", "score": 75, "weight": 0.3}],
        "issues": [{"priority": "high", "description": "无量化"}],
        "summary": {"highlights": ["有项目"], "overall_comment": "中等"},
    })

    async with AsyncSessionLocal() as s:         # 查回验证（用 JSONB 操作符）
        row = (await s.execute(text("""
            SELECT status, (scores->>'weighted_score') AS ws,
                   (structured_data->>'name') AS nm, jsonb_array_length(issues) AS cnt
            FROM resume_reviews WHERE id=:id"""), {"id": rid})).mappings().fetchone()
        print("status:", row["status"], "| 加权分:", row["ws"], "| 姓名:", row["nm"], "| 问题数:", row["cnt"])

asyncio.run(main())
