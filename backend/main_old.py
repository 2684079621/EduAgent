# backend/main.py

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.v1 import resume,qa,exam,interview
from backend.api.v1 import auth
from backend.mcp.knowledge_base_server import mcp as kb_mcp
from backend.mcp.web_search_server import mcp as ws_mcp




@asynccontextmanager
async def lifespan(app: FastAPI):
    # app.mount() 不会自动传播子应用 lifespan，必须在这里手动启动
    # _session_manager.run() 初始化 anyio task group，MCP 请求依赖它
    async with kb_mcp._session_manager.run():
        async with ws_mcp._session_manager.run():
            yield

app = FastAPI(lifespan=lifespan)

app.include_router(auth.router, prefix="/api/v1/auth")
app.include_router(resume.router, prefix="/api/v1//resume")
app.include_router(qa.router, prefix="/api/v1/qa")
app.include_router(exam.router, prefix="/api/v1/exam")
app.include_router(interview.router, prefix="/api/v1/interview")


app.mount("/mcp/kb",     kb_mcp.streamable_http_app())
app.mount("/mcp/search", ws_mcp.streamable_http_app())

@app.get("/health")
async def health():
    return {"status": "ok"}

# todo 可以在main方法中启动，或者使用命令行启动，用代码测试。
# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8008)

# todo 终端位置,根目录下启动
#   (edu_agent) PS D:\武汉6期共享\EduAgent\03_代码\EduAgent> uvicorn backend.main:app --port 8008

# todo 有服务端了，用页面进行测试。
# http://127.0.0.1:8008/docs