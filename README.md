# EduAgent · AI 助教智能体系统

基于 **LangGraph 多智能体编排** 的 AI 教学助手，覆盖智能问答、试卷批改、简历审查、模拟面试四大教学场景。后端 FastAPI + 前端 Vue 3，支持意图自动路由、多 Agent 串联 Pipeline、知识库 RAG 与 SSE 流式输出。

## 功能概览

| 场景 | 能力 | 入口 |
| --- | --- | --- |
| 智能问答 | 知识库 RAG 检索（BGE-M3 混合检索 + BGE-Reranker 精排）、联网搜索兜底 | `/api/v1/qa` |
| 试卷批改 | Word/PDF 试卷解析、标准答案比对、评分与讲评生成 | `/api/v1/exam` |
| 简历审查 | PDF/Word 简历解析、六维度评分与改进建议 | `/api/v1/resume` |
| 模拟面试 | 分阶段多轮追问、实时评估、面试报告生成 | `/api/v1/interview` |
| 统一助手 | 意图识别 → 单 Agent 直达 / 多 Agent 串联 / 澄清追问 | `/api/v1/chat` |

## 架构

```
┌──────────────────────────────────────────────┐
│  Vue 3 + Vite + Element Plus  (端口 3000)     │
└───────────────────┬──────────────────────────┘
                    │ HTTP / SSE
┌───────────────────▼──────────────────────────┐
│  FastAPI  (端口 8000)                        │
│  ├─ api/v1     auth / chat / qa / exam /     │
│  │             resume / interview            │
│  ├─ core       Orchestrator 意图路由与编排    │
│  │             LLMFactory / Retry / Memory   │
│  ├─ agents     qa · exam · resume · interview│
│  │             每个 Agent = LangGraph 状态机  │
│  └─ mcp        knowledge_base / web_search   │
│                （MCP Server 以子应用挂载）    │
└───────┬───────────────────────┬──────────────┘
        │                       │
┌───────▼────────┐   ┌──────────▼─────────────┐
│ PostgreSQL     │   │ Milvus 向量库           │
│ 业务数据/会话   │   │ 知识库检索              │
│ (5433)         │   │ (19531) + MinIO + etcd  │
└────────────────┘   └────────────────────────┘
        │
┌───────▼──────────────────────────────────────┐
│ 本地模型：BGE-M3 嵌入 · BGE-Reranker 精排 ·   │
│ MiniLM 意图分类 · DeepSeek 对话（远程 API）   │
└──────────────────────────────────────────────┘
```

**编排逻辑**：请求先经 `QueryClassifier` 做意图分类，`Orchestrator` 据此选择 `single`（单 Agent 直达）、`pipeline`（多 Agent 串联，如「改简历 → 模拟面试」）或 `clarify`（意图不明时追问）。

## 技术栈

- **后端**：FastAPI 0.117 · LangChain 1.2 · LangGraph 1.0 · SQLAlchemy 2（异步）· asyncpg · pymilvus
- **前端**：Vue 3.4 · TypeScript · Vite 5 · Pinia · Element Plus · markdown-it
- **存储**：PostgreSQL 15 · Milvus 2.4（standalone，依赖 etcd + MinIO）
- **模型**：DeepSeek（对话）· BAAI/bge-m3（嵌入）· BAAI/bge-reranker-large（精排）· all-MiniLM-L6-v2（意图分类）
- **协议**：MCP（Model Context Protocol），知识库与联网搜索均以 MCP Server 形式对内提供

## 目录结构

```
EduAgent/
├── backend/
│   ├── main.py              # 应用入口：lifespan、CORS、路由与 MCP 挂载
│   ├── config.py            # 配置中心，从 .env.local 读取
│   ├── api/v1/              # 各业务接口（auth/chat/qa/exam/resume/interview）
│   ├── agents/              # 四个 Agent，各自 graph / nodes / prompts / state
│   ├── core/                # 编排、LLM 工厂、知识库、检索、重试、日志
│   ├── db/                  # 建表与迁移
│   ├── mcp/                 # 知识库 MCP、联网搜索 MCP
│   └── models/              # 本地模型权重（未入库，需自行下载）
├── frontend/                # Vue 3 前端
├── scripts/                 # 初始化与数据脚本
├── test/                    # 接口、Graph、端到端测试
├── samples/                 # 示例文档
├── docker-compose.yml       # Postgres / Milvus / etcd / MinIO / Attu
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 启动依赖服务

```bash
cp .env.example .env.local     # 填入 DB_PASSWORD、DEEPSEEK_API_KEY、JWT_SECRET_KEY
docker compose --env-file .env.local up -d
```

启动后：PostgreSQL `localhost:5433`、Milvus `localhost:19531`、Attu 控制台 `localhost:30000`。

### 2. 下载本地模型权重

权重约 4.7 GB，未纳入仓库，需放到 `backend/models/` 下：

```bash
pip install -U huggingface_hub
huggingface-cli download BAAI/bge-m3                     --local-dir backend/models/embedding/bge-m3
huggingface-cli download BAAI/bge-reranker-large         --local-dir backend/models/reranker/bge-reranker-large
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir backend/models/classifier/all-MiniLM-L6-v2
```

国内网络可用镜像：`export HF_ENDPOINT=https://hf-mirror.com`。

### 3. 启动后端

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

接口文档：http://127.0.0.1:8000/docs ，健康检查 `/health`。首次启动会自动建表、初始化 Milvus 集合，并预热三个本地模型。

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev      # http://localhost:3000
```

### 5. 初始化数据（可选）

```bash
python scripts/init_milvus.py           # 初始化向量库集合
python scripts/build_knowledge_base.py  # 构建知识库
python scripts/seed_data.py             # 灌入测试账号
```

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/auth/register` `/login` | 注册 / 登录，签发 JWT |
| POST | `/api/v1/chat/stream` | 统一助手，SSE 流式，自动路由 |
| POST | `/api/v1/qa/ask` | 知识库问答 |
| POST | `/api/v1/exam/submit` | 提交试卷，返回批改结果 |
| POST | `/api/v1/resume/upload` | 上传简历，返回六维度报告 |
| POST | `/api/v1/interview/start` `/answer` | 模拟面试多轮对话 |

## 测试

```bash
pytest test/ -v
```

`test/` 下按 Agent 分组（`agent1~agent4`）覆盖 Graph 流程、API 接口与端到端场景。

## 说明

- `.env.local`、模型权重、`node_modules`、`__pycache__` 等均在 `.gitignore` 中，不会进入仓库。
- 联网搜索默认使用免费的 DuckDuckGo；配置 `TAVILY_API_KEY` 或 `BOCHA_API_KEY` 可切换为 Tavily / 博查。
- 项目源于教学实践，代码保留了大量中文注释用于讲解实现思路。
