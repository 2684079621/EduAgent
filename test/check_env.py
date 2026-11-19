import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
# 1. 导入新的 AsyncMilvusClient，不再需要 connections 和 utility
from pymilvus import AsyncMilvusClient

DB_URL = "postgresql+asyncpg://eduagent_user:zyh020101@localhost:5433/eduagent"
MILVUS_URI = "http://localhost:19531"  # 注意：新的客户端使用 URI 格式，需要带 http://

async def check_postgres():
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        r = await conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL 连通：", r.scalar())
    await engine.dispose()

# 2. 改为异步函数，使用 async with 管理客户端
async def check_milvus():
    # 使用 AsyncMilvusClient，传入完整的 URI
    async with AsyncMilvusClient(uri=MILVUS_URI) as client:
        # 3. 使用 await 调用异步方法获取集合列表
        collections = await client.list_collections()
        print("✅ Milvus 连通，已有集合：", collections)

async def main():
    await check_postgres()
    await check_milvus() # 4. 加上 await 调用

if __name__ == "__main__":
    asyncio.run(main())