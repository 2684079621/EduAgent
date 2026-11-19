import asyncio, httpx, fitz
from httpx import ASGITransport
from fastapi import FastAPI
from backend.api.v1.auth import router as auth_router
from backend.api.v1.resume import router as resume_router

app = FastAPI()
app.include_router(auth_router)
app.include_router(resume_router, prefix="/resume")

# 造一个测试 PDF
_d = fitz.open(); _p = _d.new_page()
_p.insert_text((50, 60), "张三 后端开发\n技能 Java Spring Boot", fontsize=11, fontname="china-s")
_d.save("upload_test.pdf"); _d.close()

async def main():
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        tok = (await c.post("/login", json={"username": "student01", "password": "Student@123456"})).json()["access_token"]
        H = {"Authorization": f"Bearer {tok}"}

        with open("upload_test.pdf", "rb") as f:
            r = await c.post("/resume/upload", headers=H, files={"file": ("resume.pdf", f, "application/pdf")})
        print("① 上传:", r.status_code, "| status:", r.json()["status"])
        rid = r.json()["review_id"]

        r2 = await c.get(f"/resume/reviews/{rid}", headers=H)
        print("② 查询:", r2.status_code, "| status:", r2.json()["status"])

        r3 = await c.get("/resume/reviews", headers=H)
        print("③ 列表 total>=1:", r3.json()["total"] >= 1)

        r4 = await c.post("/resume/upload", headers=H, files={"file": ("x.txt", b"hello", "text/plain")})
        print("④ 非PDF:", r4.status_code, "|", r4.json()["detail"])

        r5 = await c.post("/resume/upload", files={"file": ("r.pdf", b"%PDF", "application/pdf")})
        print("⑤ 无token:", r5.status_code)

        r6 = await c.delete(f"/resume/reviews/{rid}", headers=H)
        print("⑥ 删除:", r6.status_code)

        r7 = await c.get(f"/resume/reviews/{rid}", headers=H)
        print("⑦ 删后查询:", r7.status_code)

asyncio.run(main())
