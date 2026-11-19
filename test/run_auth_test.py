# run_auth_test.py（临时，仅用于本节测试）
from fastapi import FastAPI
from backend.api.v1.auth import router
import uvicorn

app = FastAPI()
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run("test.run_auth_test:app", host="localhost", port=8000, reload=True)

