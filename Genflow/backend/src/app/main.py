from fastapi import FastAPI
from app.api.v1.api import api_router
import uvicorn

app = FastAPI(title="Genflow LLM Search Service")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to Genflow LLM Search Service"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
