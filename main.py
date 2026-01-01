import uvicorn
from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="Fathom RAG Adapter",
    description="Autonomous Retrieval-Intelligence Layer",
    version="1.0.0"
)

# Connect the API routes
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    print("🚀 Fathom Autonomous RAG is starting...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)