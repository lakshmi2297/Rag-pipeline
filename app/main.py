from fastapi import FastAPI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Pipeline API",
    description="Document analysis with AI-powered querying",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
