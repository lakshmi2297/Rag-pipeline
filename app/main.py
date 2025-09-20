from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
import os
from datetime import datetime

from .models import QueryRequest, QueryResponse, IngestionResponse
from .services.ingestion import IngestionService
from .services.query_processor import QueryProcessor
from .services.semantic_search import SemanticSearch
from .services.generation import GenerationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Pipeline API",
    description="Document analysis with FastAPI and Mistral AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ingestion_service = IngestionService()
query_processor = QueryProcessor()
semantic_search = SemanticSearch()
generation_service = GenerationService()

@app.on_startup
async def startup_event():
    logger.info("🚀 RAG Pipeline API starting...")
    os.makedirs("data/documents", exist_ok=True)
    logger.info("✅ Services initialized")

@app.get("/")
async def root():
    return {
        "message": "RAG Pipeline API with FastAPI + Mistral AI",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": ["ingestion", "search", "generation"]
    }

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(400, "No files provided")
        
        processed_files = []
        total_chunks = 0
        
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(400, f"{file.filename} is not a PDF")
            
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(400, f"{file.filename} is empty")
            
            # Process document through ingestion service
            chunks = await ingestion_service.process_document(content, file.filename)
            
            # Add chunks to search index
            for chunk in chunks:
                semantic_search.add_document(chunk)
            
            processed_files.append({
                "filename": file.filename,
                "chunks_created": len(chunks)
            })
            total_chunks += len(chunks)
        
        return IngestionResponse(
            message=f"Successfully processed {len(files)} files",
            processed_files=processed_files,
            total_chunks=total_chunks,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        query = request.query.strip()
        
        if not query:
            raise HTTPException(400, "Query cannot be empty")
        
        logger.info(f"Processing query: {query}")
        
        # Step 1: Intent detection
        should_search = query_processor.detect_search_intent(query)
        
        if not should_search:
            return QueryResponse(
                answer="Hello! I'm ready to help with questions about your documents.",
                sources=[],
                confidence=1.0,
                search_triggered=False
            )
        
        # Step 2: Query transformation
        processed_query = query_processor.transform_query(query)
        
        # Step 3: Search for relevant content
        search_results = await semantic_search.search(
            processed_query, 
            top_k=request.top_k,
            include_keywords=True
        )
        
        if not search_results:
            return QueryResponse(
                answer="No relevant information found. Please upload relevant documents.",
                sources=[],
                confidence=0.0,
                search_triggered=True
            )
        
        # Step 4: Generate answer using Mistral AI
        answer, confidence = await generation_service.generate_answer(
            query, processed_query, search_results
        )
        
        # Format sources for response
        sources = [{
            "filename": r["filename"],
            "chunk_id": r["chunk_id"],
            "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
            "score": round(r["score"], 3)
        } for r in search_results]
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=round(confidence, 3),
            search_triggered=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    return semantic_search.get_stats()
