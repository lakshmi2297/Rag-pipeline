from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results")

class Source(BaseModel):
    filename: str
    chunk_id: int
    content: str  
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    search_triggered: bool

class ProcessedFile(BaseModel):
    filename: str
    chunks_created: int

class IngestionResponse(BaseModel):
    message: str
    processed_files: List[ProcessedFile]  
    total_chunks: int
    timestamp: str
