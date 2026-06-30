from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import time
import logging

from app.retrieval.retriever import Retriever
from app.generation.generator import Generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is Niemann-Pick disease?")
    top_k: int = Field(default=3, ge=1, le=10)

class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float

class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[SourceChunk]
    latency_ms: float

@router.post("/", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    start_time = time.time()
    
    # 1. Retrieve chunks
    try:
        results = Retriever.search(req.query, top_k=req.top_k)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")
        
    # 2. Generate Answer
    try:
        answer = Generator.generate_answer(req.query, results)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
        
    # 3. Format Response
    source_chunks = []
    for res in results:
        source_chunks.append(SourceChunk(
            chunk_id=res.get("id", "unknown"),
            text=res.get("text", ""),
            metadata=res.get("metadata", {}),
            distance=res.get("distance", 0.0)
        ))
        
    latency = (time.time() - start_time) * 1000
    
    return QueryResponse(
        answer=answer,
        source_chunks=source_chunks,
        latency_ms=latency
    )
