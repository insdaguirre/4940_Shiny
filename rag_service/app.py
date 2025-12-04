"""FastAPI HTTP service for RAG queries."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from pathlib import Path
from llama_index.core import Settings
from rag_query import query_rag, RAG_INDEX_PATH

app = FastAPI(title="RecycLens RAG Service", version="1.0.0")

# CORS middleware to allow requests from backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to backend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""
    material: str
    location: str
    condition: Optional[str] = ""
    context: Optional[str] = ""


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries."""
    regulations: str
    sources: list[str]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "rag-service",
        "rag_index_path": str(RAG_INDEX_PATH)
    }


@app.get("/debug")
async def debug_info():
    """Debug endpoint to verify RAG service configuration."""
    openai_key_set = bool(os.getenv("OPENAI_API_KEY"))
    index_exists = RAG_INDEX_PATH.exists()
    
    # Check if embedding model is initialized
    embedding_model_info = None
    if hasattr(Settings, 'embed_model') and Settings.embed_model is not None:
        embedding_model_info = str(type(Settings.embed_model).__name__)
    
    return {
        "status": "ok",
        "openai_api_key_set": openai_key_set,
        "rag_index_exists": index_exists,
        "rag_index_path": str(RAG_INDEX_PATH),
        "embedding_model": embedding_model_info,
        "cwd": os.getcwd()
    }


@app.post("/query", response_model=RAGQueryResponse)
async def query_regulations(request: RAGQueryRequest):
    """
    Query RAG for recycling regulations.
    
    Args:
        request: RAG query request with material, location, condition, and context
        
    Returns:
        RAG query response with regulations text and sources
    """
    try:
        regulations, sources = query_rag(
            material=request.material,
            location=request.location,
            condition=request.condition or "",
            context=request.context or ""
        )
        
        return RAGQueryResponse(
            regulations=regulations,
            sources=sources
        )
        
    except FileNotFoundError as e:
        # RAG index not found - return empty response instead of error
        # This allows the main flow to continue without RAG
        print(f"RAG index not found: {e}")
        return RAGQueryResponse(
            regulations="",
            sources=[]
        )
    except Exception as e:
        # Log error but return empty response to not break main flow
        print(f"RAG query error: {e}")
        import traceback
        traceback.print_exc()
        return RAGQueryResponse(
            regulations="",
            sources=[]
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

