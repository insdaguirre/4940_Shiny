# RAG Service

A FastAPI microservice that provides Retrieval-Augmented Generation (RAG) queries for recycling regulations.

## Overview

The RAG service queries a vector store of local recycling regulations to provide accurate, location-specific recycling information. It uses LlamaIndex to search through official recycling guidelines for Albany and Tompkins counties in New York.

## Features

- Query recycling regulations by material and location
- Automatic county detection (Albany, Tompkins)
- Vector similarity search for relevant regulations
- Returns formatted regulations text with source citations

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file (optional):

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here  # Only if needed for embeddings
PORT=8001
```

### 3. Verify Vector Store

Ensure the vector store files exist:
- For local development: `../rag/rag_index_morechunked/` directory with all JSON files
- For Railway deployment: `rag_service/rag_index_morechunked/` directory with all JSON files
- If missing, extract them from the git commit (see main README)

## Running Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

Or using Python directly:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

The service will be available at `http://localhost:8001`

## API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "rag-service",
  "rag_index_path": "/path/to/rag/rag_index_morechunked"
}
```

### POST /query

Query RAG for recycling regulations.

**Request Body:**
```json
{
  "material": "Plastic",
  "location": "Ithaca, NY",
  "condition": "clean",
  "context": "Plastic bottle"
}
```

**Response:**
```json
{
  "regulations": "Plastic bottles are recyclable in Tompkins County...",
  "sources": [
    "https://www.tompkinscountyny.gov/...",
    "./rag_pdf_data/tompkins/2025curbsiderecyclingguidelines-final-onlineviewing.pdf"
  ]
}
```

## Architecture

- **FastAPI**: HTTP server framework
- **LlamaIndex**: Vector store and query engine
- **Vector Store**: Pre-built index of recycling regulations
- **Query Engine**: Semantic search with similarity_top_k=10 (for query engine) and similarity_top_k=15 (for retriever)

## Error Handling

The service is designed to fail gracefully:
- If vector store is missing, returns empty response
- If query fails, returns empty response (doesn't break main flow)
- All errors are logged but don't crash the service

## Railway Deployment

See `RAILWAY_DEPLOYMENT.md` in the project root for deployment instructions.

The service is configured to:
- Use NIXPACKS builder
- Start with `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Access vector store at `rag_service/rag_index_morechunked/` (relative to service root for Railway deployment)
- For local development, the vector store is accessed at `../rag/rag_index_morechunked/` (relative to service root)

## Troubleshooting

### Vector store not found

- For Railway deployment: Verify `rag_service/rag_index_morechunked/` directory exists in the service root
- For local development: Verify `rag/rag_index_morechunked/` directory exists in project root
- Check that all 5 JSON files are present:
  - `default__vector_store.json`
  - `docstore.json`
  - `graph_store.json`
  - `image__vector_store.json`
  - `index_store.json`

### Import errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Query returns empty results

- Verify vector store files are not corrupted
- Check that markdown documents exist in `rag/rag_docs/`
- Review service logs for error messages

