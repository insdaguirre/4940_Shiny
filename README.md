# RecycLens MVP

An AI-powered recycling assistant that helps users identify recyclable items and find nearby recycling facilities.

## Features

- **Image Analysis**: Upload a photo of an item to identify its materials
- **Recyclability Assessment**: Get instant feedback on whether an item can be recycled
- **RAG-Enhanced Regulations**: Uses Retrieval-Augmented Generation (RAG) to provide accurate, location-specific recycling regulations from official sources
- **Local Facilities**: Find nearby recycling and disposal facilities using web search
- **Interactive Map**: View facilities on an interactive Mapbox map with markers
- **Clear Instructions**: Receive step-by-step guidance for proper disposal
- **Staged Progress Updates**: Real-time feedback during analysis (image analysis → recyclability → geocoding)

## Tech Stack

- **Frontend**: Shiny (Python) web framework
- **Backend**: Node.js + Express + TypeScript
- **RAG Service**: Python + FastAPI + LlamaIndex for querying local recycling regulations
- **AI Services**: OpenAI Chat Completions API (GPT-5) for image analysis + OpenAI Responses API (GPT-4.1 with web search) for recyclability
- **Maps**: Mapbox GL JS + Mapbox Geocoding API

## Prerequisites

- Python 3.8+
- Node.js 18+ and npm (for backend server)
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Mapbox access token ([Get one here](https://account.mapbox.com/access-tokens/)) - Optional, for map functionality

## Setup

### 1. Install Backend Dependencies

```bash
npm install
```

### 2. Install Python Dependencies

**Frontend dependencies:**
```bash
cd app
pip install -r requirements.txt
cd ..
```

**RAG service dependencies:**
```bash
cd rag_service
pip install -r requirements.txt
cd ..
```

### 3. Configure Environment Variables

Create a `.env` file in the project root for the backend:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
RAG_SERVICE_URL=http://localhost:8001
PORT=3001
NODE_ENV=development
```

Create a `.env` file in the `app/` directory for the Shiny frontend:

```bash
BACKEND_URL=http://localhost:3001
MAPBOX_ACCESS_TOKEN=pk.your-mapbox-token-here
```

Create a `.env` file in the `rag_service/` directory (optional):

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
PORT=8001
```

**Notes:**
- The Mapbox access token is optional. Without it, the map will not display, but other features will still work.
- The `RAG_SERVICE_URL` in the backend `.env` is optional. If not set, RAG queries will be skipped (graceful degradation).
- The `OPENAI_API_KEY` in the RAG service is optional and only needed if llama-index requires it for embeddings.

### 4. Run the Application

You need to run three services: the RAG service, backend server, and Shiny frontend:

**Terminal 1 - RAG Service:**
```bash
cd rag_service
uvicorn app:app --host 0.0.0.0 --port 8001
```

The RAG service will start on `http://localhost:8001`

**Terminal 2 - Backend Server:**
```bash
npm run server
```

The backend will start on `http://localhost:3001`

**Terminal 3 - Shiny Frontend:**
```bash
cd app
shiny run app.py
```

Or using Python directly:
```bash
cd app
python -m shiny run app.py
```

The Shiny app will start and display a URL (typically `http://127.0.0.1:8000`). Open this URL in your browser.

**Note:** The RAG service is optional. If it's not running, the backend will continue to work but without RAG-enhanced regulations.

## How It Works

### System Architecture

The Shiny frontend communicates with a Node.js backend API that handles AI processing:

1. **User Input**: User uploads an image, enters location, and optionally adds context via the Shiny UI
2. **Image Analysis**: Backend calls GPT-5 Chat Completions API to analyze the image and identify:
   - Primary and secondary materials
   - Item condition (clean, soiled, damaged, etc.)
   - Contaminants (food residue, grease, etc.)
   - Material category
3. **RAG Query** (if RAG service is available): Backend queries RAG service for local recycling regulations:
   - Searches vector store for location-specific regulations using similarity search (top_k=15)
   - Retrieves official recycling guidelines for the material and location
   - Returns relevant regulations to enhance the analysis
4. **Recyclability Assessment**: Backend calls GPT-4.1 Responses API with RAG context and web search to:
   - Determine recyclability based on local regulations (from RAG) and general knowledge
   - Find nearby recycling/disposal facilities
   - Generate step-by-step disposal instructions
5. **Geocoding**: Frontend geocodes facility addresses using Mapbox Geocoding API
6. **Results Display**: Shiny app displays:
   - Recyclability decision with confidence score
   - Disposal instructions
   - Interactive map with facility markers
   - Facility cards with details and links

### Staged Progress Updates

The app provides real-time feedback during analysis:

- **Stage 1**: "Analyzing image..." - Vision API call (GPT-5)
- **Stage 2**: "Querying local regulations..." - RAG service query (if available)
- **Stage 3**: "Determining recyclability and searching for facilities..." - Recyclability API call (GPT-4.1 with web search)
- **Stage 4**: "Finding recycling locations..." - Geocoding and map rendering

## Project Structure

```
RecycLens/
├── app/                    # Shiny frontend application
│   ├── app.py             # Main Shiny application
│   ├── requirements.txt   # Python dependencies
│   ├── README.md          # Shiny app documentation
│   ├── static/            # Static assets
│   │   └── styles.css     # Custom CSS styling
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── api_client.py  # API call functions
├── rag_service/           # RAG microservice
│   ├── app.py            # FastAPI HTTP server
│   ├── rag_query.py      # RAG query logic
│   ├── requirements.txt  # Python dependencies
│   ├── railway.json      # Railway configuration
│   └── Procfile          # Process file
├── rag/                   # RAG data and vector store
│   ├── rag_index_morechunked/  # Vector store files
│   └── rag_docs/         # Markdown documents
├── server/                # Backend Express server
│   ├── index.ts          # Server entry point
│   ├── routes/           # API routes
│   │   └── analyze.ts    # Main analysis endpoint
│   ├── services/         # Business logic
│   │   ├── visionService.ts    # Vision API integration
│   │   ├── gpt5Service.ts      # Responses API integration
│   │   └── ragService.ts       # RAG service client
│   └── types.ts          # Backend types
├── package.json          # Backend Node.js dependencies
├── tsconfig.json         # TypeScript configuration
└── tsconfig.server.json  # Server TypeScript configuration
```

## API Endpoints

The backend provides the following endpoints:

### POST /api/analyze/vision

Analyzes an image to identify materials.

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "stage": "vision",
  "result": {
    "primaryMaterial": "Plastic",
    "secondaryMaterials": [],
    "category": "Container",
    "condition": "clean",
    "contaminants": [],
    "confidence": 0.95,
    "shortDescription": "Plastic bottle"
  }
}
```

### POST /api/analyze/recyclability

Determines recyclability and finds facilities.

**Request Body:**
```json
{
  "visionResult": { ... },
  "location": "Ithaca, NY 14850",
  "context": "Plastic container with food residue"
}
```

**Response:**
```json
{
  "stage": "recyclability",
  "result": {
    "isRecyclable": true,
    "category": "Plastic",
    "bin": "recycling",
    "confidence": 0.87,
    "materialDescription": "Plastic bottle",
    "instructions": [
      "Rinse the container thoroughly",
      "Remove any labels if possible",
      "Place in recycling bin"
    ],
    "reasoning": "This is a clean plastic container that can be recycled.",
    "locationUsed": "Ithaca, NY 14850",
    "facilities": [
      {
        "name": "Green Valley Recycling",
        "type": "Recycling Center",
        "address": "123 Main St, Ithaca, NY",
        "url": "https://example.com",
        "notes": "Accepts plastic containers"
      }
    ]
  }
}
```

## Troubleshooting

### Backend won't start

- Check that `OPENAI_API_KEY` is set in `.env` (project root)
- Ensure port 3001 is not in use
- Check console for error messages

### RAG service won't start

- Check that the `rag/rag_index_morechunked/` directory exists with vector store files
- Verify Python dependencies are installed: `pip install -r rag_service/requirements.txt`
- Ensure port 8001 is not in use (or change `PORT` in `.env`)
- Check console for error messages about missing vector store files

### Shiny app can't connect to backend

- Ensure the backend server is running on port 3001
- Check that `BACKEND_URL` in `app/.env` matches the backend server URL
- Verify the backend is accessible at the specified URL

### RAG queries not working

- Ensure the RAG service is running on port 8001
- Check that `RAG_SERVICE_URL` in backend `.env` matches the RAG service URL
- Verify the RAG service is accessible: `curl http://localhost:8001/health`
- Check that `rag_service/rag_index_morechunked/` directory exists with all vector store files (for Railway deployment) or `rag/rag_index_morechunked/` (for local development)
- Review RAG service logs for errors

### Map not displaying

- Verify `MAPBOX_ACCESS_TOKEN` is set in `app/.env`
- Check that the token is valid and has the necessary permissions
- Check browser console for JavaScript errors

### Image upload issues

- Ensure the image file is a valid image format (JPEG, PNG, GIF, WebP)
- Check that the file size is reasonable (backend may have limits)
- Verify the backend server is running and accessible

## Development

### Backend Development

- Edit files in `server/` directory
- Restart the backend server to see changes
- Backend uses TypeScript - compile with `npm run build:server`

### Shiny Frontend Development

- Edit `app/app.py` for main application logic
- Edit `app/utils/api_client.py` for API call logic
- Edit `app/static/styles.css` for styling
- Restart the Shiny app to see changes

## License

© 2025 RecycLens. Making recycling simple.
