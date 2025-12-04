# RecycLens Shiny MVP

A Shiny (Python) web application that replicates all functionality from the React app. This is a demo version built with Shiny for showcasing purposes.

## Features

- **Image Upload**: Upload photos of items to analyze
- **Staged Analysis**: Real-time progress updates during analysis (image analysis → recyclability → geocoding)
- **Recyclability Assessment**: Get instant feedback on whether an item can be recycled
- **Local Facilities**: Find nearby recycling and disposal facilities
- **Interactive Map**: View facilities on an interactive Mapbox map with markers
- **Clear Instructions**: Receive step-by-step guidance for proper disposal
- **How it Works Page**: Step-by-step guide explaining the process

## Prerequisites

- Python 3.8+
- Node.js backend server running (see main README.md for setup)
- OpenAI API key
- Mapbox access token (optional, for map functionality)

## Setup

### 1. Install Python Dependencies

```bash
cd app
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the `app/` directory:

```bash
BACKEND_URL=http://localhost:3001
MAPBOX_ACCESS_TOKEN=pk.your-mapbox-token-here
```

**Note:** The backend URL should point to your running Node.js server. The default is `http://localhost:3001`.

**Note:** The Mapbox access token is optional. Without it, the map will not display, but other features will still work.

### 3. Start the Backend Server

Make sure the Node.js backend is running (from the project root):

```bash
npm run server
```

The backend should be running on `http://localhost:3001` (or the port specified in your `.env`).

### 4. Run the Shiny App

From the `app/` directory:

```bash
shiny run app.py
```

Or using Python directly:

```bash
python -m shiny run app.py
```

The Shiny app will start and display a URL (typically `http://127.0.0.1:8000`). Open this URL in your browser.

## How It Works

### Architecture

The Shiny app calls the existing Node.js backend API endpoints:

1. **POST /api/analyze/vision**: Analyzes the uploaded image using GPT-5 Chat Completions API
2. **POST /api/analyze/recyclability**: Determines recyclability and finds facilities using GPT-4.1 Responses API with web search

### Process Flow

1. **User Input**: User uploads an image, enters location, and optionally adds context
2. **Image Analysis**: Backend analyzes the image to identify materials, condition, and contaminants
3. **Recyclability Assessment**: Backend determines recyclability based on location and local regulations
4. **Facility Lookup**: Backend uses web search to find nearby recycling/disposal facilities
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

## File Structure

```
app/
├── app.py                 # Main Shiny application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── utils/
│   ├── __init__.py
│   └── api_client.py     # API call functions
└── static/
    └── styles.css        # Custom CSS styling
```

## Troubleshooting

### Backend Connection Errors

If you see errors about connecting to the backend:

1. Ensure the Node.js backend server is running
2. Check that `BACKEND_URL` in `.env` matches the backend server URL
3. Verify the backend is accessible at the specified URL

### Map Not Displaying

If the map doesn't appear:

1. Check that `MAPBOX_ACCESS_TOKEN` is set in `.env`
2. Verify the token is valid and has the necessary permissions
3. Check browser console for JavaScript errors

### Image Upload Issues

If image upload fails:

1. Ensure the image file is a valid image format (JPEG, PNG, GIF, WebP)
2. Check that the file size is reasonable (backend may have limits)
3. Verify the backend server is running and accessible

## Differences from React App

This Shiny version replicates the React app's functionality but uses:

- **Shiny framework** instead of React
- **Python** instead of TypeScript
- **Shiny reactive values** instead of React hooks
- **Shiny UI components** instead of React components
- **Custom CSS** instead of Tailwind CSS (though classes are similar)

The user experience and functionality remain the same.

## Development

To modify the app:

1. Edit `app.py` for main application logic
2. Edit `utils/api_client.py` for API call logic
3. Edit `static/styles.css` for styling
4. Restart the Shiny app to see changes

## License

Same as main project.

