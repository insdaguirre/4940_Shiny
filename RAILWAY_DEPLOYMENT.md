# Railway Deployment Guide for RecycLens

This guide walks you through deploying the RecycLens application on Railway.

## Architecture

The application consists of two services:
1. **Backend Service** (Node.js/TypeScript) - API server
2. **Frontend Service** (Shiny/Python) - Web application

## Prerequisites

- Railway account (sign up at https://railway.app)
- GitHub repository with your code
- OpenAI API key
- (Optional) Mapbox access token

## Step-by-Step Deployment

### 1. Connect Repository to Railway

1. Go to https://railway.app and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your `ReCycleIQ` repository
5. Railway will create a new project

### 2. Create Backend Service

1. In your Railway project, click "New Service"
2. Select "GitHub Repo" and choose your repository
3. Railway will auto-detect it as a Node.js service
4. **Important**: Leave the root directory as `/` (default)

**Configure Environment Variables:**
- Click on the service → "Variables" tab
- Add the following:
  - `OPENAI_API_KEY` = `sk-your-actual-api-key-here` (REQUIRED)
  - `RAG_SERVICE_URL` = (Set after creating RAG service - see step 4)
  - `NODE_ENV` = `production`
  - `PORT` = (Leave empty - Railway auto-assigns)

**Note**: Railway will automatically assign a port. The service will be available at a public URL like `https://your-backend-service.railway.app`

### 3. Create Frontend Service

1. In the same Railway project, click "New Service" again
2. Select "GitHub Repo" and choose the same repository
3. **Important**: Set the root directory to `/app`
   - Click on the service → Settings → Root Directory → Set to `app`
4. Railway should auto-detect it as a Python service

**Configure Environment Variables:**
- Click on the service → "Variables" tab
- Add the following:
  - `BACKEND_URL` = Use one of these options:
    - **Option A (Recommended)**: Use Railway's service reference
      - `${{backend.RAILWAY_PUBLIC_DOMAIN}}` (if backend has public domain)
      - Or use the actual public URL: `https://your-backend-service.railway.app`
    - **Option B**: Use private networking (if available)
      - `http://backend.railway.internal:3001`
  - `MAPBOX_ACCESS_TOKEN` = `pk.your-mapbox-token-here` (OPTIONAL)
  - `PORT` = (Leave empty - Railway auto-assigns)

**To find your backend URL:**
1. Go to your backend service
2. Click on "Settings" → "Networking"
3. Copy the public domain (e.g., `https://your-backend-service.railway.app`)
4. Use this as your `BACKEND_URL` in the frontend service

### 4. Create RAG Service

1. In the same Railway project, click "New Service" again
2. Select "GitHub Repo" and choose the same repository
3. **Important**: Leave the root directory as `/` (root of repository)
   - The RAG service needs access to both `rag_service/` and `rag/` directories
   - Do NOT set a custom root directory for this service
4. Railway should auto-detect it as a Python service

**Configure Environment Variables:**
- Click on the service → "Variables" tab
- Add the following:
  - `OPENAI_API_KEY` = `sk-your-actual-api-key-here` (REQUIRED - needed for embeddings)
  - `PORT` = (Leave empty - Railway auto-assigns)

**Get RAG Service URL:**
1. Go to your RAG service
2. Click on "Settings" → "Networking"
3. Copy the public domain (e.g., `https://rag-service.railway.app`)
4. Use this as your `RAG_SERVICE_URL` in the backend service (update backend environment variables)

### 5. Configure Service Networking

**For Frontend to Backend Communication:**

The frontend needs to know the backend URL. You have two options:

**Option A: Use Public Domain (Easiest)**
- Get the backend service's public domain from Railway
- Set `BACKEND_URL` in frontend service to: `https://your-backend-service.railway.app`
- Make sure to use `https://` not `http://`

**Option B: Use Railway Service Variables (Advanced)**
- In frontend service variables, you can reference the backend service
- Railway provides service references like `${{backend.RAILWAY_PUBLIC_DOMAIN}}`
- However, this may require Railway Pro plan

### 6. Deploy and Verify

1. Both services will automatically deploy when you push to GitHub
2. Or manually trigger deployment by clicking "Deploy" in each service
3. Wait for builds to complete (check the "Deployments" tab)

**Verify Backend:**
- Go to backend service → "Settings" → "Networking"
- Copy the public domain
- Visit `https://your-backend-domain.railway.app/health`
- Should return: `{"status":"ok","timestamp":"..."}`

**Verify Frontend:**
- Go to frontend service → "Settings" → "Networking"
- Copy the public domain
- Visit the URL in your browser
- You should see the RecycLens application

**Verify RAG Service:**
- Go to RAG service → "Settings" → "Networking"
- Copy the public domain
- Visit `https://your-rag-service.railway.app/health`
- Should return: `{"status":"ok","service":"rag-service",...}`

### 7. Test the Application

1. Open the frontend URL in your browser
2. Upload an image
3. Enter a location
4. Click "Check if it's Recyclable"
5. Verify that analysis works (frontend should communicate with backend)

## Troubleshooting

### Backend won't start
- Check that `OPENAI_API_KEY` is set correctly
- Check deployment logs in Railway dashboard
- Verify the service is using root directory `/`

### Frontend won't start
- Check that `BACKEND_URL` is set correctly
- Verify the service root directory is set to `/app`
- Check deployment logs for Python/Shiny errors
- Make sure `BACKEND_URL` uses `https://` not `http://`

### Frontend can't connect to backend
- Verify `BACKEND_URL` in frontend service matches backend's public domain
- Check that backend service is running and healthy
- Test backend health endpoint: `https://your-backend.railway.app/health`
- Check CORS settings (backend should allow all origins with `cors()`)

### Build failures
- Check deployment logs in Railway dashboard
- Verify all dependencies are in `requirements.txt` and `package.json`
- Make sure Python version is compatible (Railway auto-detects)

## Environment Variables Summary

### Backend Service
- `OPENAI_API_KEY` (required) - Your OpenAI API key
- `RAG_SERVICE_URL` (optional) - URL of RAG service (e.g., `https://rag-service.railway.app`)
  - If not set, RAG queries will be skipped (graceful degradation)
- `NODE_ENV` = `production` (optional but recommended)
- `PORT` - Auto-assigned by Railway

### Frontend Service
- `BACKEND_URL` (required) - Backend service public URL
- `MAPBOX_ACCESS_TOKEN` (optional) - For map functionality
- `PORT` - Auto-assigned by Railway

### RAG Service
- `OPENAI_API_KEY` (required) - For embeddings (same key as backend)
- `PORT` - Auto-assigned by Railway

## Files Created for Deployment

- `railway.json` (root) - Backend service configuration
- `app/railway.json` - Frontend service configuration
- `rag_service/railway.json` - RAG service configuration
- `rag_service/Procfile` - Process file for RAG service
- `app/nixpacks.toml` - Alternative build configuration for frontend
- `app/Procfile` - Process file for frontend (alternative)

## Next Steps

After deployment:
1. Set up custom domains (optional, Railway Pro feature)
2. Configure monitoring and alerts
3. Set up CI/CD for automatic deployments
4. Consider adding health check endpoints

## Support

If you encounter issues:
1. Check Railway deployment logs
2. Verify all environment variables are set
3. Test backend health endpoint
4. Check Railway status page for service issues

