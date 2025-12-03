"""API client for calling backend endpoints."""
import httpx
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3001")


async def analyze_vision(image_base64: str) -> Dict[str, Any]:
    """
    Call the vision analysis endpoint.
    
    Args:
        image_base64: Base64-encoded image string (with or without data URL prefix)
        
    Returns:
        VisionResponse dictionary with primaryMaterial, secondaryMaterials, category, etc.
        
    Raises:
        httpx.HTTPError: If the API call fails
    """
    url = f"{BACKEND_URL}/api/analyze/vision"
    
    # Use a longer timeout for long-running operations (increased from 120 to 180)
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            url,
            json={"image": image_base64},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", {})


async def analyze_recyclability(
    vision_result: Dict[str, Any],
    location: str,
    context: str = ""
) -> Dict[str, Any]:
    """
    Call the recyclability analysis endpoint.
    
    Args:
        vision_result: VisionResponse dictionary from analyze_vision
        location: User's location (city, state, or ZIP code)
        context: Optional additional context from user
        
    Returns:
        AnalyzeResponse dictionary with isRecyclable, bin, instructions, facilities, etc.
        
    Raises:
        httpx.HTTPError: If the API call fails
    """
    url = f"{BACKEND_URL}/api/analyze/recyclability"
    
    # Use a longer timeout for long-running operations (increased from 120 to 180)
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            url,
            json={
                "visionResult": vision_result,
                "location": location,
                "context": context
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", {})

