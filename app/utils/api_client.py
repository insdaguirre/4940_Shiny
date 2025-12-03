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
    
    # Use separate connect and read timeouts for long-running operations
    timeout = httpx.Timeout(connect=60.0, read=180.0)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json={"image": image_base64},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", {})
    except httpx.ConnectTimeout as e:
        raise Exception(f"Connection timeout: Could not connect to backend service. Please check if the backend is running.") from e
    except httpx.ReadTimeout as e:
        raise Exception(f"Request timeout: The vision analysis took too long. Please try again.") from e
    except httpx.HTTPError as e:
        raise Exception(f"HTTP error during vision analysis: {str(e)}") from e


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
    
    # Use separate connect and read timeouts for long-running operations
    timeout = httpx.Timeout(connect=60.0, read=180.0)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
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
    except httpx.ConnectTimeout as e:
        raise Exception(f"Connection timeout: Could not connect to backend service. Please check if the backend is running.") from e
    except httpx.ReadTimeout as e:
        raise Exception(f"Request timeout: The recyclability analysis took too long. Please try again.") from e
    except httpx.HTTPError as e:
        raise Exception(f"HTTP error during recyclability analysis: {str(e)}") from e

