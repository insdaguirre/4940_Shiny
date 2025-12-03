"""RAG query logic for querying recycling regulations."""
import os
from pathlib import Path
from typing import Optional
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
import dotenv

dotenv.load_dotenv()

# Path to the vector store - adjust based on service location
# In Railway, rag_service is at root, so rag/ is at ../rag/
# Locally, if running from rag_service/, rag/ is at ../rag/
RAG_INDEX_PATH = Path(__file__).parent.parent / "rag" / "rag_index_morechunked"

# Global cache for the query engine
_query_engine: Optional[RetrieverQueryEngine] = None


def get_rag_query_engine() -> RetrieverQueryEngine:
    """Load the RAG index and return a query engine (cached singleton)."""
    global _query_engine
    
    if _query_engine is not None:
        return _query_engine
    
    if not RAG_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"RAG index not found at {RAG_INDEX_PATH}. "
            f"Current working directory: {os.getcwd()}"
        )
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(RAG_INDEX_PATH))
        index = load_index_from_storage(storage_context)
        _query_engine = index.as_query_engine(similarity_top_k=3)
        return _query_engine
    except Exception as e:
        raise RuntimeError(f"Failed to load RAG index: {str(e)}")


def extract_county_from_location(location: str) -> Optional[str]:
    """
    Extract county name from location string.
    
    Args:
        location: Location string (e.g., "Ithaca, NY", "Albany, NY 12201")
        
    Returns:
        County name ("albany" or "tompkins") or None if not detected
    """
    location_lower = location.lower()
    
    # Check for albany
    if "albany" in location_lower:
        return "albany"
    
    # Check for tompkins (Ithaca is in Tompkins County)
    if "tompkins" in location_lower or "ithaca" in location_lower:
        return "tompkins"
    
    return None


def query_rag(
    material: str,
    location: str,
    condition: str = "",
    context: str = ""
) -> tuple[str, list[str]]:
    """
    Query RAG for recycling information.
    
    Args:
        material: Primary material (e.g., "Plastic", "Glass", "Metal")
        location: User location (e.g., "Ithaca, NY", "Albany, NY 12201")
        condition: Item condition (e.g., "clean", "soiled", "damaged")
        context: Additional context from user
        
    Returns:
        Tuple of (regulations_text, sources_list)
        Returns empty string and empty list if query fails
    """
    try:
        query_engine = get_rag_query_engine()
        
        # Extract county from location if possible
        county = extract_county_from_location(location)
        
        # Build query
        query_parts = [
            f"What are the recycling regulations for {material} in {location}?",
        ]
        
        if condition:
            query_parts.append(f"Item condition: {condition}")
        
        if context:
            query_parts.append(f"Additional context: {context}")
        
        query_parts.extend([
            "",
            "Please provide:",
            "1. Is this item recyclable in this location?",
            "2. What are the specific requirements (cleaning, preparation)?",
            "3. Which bin should it go in?",
            "4. Any special instructions or restrictions?",
        ])
        
        if county:
            query_parts.append(f"\nFocus on regulations for {county.capitalize()} County, New York.")
        
        query = "\n".join(query_parts)
        
        # Log the query being executed
        print(f"RAG Query: material={material}, location={location}, county={county}")
        print(f"RAG Query text: {query[:200]}...")  # Log first 200 chars
        
        # Execute query
        response = query_engine.query(query)
        response_text = str(response)
        
        # Extract sources from response metadata if available
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                    metadata = node.node.metadata
                    if 'source_url' in metadata:
                        sources.append(metadata['source_url'])
                    elif 'source_file' in metadata:
                        sources.append(metadata['source_file'])
        
        # Log what was returned
        print(f"RAG Response: regulations_length={len(response_text)}, sources_count={len(sources)}")
        if sources:
            print(f"RAG Sources: {sources}")
        
        return response_text, sources
        
    except FileNotFoundError as e:
        print(f"RAG index not found: {e}")
        return "", []
    except Exception as e:
        print(f"RAG query error: {e}")
        import traceback
        traceback.print_exc()
        return "", []

