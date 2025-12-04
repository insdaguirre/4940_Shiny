"""RAG query logic for querying recycling regulations."""
import os
from pathlib import Path
from typing import Optional
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
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
    
    # Initialize OpenAI embedding model
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for RAG service. "
            "Please set it in your Railway environment variables."
        )
    
    Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key)
    print(f"✓ Initialized OpenAI embeddings for RAG queries")
    
    if not RAG_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"RAG index not found at {RAG_INDEX_PATH}. "
            f"Current working directory: {os.getcwd()}"
        )
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(RAG_INDEX_PATH))
        index = load_index_from_storage(storage_context)
        # Increase similarity_top_k to retrieve more relevant chunks
        _query_engine = index.as_query_engine(similarity_top_k=5)
        print(f"✓ RAG query engine loaded successfully from {RAG_INDEX_PATH}")
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


def normalize_and_expand_material(material: str) -> list[str]:
    """
    Normalize and expand material names to improve RAG retrieval.
    
    Maps brand names, singular forms, and specific types to the terminology
    used in the RAG knowledge base.
    
    Args:
        material: Material name from vision service (e.g., "Battery", "Tupperware")
        
    Returns:
        List of material terms to search for (original + normalized/expanded terms)
    """
    material_lower = material.lower().strip()
    terms = [material]  # Always include original term
    
    # Battery normalization - map singular and specific types to "Batteries"
    if "battery" in material_lower:
        if "batteries" not in material_lower:
            terms.append("Batteries")
        # Also include specific battery types that might be in docs
        if "lithium" in material_lower:
            terms.extend(["Lithium batteries", "Batteries"])
        elif "alkaline" in material_lower:
            terms.extend(["Alkaline batteries", "Batteries"])
        elif "lead" in material_lower or "acid" in material_lower:
            terms.extend(["Car Batteries", "Lead acid batteries", "Batteries"])
        else:
            terms.append("Batteries")
    
    # Plastic container normalization - map brand names and generic terms
    plastic_indicators = [
        "tupperware", "rubbermaid", "gladware", "ziploc container",
        "plastic container", "plastic food container", "food storage container"
    ]
    
    if any(indicator in material_lower for indicator in plastic_indicators):
        terms.extend([
            "Plastic Containers",
            "plastic containers",
            "Plastic containers #1",
            "Plastic containers #2",
            "Plastic containers #5"
        ])
    
    # Generic plastic normalization
    if "plastic" in material_lower and "container" not in material_lower:
        terms.append("Plastic Containers")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)
    
    return unique_terms


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
        
        # Normalize and expand material terms for better retrieval
        material_terms = normalize_and_expand_material(material)
        
        # Helper function to build a query for a specific material term
        def build_query(material_term: str) -> str:
            query_parts = [
                f"What are the recycling regulations for {material_term} in {location}?",
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
            
            return "\n".join(query_parts)
        
        # Helper function to extract sources from a response
        def extract_sources_from_response(response) -> list[str]:
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                        metadata = node.node.metadata
                        if 'source_url' in metadata:
                            sources.append(metadata['source_url'])
                        elif 'source_file' in metadata:
                            sources.append(metadata['source_file'])
            return sources
        
        # Try primary query with original material
        primary_query = build_query(material)
        print(f"RAG Query (primary): material={material}, expanded_terms={material_terms}, location={location}, county={county}")
        print(f"RAG Query text: {primary_query[:300]}...")  # Log first 300 chars
        
        response = query_engine.query(primary_query)
        response_text = str(response)
        sources = extract_sources_from_response(response)
        
        # Check if primary query was successful
        # Consider it successful if we have substantial text (>50 chars) OR sources
        primary_successful = (response_text and len(response_text.strip()) > 50) or len(sources) > 0
        
        # If primary query didn't return good results, try alternative terms
        if not primary_successful and len(material_terms) > 1:
            print(f"Primary query returned insufficient results (text_len={len(response_text)}, sources={len(sources)}), trying alternative terms...")
            
            # Try each alternative term (skip the first one as it's the original)
            for alt_term in material_terms[1:]:
                if alt_term.lower() == material.lower():
                    continue  # Skip if it's the same as original
                
                try:
                    alt_query = build_query(alt_term)
                    print(f"RAG Query (alternative): trying material={alt_term}")
                    
                    alt_response = query_engine.query(alt_query)
                    alt_text = str(alt_response)
                    alt_sources = extract_sources_from_response(alt_response)
                    
                    # If this alternative query gives better results, use it
                    alt_successful = (alt_text and len(alt_text.strip()) > 50) or len(alt_sources) > 0
                    
                    if alt_successful:
                        # Use the alternative result if it's better
                        if len(alt_text.strip()) > len(response_text.strip()) or len(alt_sources) > len(sources):
                            response_text = alt_text
                            sources = alt_sources
                            print(f"Alternative query with '{alt_term}' returned better results")
                            # If we got good results, we can stop trying more alternatives
                            if len(alt_text.strip()) > 100 and len(alt_sources) > 0:
                                break
                    else:
                        print(f"Alternative query with '{alt_term}' also returned insufficient results")
                        
                except Exception as e:
                    print(f"Error querying with alternative term '{alt_term}': {e}")
                    continue
        
        # Log final results
        print(f"RAG Response (final): regulations_length={len(response_text)}, sources_count={len(sources)}")
        if sources:
            print(f"RAG Sources: {sources}")
        else:
            print("RAG Response: No sources found")
        
        return response_text, sources
        
    except FileNotFoundError as e:
        print(f"RAG index not found: {e}")
        return "", []
    except Exception as e:
        print(f"RAG query error: {e}")
        import traceback
        traceback.print_exc()
        return "", []

