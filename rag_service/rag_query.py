"""RAG query logic for querying recycling regulations."""
import os
from pathlib import Path
from typing import Optional
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.retrievers import BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import dotenv

dotenv.load_dotenv()

# Path to the vector store
# For Railway deployment: Index is copied into rag_service/rag_index_morechunked/
# For local development: Index is also available at ../rag/rag_index_morechunked/
RAG_INDEX_PATH = Path(__file__).parent / "rag_index_morechunked"

# Global cache for the query engine and index
_query_engine: Optional[RetrieverQueryEngine] = None
_index = None


def get_rag_query_engine() -> RetrieverQueryEngine:
    """Load the RAG index and return a query engine (cached singleton)."""
    global _query_engine, _index
    
    if _query_engine is not None:
        return _query_engine
    
    # Initialize OpenAI embedding model and LLM
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for RAG service. "
            "Please set it in your Railway environment variables."
        )
    
    Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key)
    Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
    print(f"✓ Initialized OpenAI embeddings and LLM for RAG queries")
    
    if not RAG_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"RAG index not found at {RAG_INDEX_PATH}. "
            f"Current working directory: {os.getcwd()}"
        )
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(RAG_INDEX_PATH))
        _index = load_index_from_storage(storage_context)
        # Increase similarity_top_k to retrieve more relevant chunks
        _query_engine = _index.as_query_engine(similarity_top_k=10)
        print(f"✓ RAG query engine loaded successfully from {RAG_INDEX_PATH}")
        return _query_engine
    except Exception as e:
        raise RuntimeError(f"Failed to load RAG index: {str(e)}")


def get_rag_retriever() -> BaseRetriever:
    """Get a retriever for direct chunk retrieval (bypasses LLM synthesis)."""
    global _index
    
    # Ensure index is loaded
    if _index is None:
        get_rag_query_engine()
    
    if _index is None:
        raise RuntimeError("Failed to load RAG index")
    
    # Create retriever with higher top_k for better coverage
    retriever = _index.as_retriever(similarity_top_k=15)
    return retriever


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
        # Simplified query that matches document style for better embedding similarity
        def build_query(material_term: str) -> str:
            # Build a simple, keyword-rich query that matches how documents are written
            query_parts = [material_term, "recycling"]
            
            if county:
                query_parts.append(county.capitalize())
                query_parts.append("County")
            elif "ithaca" in location.lower() or "tompkins" in location.lower():
                query_parts.append("Tompkins")
                query_parts.append("County")
            elif "albany" in location.lower():
                query_parts.append("Albany")
                query_parts.append("County")
            
            # Add location context
            if "new york" in location.lower() or "ny" in location.lower():
                query_parts.append("New York")
            
            # Add condition/context if relevant
            if condition and condition.lower() not in ["unknown", "none", ""]:
                query_parts.append(condition)
            
            # Simple, direct query that will match document chunks
            return " ".join(query_parts)
        
        # Helper function to extract sources from nodes
        def extract_sources_from_nodes(nodes) -> list[str]:
            sources = []
            for node in nodes:
                if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                    metadata = node.node.metadata
                    if 'source_url' in metadata:
                        sources.append(metadata['source_url'])
                    elif 'source_file' in metadata:
                        sources.append(metadata['source_file'])
            return sources
        
        # Helper function to extract text from nodes
        def extract_text_from_nodes(nodes) -> str:
            texts = []
            for node in nodes:
                if hasattr(node, 'node') and hasattr(node.node, 'text'):
                    texts.append(node.node.text)
                elif hasattr(node, 'text'):
                    texts.append(node.text)
            return "\n\n".join(texts)
        
        # Try primary query with original material
        primary_query = build_query(material)
        print(f"RAG Query (primary): material={material}, expanded_terms={material_terms}, location={location}, county={county}")
        print(f"RAG Query text: {primary_query}")
        
        # First, try LLM synthesis with query engine
        response = query_engine.query(primary_query)
        response_text = str(response)
        
        # Get source nodes from response
        sources = []
        if hasattr(response, 'source_nodes'):
            sources = extract_sources_from_nodes(response.source_nodes)
        
        # Check if primary query was successful
        # Consider it successful if we have substantial text (>50 chars) OR sources
        primary_successful = (response_text and len(response_text.strip()) > 50) or len(sources) > 0
        
        # If LLM synthesis returned empty but we have sources, use retriever to get raw chunks
        if not primary_successful and len(sources) == 0:
            print(f"LLM synthesis returned empty, trying direct retrieval...")
            try:
                retriever = get_rag_retriever()
                retrieved_nodes = retriever.retrieve(primary_query)
                
                if retrieved_nodes:
                    print(f"Retrieved {len(retrieved_nodes)} chunks directly")
                    # Extract raw text from chunks
                    raw_text = extract_text_from_nodes(retrieved_nodes)
                    raw_sources = extract_sources_from_nodes(retrieved_nodes)
                    
                    if raw_text and len(raw_text.strip()) > 50:
                        response_text = raw_text
                        sources = raw_sources
                        primary_successful = True
                        print(f"Using raw retrieved chunks (text_len={len(raw_text)}, sources={len(sources)})")
                    else:
                        print(f"Retrieved chunks but text too short (text_len={len(raw_text)})")
                else:
                    print("No chunks retrieved from vector store")
            except Exception as e:
                print(f"Error using direct retriever: {e}")
                import traceback
                traceback.print_exc()
        
        # If primary query didn't return good results, try alternative terms
        if not primary_successful and len(material_terms) > 1:
            print(f"Primary query returned insufficient results (text_len={len(response_text)}, sources={len(sources)}), trying alternative terms...")
            
            # Try each alternative term (skip the first one as it's the original)
            for alt_term in material_terms[1:]:
                if alt_term.lower() == material.lower():
                    continue  # Skip if it's the same as original
                
                try:
                    alt_query = build_query(alt_term)
                    print(f"RAG Query (alternative): trying material={alt_term}, query={alt_query}")
                    
                    # Try LLM synthesis first
                    alt_response = query_engine.query(alt_query)
                    alt_text = str(alt_response)
                    alt_sources = []
                    if hasattr(alt_response, 'source_nodes'):
                        alt_sources = extract_sources_from_nodes(alt_response.source_nodes)
                    
                    # If LLM synthesis failed, try direct retrieval
                    alt_successful = (alt_text and len(alt_text.strip()) > 50) or len(alt_sources) > 0
                    
                    if not alt_successful:
                        print(f"LLM synthesis failed for '{alt_term}', trying direct retrieval...")
                        try:
                            retriever = get_rag_retriever()
                            retrieved_nodes = retriever.retrieve(alt_query)
                            
                            if retrieved_nodes:
                                raw_text = extract_text_from_nodes(retrieved_nodes)
                                raw_sources = extract_sources_from_nodes(retrieved_nodes)
                                
                                if raw_text and len(raw_text.strip()) > 50:
                                    alt_text = raw_text
                                    alt_sources = raw_sources
                                    alt_successful = True
                                    print(f"Direct retrieval succeeded for '{alt_term}' (text_len={len(raw_text)})")
                        except Exception as e:
                            print(f"Error using direct retriever for '{alt_term}': {e}")
                    
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
                    import traceback
                    traceback.print_exc()
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

