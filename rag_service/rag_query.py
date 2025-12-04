"""RAG query logic for querying recycling regulations."""
import os
from pathlib import Path
from typing import Optional, Any
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
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
    # Use a valid, supported OpenAI chat model for LlamaIndex metadata/synthesis
    Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-4.1")
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


def get_rag_retriever() -> Any:
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
    Query RAG for recycling information using direct vector retrieval only.

    This bypasses LLM synthesis and always returns raw chunks from the index.
    """
    try:
        # Ensure index and retriever are ready
        get_rag_query_engine()  # loads _index and configures Settings
        retriever = get_rag_retriever()

        county = extract_county_from_location(location)
        material_terms = normalize_and_expand_material(material)

        def build_query(term: str) -> str:
            parts = [term, "recycling"]
            if county:
                parts += [county.capitalize(), "County"]
            if "new york" in location.lower() or "ny" in location.lower():
                parts.append("New York")
            if condition and condition.lower() not in ["unknown", "none", ""]:
                parts.append(condition)
            return " ".join(parts)

        def extract_sources_from_nodes(nodes) -> list[str]:
            out: list[str] = []
            for node in nodes:
                n = getattr(node, "node", node)
                md = getattr(n, "metadata", {}) or {}
                if "source_url" in md:
                    out.append(md["source_url"])
                elif "source_file" in md:
                    out.append(md["source_file"])
            return out

        def extract_text_from_nodes(nodes) -> str:
            texts: list[str] = []
            for node in nodes:
                n = getattr(node, "node", node)
                txt = getattr(n, "text", None)
                if txt:
                    texts.append(txt)
            return "\n\n".join(texts)

        best_text = ""
        best_sources: list[str] = []

        for term in material_terms:
            q = build_query(term)
            print(f"RAG RAW RETRIEVAL: term={term}, query={q}")
            try:
                nodes = retriever.retrieve(q)
            except Exception as e:
                print(f"Error retrieving for term '{term}': {e}")
                continue

            if not nodes:
                print(f"No nodes retrieved for term '{term}'")
                continue

            raw_text = extract_text_from_nodes(nodes)
            raw_sources = extract_sources_from_nodes(nodes)
            print(
                f"Term '{term}' → {len(nodes)} nodes, "
                f"text_len={len(raw_text)}, sources={len(raw_sources)}"
            )

            if raw_text and raw_text.strip():
                if (len(raw_text) > len(best_text)) or (
                    len(raw_sources) > len(best_sources)
                ):
                    best_text = raw_text
                    best_sources = raw_sources

            if best_text and best_sources:
                break

        print(
            f"RAG RAW RESPONSE: regulations_length={len(best_text)}, "
            f"sources_count={len(best_sources)}"
        )
        if best_sources:
            print(f"RAG RAW Sources: {best_sources}")
        else:
            print("RAG RAW: No sources found")

        return best_text or "", best_sources

    except FileNotFoundError as e:
        print(f"RAG index not found: {e}")
        return "", []
    except Exception as e:
        print(f"RAG query error (raw mode): {e}")
        import traceback
        traceback.print_exc()
        return "", []

