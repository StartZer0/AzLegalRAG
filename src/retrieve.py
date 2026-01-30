"""
Retrieval module for AzLegalRAG.
Implements semantic search and hybrid search strategies.
"""

import unicodedata

try:
    from .config import TOP_K
except ImportError:
    from config import TOP_K


def normalize_text(text):
    """
    Normalize Azerbaijani text to fix decomposed Unicode characters.
    Converts NFD (decomposed) to NFC (composed) form.
    e.g., 'müqavilə' stored as separate chars becomes proper 'müqavilə'
    """
    if text is None:
        return ""
    # NFC normalization: combines base characters with combining marks
    normalized = unicodedata.normalize('NFC', text)
    # Also remove any remaining zero-width characters
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn' or unicodedata.combining(c) == 0)
    return normalized


def semantic_search(vectorstore, query, k=None):
    """
    Perform semantic similarity search.
    
    Args:
        vectorstore: Chroma vectorstore instance
        query: User query string
        k: Number of results to return
    
    Returns:
        List of relevant documents with normalized text
    """
    k = k or TOP_K
    results = vectorstore.similarity_search(query, k=k)
    # Normalize text in results
    for doc in results:
        doc.page_content = normalize_text(doc.page_content)
    return results


def mmr_search(vectorstore, query, k=None, fetch_k=None):
    """
    Perform Maximum Marginal Relevance search for diverse results.
    
    Args:
        vectorstore: Chroma vectorstore instance
        query: User query string
        k: Number of results to return
        fetch_k: Number of candidates to fetch before reranking
    
    Returns:
        List of relevant and diverse documents
    """
    k = k or TOP_K
    fetch_k = fetch_k or k * 3
    results = vectorstore.max_marginal_relevance_search(
        query, 
        k=k, 
        fetch_k=fetch_k
    )
    # Normalize text in results
    for doc in results:
        doc.page_content = normalize_text(doc.page_content)
    return results


def search_with_scores(vectorstore, query, k=None):
    """
    Search with similarity scores.
    
    Args:
        vectorstore: Chroma vectorstore instance
        query: User query string
        k: Number of results to return
    
    Returns:
        List of (document, score) tuples
    """
    k = k or TOP_K
    results = vectorstore.similarity_search_with_score(query, k=k)
    # Normalize text in results
    normalized_results = []
    for doc, score in results:
        doc.page_content = normalize_text(doc.page_content)
        normalized_results.append((doc, score))
    return normalized_results


def format_context(documents):
    """Format retrieved documents as context string with normalized text."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        content = normalize_text(doc.page_content)
        context_parts.append(f"[{i}] Source: {source}\n{content}")
    return "\n\n".join(context_parts)


if __name__ == "__main__":
    from embed import load_vectorstore
    
    vs = load_vectorstore()
    results = semantic_search(vs, "Emek muqavilesi")
    print(f"Found {len(results)} results")
    for doc in results:
        print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")
