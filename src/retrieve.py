"""
Retrieval module for AzLegalRAG.
Implements semantic search and hybrid search strategies.
"""

try:
    from .config import TOP_K
except ImportError:
    from config import TOP_K


def semantic_search(vectorstore, query, k=None):
    """
    Perform semantic similarity search.
    
    Args:
        vectorstore: Chroma vectorstore instance
        query: User query string
        k: Number of results to return
    
    Returns:
        List of relevant documents
    """
    k = k or TOP_K
    return vectorstore.similarity_search(query, k=k)


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
    return vectorstore.max_marginal_relevance_search(
        query, 
        k=k, 
        fetch_k=fetch_k
    )


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
    return vectorstore.similarity_search_with_score(query, k=k)


def format_context(documents):
    """Format retrieved documents as context string."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(context_parts)


if __name__ == "__main__":
    from embed import load_vectorstore
    
    vs = load_vectorstore()
    results = semantic_search(vs, "Emek muqavilesi")
    print(f"Found {len(results)} results")
    for doc in results:
        print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")
