"""
Retrieval module for AzLegalRAG.
Implements semantic search and hybrid search strategies.
"""

import unicodedata
import re

try:
    from .config import TOP_K
except ImportError:
    from config import TOP_K


def normalize_text(text):
    """
    Fix Azerbaijani text with split characters from bad PDF extraction.
    
    The source dataset has words broken across lines at special characters.
    This function aggressively joins lines that appear to be mid-word breaks.
    """
    if not text:
        return ""
    
    # First: NFC normalization for combining characters
    text = unicodedata.normalize('NFC', text)
    
    # Aggressive approach: Remove newlines that appear within words
    # Pattern: letter/special-char, newline, letter/special-char (no space/punct between)
    # This indicates a mid-word break that should be removed
    
    # Azerbaijani alphabet (Latin + special chars)
    az_chars = r'[a-zA-ZəüöşçğıƏÜÖŞÇĞI]'
    
    # Replace newlines between word characters with nothing (join them)
    # But preserve newlines that have punctuation/spaces around them
    
    # Step 1: Replace single newline between letters with nothing
    text = re.sub(f'({az_chars})\\n({az_chars})', r'\1\2', text)
    
    # Step 2: Repeat to catch chained single-char lines
    for _ in range(10):  # Multiple passes to handle chains
        prev = text
        text = re.sub(f'({az_chars})\\n({az_chars})', r'\1\2', text)
        if text == prev:
            break
    
    # Step 3: Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Step 4: Clean up multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


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
    # Test the normalize function
    test_text = """Kollektiv m
ü
qavil
ə
v
ə
sazi
ş
bir ild
ə
n
üç
il
ə
d
ə
k m
ü
dd
ə
t
ə
ba
ğ
lan
ı
la bil
ə
r."""
    
    print("=== ORIGINAL ===")
    print(test_text[:100])
    print()
    print("=== NORMALIZED ===")
    print(normalize_text(test_text))
