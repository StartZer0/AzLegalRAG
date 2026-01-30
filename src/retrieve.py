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

# Azerbaijani special characters that often get split to separate lines
AZ_SPECIAL_CHARS = set('əüöşçğıƏÜÖŞÇĞI')


def normalize_text(text):
    """
    Fix Azerbaijani text with split characters from bad PDF extraction.
    
    The source dataset has Azerbaijani special characters (ə, ü, ö, ş, ç, ğ, ı)
    split onto separate lines. This function:
    1. Joins lines that contain only special characters to the previous line
    2. Applies Unicode NFC normalization
    """
    if not text:
        return ""
    
    # First: NFC normalization for combining characters
    text = unicodedata.normalize('NFC', text)
    
    # Split into lines
    lines = text.split('\n')
    result = []
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            # Keep empty lines for paragraph structure
            if result and result[-1] != '':
                result.append('')
            continue
        
        # Check if this line contains only Azerbaijani special characters
        # These should be joined to the previous line
        is_only_special = all(c in AZ_SPECIAL_CHARS for c in stripped)
        
        # Also check for very short lines (1-2 chars) that are likely split
        is_very_short = len(stripped) <= 2
        
        if (is_only_special or is_very_short) and result:
            # Join to previous non-empty line
            for i in range(len(result) - 1, -1, -1):
                if result[i]:
                    result[i] += stripped
                    break
            else:
                result.append(stripped)
        else:
            result.append(stripped)
    
    # Join with newlines
    fixed_text = '\n'.join(result)
    
    # Clean up multiple consecutive newlines
    fixed_text = re.sub(r'\n{3,}', '\n\n', fixed_text)
    
    return fixed_text


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
