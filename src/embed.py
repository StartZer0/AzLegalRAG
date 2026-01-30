"""
Embedding and vector store module for AzLegalRAG.
Uses BAAI/bge-m3 embeddings with ChromaDB for storage.
"""

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from .config import EMBEDDING_MODEL, VECTORSTORE_DIR
except ImportError:
    from config import EMBEDDING_MODEL, VECTORSTORE_DIR


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embeddings():
    """Initialize embedding model."""
    device = get_device()
    print(f"Loading embedding model: {EMBEDDING_MODEL} on {device}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vectorstore(documents, persist_dir=None):
    """
    Create ChromaDB vectorstore from documents.
    
    Args:
        documents: List of LangChain Document objects
        persist_dir: Directory to persist the vectorstore
    
    Returns:
        Chroma vectorstore instance
    """
    persist_dir = persist_dir or VECTORSTORE_DIR
    embeddings = get_embeddings()
    
    print(f"Creating vectorstore with {len(documents)} documents...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Vectorstore created and persisted to {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir=None):
    """Load existing vectorstore from disk."""
    persist_dir = persist_dir or VECTORSTORE_DIR
    print(f"Loading vectorstore from {persist_dir}")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings()
    )


if __name__ == "__main__":
    # Test embedding
    embeddings = get_embeddings()
    test_text = "Emek Mecellesi Azerbaycan Respublikasi"
    result = embeddings.embed_query(test_text)
    print(f"Embedding dimension: {len(result)}")
