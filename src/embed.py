"""
Embedding and vector store module for AzLegalRAG.
Uses Alibaba-NLP/gte-Qwen2-7B-instruct (77.18% NDCG on Azerbaijani).
Implements sequential loading to fit on L4 GPU (24GB).
"""

import gc
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from .config import EMBEDDING_MODEL, VECTORSTORE_DIR, EMBEDDING_DIM
except ImportError:
    from config import EMBEDDING_MODEL, VECTORSTORE_DIR, EMBEDDING_DIM


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_gpu_memory():
    """Clear GPU memory for sequential loading."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("GPU memory cleared")


def get_embeddings(device=None):
    """
    Initialize gte-Qwen2-7B-instruct embedding model.
    
    This is a 7B parameter model (~14GB VRAM).
    Use sequential loading - unload before loading LLM.
    """
    device = device or get_device()
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": device,
            "torch_dtype": torch.float16  # Use float16 for memory efficiency
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 8  # Smaller batch for large model
        }
    )


def unload_embeddings(embeddings):
    """Unload embedding model to free GPU memory."""
    print("Unloading embedding model...")
    del embeddings
    clear_gpu_memory()


def create_vectorstore(documents, persist_dir=None, embeddings=None):
    """
    Create ChromaDB vectorstore from documents.
    
    Args:
        documents: List of LangChain Document objects
        persist_dir: Directory to persist the vectorstore
        embeddings: Optional pre-loaded embeddings (for batch processing)
    
    Returns:
        Chroma vectorstore instance
    """
    persist_dir = persist_dir or VECTORSTORE_DIR
    
    # Load embeddings if not provided
    should_unload = False
    if embeddings is None:
        embeddings = get_embeddings()
        should_unload = True
    
    print(f"Creating vectorstore with {len(documents)} documents...")
    print(f"This may take 30-60 minutes for 50K documents...")
    
    # Process in batches to avoid memory issues
    batch_size = 500
    vectorstore = None
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir
            )
        else:
            vectorstore.add_documents(batch)
    
    print(f"Vectorstore created and persisted to {persist_dir}")
    
    if should_unload:
        unload_embeddings(embeddings)
    
    return vectorstore


def load_vectorstore(persist_dir=None, load_embeddings=True):
    """
    Load existing vectorstore from disk.
    
    Args:
        persist_dir: Directory where vectorstore is persisted
        load_embeddings: Whether to load embedding model (needed for queries)
    
    Returns:
        Chroma vectorstore instance
    """
    persist_dir = persist_dir or VECTORSTORE_DIR
    print(f"Loading vectorstore from {persist_dir}")
    
    if load_embeddings:
        embeddings = get_embeddings()
    else:
        embeddings = None
    
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )


if __name__ == "__main__":
    # Test embedding
    print("Testing gte-Qwen2-7B-instruct loading...")
    embeddings = get_embeddings()
    test_text = "Emek Mecellesi Azerbaycan Respublikasi"
    result = embeddings.embed_query(test_text)
    print(f"Embedding dimension: {len(result)} (expected: {EMBEDDING_DIM})")
    unload_embeddings(embeddings)
