"""
Embedding and vector store module for AzLegalRAG.
Uses Alibaba-NLP/gte-Qwen2-7B-instruct (77.18% NDCG on Azerbaijani).
Custom embedding class for proper 7B model loading.
"""

import gc
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
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


class GteQwen2Embeddings(Embeddings):
    """
    Custom embeddings class for gte-Qwen2-7B-instruct.
    
    Uses AutoModel with last-token pooling and instruction prefix
    as recommended by Alibaba for this model.
    """
    
    def __init__(self, model_name: str = None, device: str = None, batch_size: int = 8):
        self.model_name = model_name or EMBEDDING_MODEL
        self.device = device or get_device()
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading embedding model: {self.model_name}")
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("Embedding model loaded successfully!")
    
    def _embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Embed a batch of texts."""
        # Add instruction prefix for queries (as per GTE-Qwen2 docs)
        if is_query:
            texts = [f"Instruct: Given a query, retrieve passages that answer the query.\nQuery: {t}" for t in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last token pooling (recommended for GTE-Qwen2)
            embeddings = outputs.last_hidden_state[:, -1, :]
            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().float().tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_batch(batch, is_query=False)
            all_embeddings.extend(embeddings)
            
            if (i // self.batch_size) % 100 == 0 and i > 0:
                print(f"Embedded {i}/{len(texts)} documents")
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._embed_batch([text], is_query=True)[0]
    
    def unload(self):
        """Unload model to free GPU memory."""
        print("Unloading embedding model...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        clear_gpu_memory()


# Global instance for singleton pattern
_embeddings_instance = None


def get_embeddings(device=None):
    """Get or create embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = GteQwen2Embeddings(device=device)
    return _embeddings_instance


def unload_embeddings(embeddings=None):
    """Unload embedding model to free GPU memory."""
    global _embeddings_instance
    if embeddings is not None:
        embeddings.unload()
    if _embeddings_instance is not None:
        _embeddings_instance.unload()
        _embeddings_instance = None


def create_vectorstore(documents, persist_dir=None, embeddings=None):
    """
    Create ChromaDB vectorstore from documents.
    
    Args:
        documents: List of LangChain Document objects
        persist_dir: Directory to persist the vectorstore
        embeddings: Optional pre-loaded embeddings
    
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
    print(f"This may take 30-60 minutes for 465K chunks...")
    
    # Process in batches to avoid memory issues
    batch_size = 500
    vectorstore = None
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({i}/{len(documents)})")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir
            )
        else:
            vectorstore.add_documents(batch)
        
        # Clear cache periodically
        if i % 5000 == 0 and i > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
