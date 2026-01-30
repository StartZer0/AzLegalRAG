"""
Document ingestion module for AzLegalRAG.
Loads and chunks Azerbaijani legal documents from e-qanun corpus.
"""

import unicodedata
import re
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from .config import CHUNK_SIZE, CHUNK_OVERLAP, DATASET_NAME
except ImportError:
    from config import CHUNK_SIZE, CHUNK_OVERLAP, DATASET_NAME


def normalize_azerbaijani_text(text):
    """
    Clean Azerbaijani text from bad PDF extraction.
    
    The source dataset has characters split across lines at 
    Azerbaijani special characters (ə, ü, ö, ş, ç, ğ, ı).
    This function joins them back into proper words.
    """
    if not text:
        return ""
    
    # NFC normalization for combining characters
    text = unicodedata.normalize('NFC', text)
    
    # Azerbaijani alphabet (Latin + special chars)
    az_chars = r'[a-zA-ZəüöşçğıƏÜÖŞÇĞI0-9]'
    
    # Aggressively join letters separated by newlines
    # This handles the case: "m\nü\nqavil\nə" → "müqavilə"
    for _ in range(20):  # Multiple passes for chained splits
        prev = text
        text = re.sub(f'({az_chars})\\n({az_chars})', r'\1\2', text)
        if text == prev:
            break
    
    # Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Clean up multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def load_eqanun():
    """Load Azerbaijani legal corpus from HuggingFace."""
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Loaded {len(dataset)} documents")
    return dataset


def chunk_documents(dataset, chunk_size=None, chunk_overlap=None, normalize=True):
    """
    Chunk legal documents with structure-aware splitting.
    
    Args:
        dataset: HuggingFace dataset with 'text' and 'id' fields
        chunk_size: Size of each chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)
        normalize: If True, clean up text before chunking (fixes split chars)
    
    Returns:
        List of LangChain Document objects with metadata
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    documents = []
    total = len(dataset)
    
    for i, item in enumerate(dataset):
        if i % 5000 == 0:
            print(f"Processing document {i}/{total}")
        
        text = item.get("text", "")
        
        # Skip empty documents
        if not text or len(text.strip()) < 50:
            continue
        
        # Normalize text if requested (fixes split Azerbaijani chars)
        if normalize:
            text = normalize_azerbaijani_text(text)
            
        doc = Document(
            page_content=text,
            metadata={
                "id": item["id"],
                "source": f"https://e-qanun.az/framework/{item['id']}"
            }
        )
        documents.extend(splitter.split_documents([doc]))
    
    print(f"Created {len(documents)} chunks from {total} documents")
    return documents


if __name__ == "__main__":
    # Test loading and normalization
    ds = load_eqanun()
    print(f"Sample document ID: {ds[0]['id']}")
    
    # Test normalization
    raw_text = ds[0]['text'][:200]
    clean_text = normalize_azerbaijani_text(raw_text)
    print(f"\n=== RAW ===\n{raw_text}")
    print(f"\n=== CLEANED ===\n{clean_text}")
