"""
Document ingestion module for AzLegalRAG.
Loads and chunks Azerbaijani legal documents from e-qanun corpus.
"""

from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

try:
    from .config import CHUNK_SIZE, CHUNK_OVERLAP, DATASET_NAME
except ImportError:
    from config import CHUNK_SIZE, CHUNK_OVERLAP, DATASET_NAME


def load_eqanun():
    """Load Azerbaijani legal corpus from HuggingFace."""
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Loaded {len(dataset)} documents")
    return dataset


def chunk_documents(dataset, chunk_size=None, chunk_overlap=None):
    """
    Chunk legal documents with structure-aware splitting.
    
    Args:
        dataset: HuggingFace dataset with 'text' and 'id' fields
        chunk_size: Size of each chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)
    
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
        
        # Skip empty documents
        if not item.get("text") or len(item["text"].strip()) < 50:
            continue
            
        doc = Document(
            page_content=item["text"],
            metadata={
                "id": item["id"],
                "source": f"https://e-qanun.az/framework/{item['id']}"
            }
        )
        documents.extend(splitter.split_documents([doc]))
    
    print(f"Created {len(documents)} chunks from {total} documents")
    return documents


if __name__ == "__main__":
    # Test loading
    ds = load_eqanun()
    print(f"Sample document ID: {ds[0]['id']}")
    print(f"Sample text preview: {ds[0]['text'][:200]}...")
