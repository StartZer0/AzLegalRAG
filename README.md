# AzLegalRAG

Azerbaijani Legal Document Q&A System using Retrieval-Augmented Generation (RAG).

## Overview

This project demonstrates RAG implementation for legal document retrieval and question answering, using the official Azerbaijani legal corpus from [e-qanun.az](https://e-qanun.az).

## Features

- Semantic search over 50,989 legal documents
- Natural language Q&A in Azerbaijani and English
- Source attribution with direct links to e-qanun.az
- Interactive Streamlit interface

## Tech Stack

| Component | Tool |
|-----------|------|
| Embedding | BAAI/bge-m3 (65.8% NDCG) |
| Vector DB | ChromaDB |
| LLM | Mistral-7B-Instruct-v0.2 |
| Framework | LangChain |
| UI | Streamlit |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build Vector Store (Run Once)

```python
from src.ingest import load_eqanun, chunk_documents
from src.embed import create_vectorstore

dataset = load_eqanun()
chunks = chunk_documents(dataset)
vectorstore = create_vectorstore(chunks)
```

### 3. Run Streamlit App

```bash
streamlit run src/app.py
```

## Project Structure

```
AzLegalRAG/
├── src/
│   ├── config.py       # Configuration
│   ├── ingest.py       # Data loading
│   ├── embed.py        # Embeddings
│   ├── retrieve.py     # Search
│   ├── generate.py     # LLM generation
│   └── app.py          # Streamlit UI
├── notebooks/
│   └── rag_colab.ipynb # Colab demo
├── vectorstore/        # ChromaDB storage
└── requirements.txt
```

## Usage Example

```python
from src.embed import load_vectorstore
from src.generate import create_rag_chain

vectorstore = load_vectorstore()
chain = create_rag_chain(vectorstore)

result = chain({"query": "Emek muqavilesi nedir?"})
print(result["result"])
```

## Dataset

**allmalab/eqanun**: 50,989 Azerbaijani legal documents from the official e-qanun.az portal.

## Citation

```bibtex
@inproceedings{isbarov-etal-2024-open,
    title = "Open foundation models for Azerbaijani language",
    author = "Isbarov, Jafar and Huseynova, Kavsar and Mammadov, Elvin and Hajili, Mammad and Ataman, Duygu",
    booktitle = "SIGTURK 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.sigturk-1.2"
}
```

## License

MIT
