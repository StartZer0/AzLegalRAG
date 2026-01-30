# Configuration constants for AzLegalRAG

# Models
EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
EMBEDDING_DIM = 3584  # gte-Qwen2-7B dimension
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector store
VECTORSTORE_DIR = "./vectorstore"

# Retrieval
TOP_K = 5

# Dataset
DATASET_NAME = "allmalab/eqanun"
