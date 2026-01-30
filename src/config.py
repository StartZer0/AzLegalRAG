# Configuration constants for AzLegalRAG

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # bge-m3 dimension
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
