import os

#MODEL SETTINGS
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen2.5:1.5b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

#RETRIEVAL SETTINGs
SIMILARITY_TOP_K = 5
RESPONSE_MODE = "tree_summarize"

#PATHS
DATA_DIR = "./data"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "techcorp_docs"

CONFIDENCE_THRESHOLD = 0.5