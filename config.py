#MODEL SETTINGS
EMBEDDING_MODEL = "mxbai-embed-large:335m"
LLM_MODEL = "gemma3:1b"
OLLAMA_URL = "http://localhost:11434"

#RETRIEVAL SETTINGs
SIMILARITY_TOP_K = 5
RESPONSE_MODE = "tree_summarize"

#PATHS
DATA_DIR = "./data"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "techcorp_docs"

CONFIDENCE_THRESHOLD = 0.5