# Business Document RAG with ML

A Document Intelligence & Retrieval-Augmented Generation (RAG) system that answers user questions from internal business documents using local LLMs, semantic embeddings, ChromaDB, and a machine-learning query classifier.

Supports category-aware retrieval, semantic search, confidence-based filtering, a REST API, and a web UI.

---

## Features

- PDF document ingestion with category tagging (HR, Finance, Engineering, etc.)
- ML query classifier (TF-IDF + best model selected via cross-validation)
- Metadata-filtered semantic search using ChromaDB
- Local LLM inference via Ollama (no API keys, no cost)
- FastAPI REST backend with health and query endpoints
- Streamlit web frontend with chat history display
- Streaming and non-streaming response modes
- Query metrics logging (latency, confidence, sources) to JSONL
- Persistent vector store — no re-indexing on every run
- MLflow experiment tracking and model registry
- Fully containerized with Docker Compose

---

## Architecture

```
User Question
      ↓
Query Classifier (TF-IDF + ML)
      ↓
Metadata Filter (Category)
      ↓
ChromaDB Vector Store (Embeddings)
      ↓
Relevant Context Retrieval
      ↓
LLM via Ollama (gemma3:1b)
      ↓
Final Answer + Sources
```

---

## Project Structure

```
BUSINESS-RAG-WITH-ML/
│
├── data/
│   ├── processed/
│   │   ├── document_labels.json        # Category labels per PDF
│   │   └── question_training_data.csv  # Training data for classifier
│   ├── benefits_guide.pdf
│   ├── employee_handbook.pdf
│   ├── engineering_handbook.pdf
│   ├── finance_policies.pdf
│   ├── financial_controls.pdf
│   └── onboarding_guide.pdf
│
├── models/
│   ├── document_classifier.pkl         # Trained ML classifier
│   ├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
│   └── classifier_metadata.json        # Model metrics and config
│
├── chroma_db/                          # Persistent ChromaDB vector store
│
├── config.py                           # Central configuration (models, paths, settings)
├── label_documents.py                  # Assign category labels to PDFs
├── generate_question.py                # Synthetic training question generation
├── training_classifier.py             # Train and select best ML classifier
├── test_classifier.py                  # Evaluate classifier with sample questions
├── rag.py                              # Core RAG system (CLI mode)
├── api.py                              # FastAPI REST backend
├── front.py                            # Streamlit web frontend
├── docker-compose.yml                  # Full stack container setup
├── Dockerfile
├── requirements.txt
├── query_metrics.jsonl                 # Query performance logs (auto-created)
└── rag_system.log                      # System logs (auto-created)
```

---

## How It Works

### 1. Document Indexing
- PDFs are loaded from `data/`
- Each document is tagged with a category from `document_labels.json`
- Text is chunked using semantic splitting and embedded via `nomic-embed-text`
- Embeddings are stored persistently in ChromaDB — no re-indexing on restart

### 2. Query Classification
- User question is vectorized using TF-IDF
- ML classifier (LR / Random Forest / XGBoost — best selected by CV precision) predicts a category
- If confidence > 0.5, search is filtered to that category only; otherwise all documents are searched

### 3. Retrieval & Answering
- Top-K semantically similar chunks are retrieved from ChromaDB
- Context is passed to the LLM (qwen2.5:1.5b via Ollama)
- Answer is returned along with source metadata (filename, page, score, category)

---

## Classifier Categories

| Category | Example Documents |
|---|---|
| `HR` | Employee handbook, benefits guide |
| `ENGINEERING` | Engineering handbook |
| `FINANCE` | Finance policies, financial controls |
| `GENERAL` | Onboarding guide |

---

## Quick Start

### Option A — Docker (Recommended)

```bash
git clone https://github.com/AkhilKandoi/Business-Document-RAG.git
cd Business-Document-RAG

docker compose up --build
```

Services after startup:

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Ollama | http://localhost:11434 |

> The `ollama-init` container automatically pulls `gemma3:1b` and `mxbai-embed-large:335m` on first run. This may take a few minutes depending on your connection.

---

### Option B — Local (Manual)

#### 1. Install Ollama
Download from [ollama.com](https://ollama.com) and run:
```bash
ollama serve
ollama pull gemma3:1b
ollama pull mxbai-embed-large:335m
```

#### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

#### 3. Label your documents
Edit `label_documents.py` to map your PDF filenames to categories, then run:
```bash
python label_documents.py
```

#### 4. train the classifier
```bash
python training_classifier.py
```

View MLflow experiments:
```bash
mlflow ui
# Open http://localhost:5000
```

#### 5. (Optional) Evaluate the classifier
```bash
python test_classifier.py
```

#### 6. Run the system

**CLI mode:**
```bash
python rag.py
```

**API + Frontend:**
```bash
# Terminal 1
python api.py

# Terminal 2
streamlit run front.py
```

---

## Embedding Model Warning

The embedding model used to build `chroma_db/` must match `EMBEDDING_MODEL` in `config.py`. If you change the model, delete the existing vector store and re-index:

```bash
rm -rf ./chroma_db
python rag.py
```

Mismatched embedding dimensions will cause a runtime error on the first query.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root / version info |
| `GET` | `/health` | Health check (API + Ollama status) |
| `GET` | `/info` | Document count, categories, classifier status |
| `POST` | `/query` | Ask a question |
| `GET` | `/docs` | Swagger interactive docs |

### Non-streaming query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the parental leave policy?", "stream": false}'
```

Response:
```json
{
  "answer": "Employees are entitled to 16 weeks of paid parental leave...",
  "sources": [
    {
      "filename": "employee_handbook.pdf",
      "page": "12",
      "score": 0.871,
      "text": "...",
      "category": "HR"
    }
  ],
  "category": "HR",
  "confidence": 0.94,
  "latency": 2.13
}
```

### Streaming query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "What is the code review process?", "stream": true}'
```

Tokens arrive as SSE events:
```
data: {"type": "chunk", "text": "The code review"}
data: {"type": "chunk", "text": " process requires..."}
data: {"type": "done", "sources": [...], "category": "ENGINEERING", "confidence": 0.91, "latency": 1.84}
```

---

## Configuration

All settings are in `config.py`:

```python
EMBEDDING_MODEL  = "mxbai-embed-large:335m"
LLM_MODEL        = "gemma3:1b"
OLLAMA_URL       = "http://localhost:11434"   # overridden by env var in Docker

SIMILARITY_TOP_K     = 5
RESPONSE_MODE        = "tree_summarize"
CONFIDENCE_THRESHOLD = 0.5

DATA_DIR        = "./data"
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "techcorp_docs"
```

To swap models, change `EMBEDDING_MODEL` or `LLM_MODEL` to any model available via `ollama list`. Remember to delete `chroma_db/` if you change `EMBEDDING_MODEL`.

---

## Metrics & Logging

Every query is automatically logged to `query_metrics.jsonl`:

```json
{
  "timestamp": "2026-02-24T19:20:09",
  "question": "What is the code review process?",
  "category": "ENGINEERING",
  "confidence": 0.664,
  "answer_length": 771,
  "num_sources": 5,
  "latency_seconds": 2.13
}
```

General system events and errors are logged to `rag_system.log`.

MLflow experiment results are stored in `mlflow.db` and viewable via:
```bash
mlflow ui   # http://localhost:5000
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM & Embeddings | Ollama (gemma3:1b, mxbai-embed-large) |
| RAG Framework | LlamaIndex |
| Vector Store | ChromaDB |
| ML Classifier | scikit-learn, XGBoost (TF-IDF features) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |