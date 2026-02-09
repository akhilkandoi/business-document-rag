#   BUSINESS DOCUMENT RAG WITH ML

A Document Intelligence & Retrieval-Augmented Generation (RAG) system that answers user questions from internal business documents using LLMs, embeddings, ChromaDB, and a machine-learning query classifier.

This project supports category-aware retrieval, semantic search, and confidence-based filtering to improve answer relevance.


## Features
- PDF document ingestion (HR, Finance, Engineering, etc.)
- ML-based query classification (TF-IDF + classifier)
- Metadata-filtered semantic search using ChromaDB
- Local LLM inference via Ollama
- Streaming and non-streaming responses
- Query metrics logging (latency, confidence, sources)
- Persistent vector store (no re-indexing every run)


## Architecture Overview
```
User Question
      ↓
Query Classifier (ML)
      ↓
Metadata Filter (Category)
      ↓
Chroma Vector Store (Embeddings)
      ↓
Relevant Context Retrieval
      ↓
LLM (Ollama)
      ↓
Final Answer + Sources
```


## Project Structure

```
BUSINESS-RAG-WITH-ML/
│
├── chroma_db/                # Persistent Chroma vector database
├── data/
│   ├── processed/            # Preprocessed training data (datasets are generated artificially.)
│   │   ├── document_labels.json
│   │   ├── question_training_data.csv
│   │   └── training_data.csv
│   │
│   ├── benefits_guide.pdf
│   ├── employee_handbook.pdf
│   ├── engineering_handbook.pdf
│   ├── finance_policies.pdf
│   ├── financial_controls.pdf
│   └── onboarding_guide.pdf
│
├── models/                   # Trained ML models & vectorizers
│   ├── classifier_metadata.json
│   ├── document_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── config.py                 # Central configuration 
├── generate_question.py      # Synthetic question generation
├── label_documents.py        # Assigns labels to documents
├── prep_training_data.py     # Training data preprocessing
├── training_classifier.py    # Train query classifier
├── test_classifier.py        # Evaluate classifier
├── rag.py                    # Main RAG application
├── query_metrics.jsonl       # Query logs & performance metrics
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## How It Works
1. Document Indexing

    - PDFs are loaded from data/

    - Each document is tagged with a category using document_labels.json

    - Text is chunked and embedded

    - Embeddings are stored in ChromaDB

2. Query Classification

    - User question is vectorized (TF-IDF)

    - ML classifier predicts a document category

    - Confidence score determines filtered vs global search

3. Retrieval & Answering

    - Relevant chunks are retrieved using semantic similarity

    - Context is passed to the LLM

    - Answer is generated without exposing document references 

## How to run the project on your PC

### 1. Clone the repo:
```
git clone https://github.com/AkhilKandoi/Business-Document-RAG.git
```

### 2. Download Ollama:
```
[Ollama](https://ollama.com) (~1.3GB)
```

### 3. Pull required Models:
```

You may use any of your liking, just change the model names in config.py

ollama pull gemma3:1b          (LLM Model)       (~815MB)
ollama pull nomic-embed-text   (Embedding Model) (~274MB)
```

### 4. Start ollama:
```
Open command prompt and type: ollama serve
```

### 5. Install Dependencies:
```
Download Python if not downloaded.

pip install -r requirements.txt
```
### 6. Run the Application
```
python rag.py
```

## Future Improvements
- Web-based UI (Streamlit/React)
- Improved classifier
- Cloud deployment

## Tech Stack
- Python
- LlamaIndex
- ChromaDB
- Ollama (LLM + Embeddings)
- scikit-learn
- TF-IDF
