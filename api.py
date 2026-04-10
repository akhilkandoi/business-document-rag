from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import uvicorn
from rag import QASystem, check_ollama
import asyncio

qa_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_system
    for attempt in range(5):
        if check_ollama():
            qa_system = QASystem()
            break
        print(f"Waiting for Ollama... attempt {attempt+1}/5")
        await asyncio.sleep(10)
    yield
    print("shutting down")


app = FastAPI(
    title="Document Q&A API",
    description="RAG-based Question Answering API",
    version="1.0.0",
    lifespan = lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



class QueryRequest(BaseModel):
    question: str
    stream: bool = False

class Source(BaseModel):
    filename: str
    page: str
    score: float
    text: str
    category: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    category: str
    confidence: float
    latency: float

class HealthResponse(BaseModel):
    status: str
    system_ready: bool
    ollama_running: bool

#health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    #check if the api and services are running
    ollama_status = check_ollama()
    system_ready = qa_system is not None

    return {
        "status":"healthy" if system_ready else "Initializing",
        "system_ready": system_ready,
        "ollama_running":ollama_status
    }

#query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    #query the document collection
    if qa_system is None:
        raise HTTPException(
            status_code=503,
            detail="QA system not initialized. Check if ollama is running"
        )
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail="Question too long")
    try:
        #query the system
        result = qa_system.query(request.question, stream=request.stream)

        #format sources
        sources=[]
        for node in result['sources']:
            sources.append({
                "filename":node.metadata.get('file_name', 'Unkown'),
                "page": node.metadata.get('page_label', 'N/A'),
                "score": round(node.score, 3),
                "text": node.text[:200] + "...",
                "category": node.metadata.get('category', 'N/A')
            })
        
        return{
            "answer": result['answer'],
            "sources": sources,
            "category": result['category'],
            "confidence": result['confidence'],
            "latency": result['latency']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def get_system_info():
    #get info about system
    if qa_system is None:
        return {"error": "System not initialized"}
    
    try:
        doc_count = qa_system.chroma_collection.count() if hasattr(qa_system, 'chroma_collection') else 0
        
        return {
            "document_count": doc_count,
            "categories": list(qa_system.document_categories.keys()) if qa_system.document_categories else [],
            "classifier_loaded": qa_system.classifier.is_loaded
        }
    except Exception as e:
        return {"error": str(e)}
    
#root endpoint
@app.get("/")
async def root():
    #root endpoint
    return{
        "message": "Document Q&A API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
