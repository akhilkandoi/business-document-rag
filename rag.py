import os
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Tuple, Optional

import joblib
import requests
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    PromptTemplate
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from config import (
    OLLAMA_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    SIMILARITY_TOP_K,
    RESPONSE_MODE,
)


#---------LOGGING & CONFIGURATION------------------
logging.basicConfig(
    filename='rag_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


#-----------UTILITY FUNCTIONS--------------------

def check_ollama()->bool:
    try:
        requests.get(OLLAMA_URL, timeout=5)
        print("Ollama is running")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Ollama is not running. Start with: 'Ollama serve' in command prompt. Error: {e}")
        return False


def load_document_categories()->dict:
    """Load document category mapping from json"""
    filepath="./data/processed/document_labels.json"
    try:
        with open(filepath) as f:
            categories = json.load(f)
        logger.info(f"Loaded categories for {len(categories)} documents")
        return categories
    except FileNotFoundError:
        logger.warning(f"No document labels found at {filepath}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return {}




def log_query_metrics(question:str, response, latency:float, category:str='N/A', confidence:float = 0.0)->None:
    """Log query metrics"""
    metrics={
        "timestamp":datetime.now().isoformat(),
        "question":question,
        "category" : category,
        "confidence":round(confidence, 3),
        "answer_length":len(str(response)),
        "num_sources": len(response.source_nodes),
        "latency_seconds": round(latency, 2)
    }

    with open("query_metrics.jsonl", "a") as f:
        f.write(json.dumps(metrics, indent=2) + "\n")


#---------CLASSIFIER---------------

class QueryClassifier:
    """ML-based query classifier for category prediction"""
    def __init__(self):
        self.classifier = None
        self.vectorizer = None
        self.is_loaded = False
        self.is_xgboost = False
        self.categories = []   # ordered label list for XGBoost int->str decoding
        self._load_models()
    
    def _load_models(self)->None:
        """Load trained classifier and vectorizer"""
        try:
            self.classifier = joblib.load("./models/document_classifier.pkl")
            self.vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")


            try:
                with open("./models/classifier_metadata.json") as f:
                    meta = json.load(f)
                model_type = meta.get("model_type", "")
                self.is_xgboost = model_type == "XGBoost"
                self.categories = meta.get("categories", [])
            except (FileNotFoundError, json.JSONDecodeError):
                self.is_xgboost = False
                self.categories = []

            self.is_loaded = True
            print(f"Classifier loaded ({meta.get('model_type', 'unknown')})")
        except FileNotFoundError:
            print("No Classifier found")
            self.is_loaded = False

    def classify(self, question:str)->Tuple[str, float]:
        """Classify query into category"""
        if not self.is_loaded:
            return "ALL", 0.0
        question_vec = self.vectorizer.transform([question])
        raw = self.classifier.predict(question_vec)[0]

        # XGBoost was trained on LabelEncoder-encoded ints; convert back to string.
        if self.is_xgboost and self.categories:
            category = self.categories[int(raw)]
        else:
            category = str(raw)

        proba = self.classifier.predict_proba(question_vec)[0]
        confidence = float(max(proba))

        return category, confidence
            

#------RAG SYSTEM--------------


class QASystem:
    """Main RAG System for document question answering"""
    def __init__(self):
        self.index = None
        self.classifier = QueryClassifier()
        self.document_categories = load_document_categories()

        #configure LLM and Embeddings
        self._configure_models()

        #Initialize vector store
        self._initialize_vector_store()

        #load or create index
        self._load_or_create_index()

    def _configure_models(self)->None:
        """Configure LLM and embedding models"""
        Settings.embed_model = OllamaEmbedding(
            model_name=EMBEDDING_MODEL ,
            base_url=OLLAMA_URL,
            request_timeout=120.0
        )
        Settings.llm = Ollama(
            model=LLM_MODEL,
            request_timeout=120.0
        )

        Settings.text_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model,
        )

    def _initialize_vector_store(self) -> None:
        """Initialize ChromaDB vector store"""
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def _load_or_create_index(self)->None:
        """Load existing index or creates a new one"""
        if self.chroma_collection.count() > 0:
            print("Loading index...")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store, 
                embed_model = Settings.embed_model,
            )
        else:
            print("Creating new index...")
            self._create_index()

    def _create_index(self)->None:
        """Create new index from documents"""
        documents = SimpleDirectoryReader(input_dir="./data", required_exts=['.pdf'], recursive=False).load_data()
        print(f"Loaded {len(documents)} documents")
        print("Tagging documents...")
        for doc in documents:
            filename = doc.metadata.get('file_name','')
            category = self.document_categories.get(filename, 'GENERAL')
            doc.metadata['category'] = category

        self.index = VectorStoreIndex.from_documents(
            documents, 
            transformations=[Settings.text_splitter],
            embed_model = Settings.embed_model,
            storage_context=self.storage_context
        )
        print("Index created")

    def _create_metadata_filter(self, category:str)->Optional[MetadataFilters]:
        """Create metadata filter for category"""
        if category == "ALL":
            return None
        return MetadataFilters(
            filters=[ExactMatchFilter(key="category", value=category)]
        )
    
    def _build_query_engine(self, filters: Optional[MetadataFilters]):
        """Build query engine with optional filters"""
        query_engine = self.index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K,
            response_mode=RESPONSE_MODE,
            streaming=True,
            filters=filters
        )
        qa_prompt = PromptTemplate(
            "Use following information to answer the question. Do not mention that you got the information from documents. Give full sentence answers.\n\n"
            "Context: \n{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer:"
        )

        query_engine.update_prompts({"response_synthesizer:text_qa_template":qa_prompt})

        return query_engine
    
    def query(self, question:str, stream:bool = False) -> dict:
        """Query the system with question"""

        if not question or not question.strip():
            raise ValueError("Question can't be empty")
        if len(question)>500:
            raise ValueError("Question too long (max 500 charas)")

        category, confidence = self.classifier.classify(question)

        if confidence > 0.5 and category != "ALL":
            filters = self._create_metadata_filter(category) 
            search_scope = f"{category} documents only"
        else:
            filters=None
            category="ALL"
            search_scope="all documents"
        
        print(f"Category: {category} ({confidence:.1%})")
        print(f"Search {search_scope}...")

        start_time = time.time()

        try:
            query_engine = self._build_query_engine(filters)
            if stream: 
                response = query_engine.query(question)
                latency = time.time() - start_time

                #log metrics
                log_query_metrics(question, response, latency, category, confidence)
                logger.info(f"Query: {question} | Category: {category} ({confidence:.1%})")

                return {
                    'answer':response.response_gen,
                    'sources':response.source_nodes,
                    'category':category,
                    'confidence':confidence,
                    'latency':latency
                }
            else:
                response = query_engine.query(question)
                latency = time.time() - start_time

                #log metrics
                log_query_metrics(question, response, latency, category, confidence)
                logger.info(f"Query: {question} | Category: {category} ({confidence:.1%})")

                return {
                    'answer':str(response),
                    'sources':response.source_nodes,
                    'category':category,
                    'confidence':confidence,
                    'latency':latency
                }
        except Exception as e:
            logger.error(f"Query failed:{str(e)}")
            logger.error(traceback.format_exc())
            raise

        



#-------DISPLAY-----------

def print_header()->None:
    """Print system header"""
    print("\n"+"="*60)
    print("Document intelligence system")
    print("\n"+"="*60)
    print("Type 'quit' to exit.\n")

def print_sources(sources)->None:
    print("\n---Sources---")

    for i, node in enumerate(sources, 1):
        cat = node.metadata.get('category','N/A')
        score = node.score
        filename = node.metadata.get('file_name', 'Unknown')
        page = node.metadata.get('page_label','N/A')

        print(f" [{i}]  {cat:12s} | Score: {score:.3f} | {filename:30s} (p.{page})")

def print_answer(result:dict)->None:
    print(f"Answer: \n {result['answer']}\n")
    print_sources(result['sources'])
    print(f"latency: {result['latency']:.2f}s")



#------------MAIN APP----------

def main():
    """MAIN APP LOOP"""
    if not check_ollama():
        return
    
    print("Initializing system...")
    qa_system = QASystem()

    print_header()

    while True:
        try: 
            question = input("Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q', 'e']:
                print("Goodbye!")
                break

            if not question.strip():
                continue
            
            result = qa_system.query(question)
            print_answer(result)
        except KeyboardInterrupt:
            print("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())


def update_index(index, new_doc_path):
    """Add new document to existing index"""
    from llama_index.core import Document
    try:
        new_docs = SimpleDirectoryReader(new_doc_path).load_data()
        for doc in new_docs:
            index.insert(doc)
        print(f"Added {len(new_docs)} new documents.")
    except: 
        print("Invalid document path.")

    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()