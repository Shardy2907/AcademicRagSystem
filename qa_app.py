import time
import os
import sys
from loguru import logger
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Configuration Constants ---
QDRANT_PATH = "qdrant_local"
COLLECTION_NAME = "university_docs"
# Using standard BGE model via the new HuggingFaceEmbeddings class
HF_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_LLM_MODEL = "phi3:mini"

# Configure Logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")

def initialize_qa_chain():
    """Initializes the RAG chain using modern libraries."""

    # 1. Initialize Embeddings (Updated Class)
    logger.info(f"Loading Embeddings: {HF_EMBEDDING_MODEL_NAME}...")
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}

    try:
        # Switched to HuggingFaceEmbeddings to fix deprecation warning
        embeddings = HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        logger.exception("Failed to load embeddings.")
        raise

    # 2. Load Qdrant
    logger.info(f"Connecting to Qdrant DB at {QDRANT_PATH}...")
    if not os.path.exists(QDRANT_PATH):
        logger.error(f"Directory '{QDRANT_PATH}' not found. Please run ingest.py first.")
        raise FileNotFoundError(f"Directory '{QDRANT_PATH}' not found.")

    try:
        client = QdrantClient(path=QDRANT_PATH, prefer_grpc=False)
        
        # The new QdrantVectorStore uses 'embedding' (singular)
        db = QdrantVectorStore(
            client=client, 
            collection_name=COLLECTION_NAME, 
            embedding=embeddings, # <--- This must be SINGULAR
        )
        logger.success("Qdrant Client Connected.")
    except Exception as e:
        logger.exception("Failed to connect to Qdrant.")
        raise

    # 3. Create Retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 4. Initialize Ollama
    logger.info(f"Initializing LLM: {OLLAMA_LLM_MODEL}...")
    try:
        llm = Ollama(
            model=OLLAMA_LLM_MODEL,
            base_url="http://localhost:11434",
            temperature=0.3,
            num_predict=512,
            timeout=120
        )
    except Exception as e:
        logger.exception("Failed to connect to Ollama.")
        raise

    # 5. Create Chain
    logger.info("Building RetrievalQA Chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=False 
    )
    
    return qa_chain

def main():
    logger.info("Starting RAG System Initialization...")
    
    try:
        qa_chain = initialize_qa_chain()
    except Exception:
        logger.critical("System initialization failed. Exiting.")
        return

    logger.success("System Ready.")
    print("\n" + "="*50)
    print("RAG Chatbot Ready (Type 'exit' to quit)")
    print("="*50)

    while True:
        try:
            query = input("\nQuery: ")
            if query.lower() in ["quit", "exit"]: 
                logger.info("User requested exit.")
                break
            if not query.strip(): continue
            
            logger.info(f"Processing query: '{query}'")
            start = time.time()
            
            # Run the query
            response = qa_chain.invoke({"query": query})
            
            elapsed = time.time() - start
            logger.info(f"Response generated in {elapsed:.2f}s")
            
            # Print Answer
            print("\n" + "-"*80)
            print(f"ANSWER:")
            print(response["result"].strip())
            print("-" * 80)
            
            # Print Sources
            source_docs = response.get("source_documents", [])
            if source_docs:
                print("SOURCES:")
                for i, doc in enumerate(source_docs, 1):
                    src = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page = doc.metadata.get('page', 'N/A')
                    print(f"   {i}. {src} (Page {page})")
            else:
                logger.warning("No relevant source documents found.")
                
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()