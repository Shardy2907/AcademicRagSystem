import os
import sys
from loguru import logger
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Configuration Constants ---
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
QDRANT_PATH = os.path.join(PROJECT_ROOT, "qdrant_local")
COLLECTION_NAME = "university_docs"
HF_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384


# Configure Logger to write to file and stderr
logger.remove() # Remove default handler to avoid double logging if re-run
logger.add(sys.stderr, level="INFO")

def ingest_data():
    logger.info("DATA INGESTION STARTED")

    # 1. Load Documents
    logger.info(f"Loading documents from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data directory '{DATA_PATH}' not found.")
        return

    try:
        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True,
            recursive=True
        )
        documents = loader.load()
    except Exception as e:
        logger.exception(f"Failed to load documents: {e}")
        return

    if not documents:
        logger.warning("No documents found.")
        return
    logger.success(f"Loaded {len(documents)} documents.")

    # 2. Split Text
    logger.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    logger.success(f"Split into {len(chunks)} chunks.")

    # 3. Initialize Embeddings
    logger.info(f"Initializing Embeddings ({HF_EMBEDDING_MODEL_NAME})...")
    try:
        model_kwargs = {"device": "cpu"} 
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.success("Embeddings initialized.")
    except Exception as e:
        logger.exception(f"Failed to initialize embeddings: {e}")
        return

    # 4. Initialize Qdrant Client & Create Collection Manually
    logger.info(f"Connecting to Qdrant at '{QDRANT_PATH}'...")
    
    try:
        client = QdrantClient(path=QDRANT_PATH)

        # Check if collection exists and delete it (Reset)
        if client.collection_exists(COLLECTION_NAME):
            logger.warning(f"Deleting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)

        # Manually create collection
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=EMBEDDING_DIMENSION, 
                distance=qdrant_models.Distance.COSINE
            ),
        )
        logger.success(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        logger.exception(f"Failed to create Qdrant collection: {e}")
        return

    # 5. Add Documents
    logger.info("Indexing documents into Vector Store...")
    
    try:
        vector_store = Qdrant(
            client=client, 
            collection_name=COLLECTION_NAME, 
            embeddings=embeddings
        )
        vector_store.add_documents(chunks)
        logger.success(f"Successfully added {len(chunks)} vectors to Qdrant.")
    except Exception as e:
        logger.exception(f"Failed to index documents: {e}")
        return
    
    logger.info("="*40)
    logger.success("INGESTION COMPLETE")
    logger.info("="*40)

if __name__ == "__main__":
    ingest_data()