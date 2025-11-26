import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# --- Configuration Constants ---
DATA_PATH = "data/"
QDRANT_PATH = "qdrant_local"  # Directory for local Qdrant (embedded mode)
COLLECTION_NAME = "university_docs"
HF_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def ingest_data():
    print(f"Starting document loading from {DATA_PATH}...")

    # 1. Load PDF documents
    try:
        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True,
            recursive=True
        )
        documents = loader.load()

        if not documents:
            print(f"⚠️ Warning: No documents found in {DATA_PATH}.")
            return

        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error while loading documents: {e}")
        return

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Embeddings
    model_kwargs = {"device": "cpu"}  # or "cuda"
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=HF_EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 4. Create Qdrant client (local file-based)
    print("Initializing local Qdrant instance...")
    client = QdrantClient(
    path=QDRANT_PATH, 
    prefer_grpc=False # critical line
    )


    # 5. Store documents in Qdrant
    print("Storing vectors into Qdrant...")

    Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        qdrant_client=client,
        collection_name=COLLECTION_NAME,
    )

    print("✅ Data ingestion complete. Qdrant vector store created!")


if __name__ == "__main__":
    ingest_data()
