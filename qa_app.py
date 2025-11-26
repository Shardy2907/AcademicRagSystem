from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import time
import os
import sys

# --- Configuration Constants ---
CHROMA_DB_PATH = "chroma_db"
HF_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_LLM_MODEL = "phi3:mini"

def initialize_qa_chain():
    """Initializes the hybrid RAG chain."""

    # 1. Initialize the Hugging Face Embeddings
    print("Initializing HuggingFace embedding function...")
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=HF_EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("✓ Embeddings initialized")

    # 2. Load the Chroma Vector Store
    print(f"Loading Chroma DB from {CHROMA_DB_PATH}...")
    
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"ERROR: ChromaDB directory '{CHROMA_DB_PATH}' does not exist!")
        raise FileNotFoundError(f"ChromaDB directory not found: {CHROMA_DB_PATH}")
    
    print(f"✓ ChromaDB directory exists")
    
    try:
        print("Loading ChromaDB...")
        start = time.time()
        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        elapsed = time.time() - start
        print(f"✓ ChromaDB loaded in {elapsed:.2f}s")
    except Exception as e:
        print(f"ERROR loading ChromaDB: {e}")
        raise
    
    print("✓ Creating retriever...")
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("✓ Retriever created")

    # 3. Initialize the Ollama LLM
    print(f"Loading Ollama LLM: {OLLAMA_LLM_MODEL}...")
    
    llm = Ollama(
        model=OLLAMA_LLM_MODEL,
        base_url="http://localhost:11434",
        temperature=0.7,
        num_predict=256,
        timeout=120  # 2 minute timeout
    )
    
    # Test the LLM connection
    try:
        print("Testing LLM connection...")
        start = time.time()
        test_response = llm.invoke("Hi")
        elapsed = time.time() - start
        print(f"✓ LLM test successful in {elapsed:.2f}s")
        print(f"  Response preview: {test_response[:80]}...")
    except Exception as e:
        print(f"ERROR: Ollama LLM test failed: {e}")
        raise

    # 4. Create the Retrieval-Augmented QA Chain
    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    print("✓ QA Chain created")
    
    return qa_chain

def main():
    """Main function to run the interactive query loop."""
    print("="*80)
    print("RAG SYSTEM INITIALIZATION")
    print("="*80)
    
    try:
        qa_chain = initialize_qa_chain()
    except Exception as e:
        print(f"\n❌ Failed to initialize the QA chain: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)
    print("✓ RAG SYSTEM READY")
    print("="*80)
    print("Enter 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("\nQuery: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
            
        if query.lower() in ["quit", "exit"]:
            print("Exiting...")
            break
        
        if not query.strip():
            print("Please enter a valid query.")
            continue
            
        print("\n" + "-"*80)
        print(f"Processing: '{query}'")
        print("-"*80)
        
        try:
            print("[Step 1/4] Embedding query...")
            sys.stdout.flush()
            
            print("[Step 2/4] Searching vector database...")
            sys.stdout.flush()
            
            print("[Step 3/4] Retrieving context...")
            sys.stdout.flush()
            
            print("[Step 4/4] Generating answer with LLM (this may take 30-120 seconds)...")
            print("            If this hangs, press Ctrl+C to cancel")
            sys.stdout.flush()
            
            start_time = time.time()
            
            # The actual invocation with detailed error handling
            try:
                result = qa_chain.invoke({"query": query})
            except KeyboardInterrupt:
                print("\n⚠ Query cancelled by user")
                continue
            except Exception as invoke_error:
                print(f"\n❌ Error during qa_chain.invoke(): {invoke_error}")
                print(f"   Error type: {type(invoke_error).__name__}")
                import traceback
                traceback.print_exc()
                continue
            
            elapsed = time.time() - start_time
            print(f"\n✓ Completed in {elapsed:.2f}s")
            
            # Validate result
            if result is None:
                print("❌ ERROR: Result is None")
                continue
                
            if not isinstance(result, dict):
                print(f"❌ ERROR: Result is not a dict, it's a {type(result)}")
                print(f"   Result value: {result}")
                continue
            
            if "result" not in result:
                print(f"❌ ERROR: 'result' key not in response")
                print(f"   Available keys: {list(result.keys())}")
                continue
            
            # Display the results
            print("\n" + "="*80)
            print("ANSWER:")
            print("="*80)
            answer = result["result"]
            if answer:
                print(answer.strip())
            else:
                print("(Empty response)")
            print("="*80)
            
            source_docs = result.get("source_documents", [])
            if source_docs:
                print(f"\nSOURCE DOCUMENTS ({len(source_docs)} found):")
                print("-"*80)
                for i, doc in enumerate(source_docs, 1):
                    print(f"{i}. Source: {doc.metadata.get('source', 'N/A')} (Page: {doc.metadata.get('page', 'N/A')})")
                    content_preview = doc.page_content[:150].strip().replace('\n', ' ')
                    print(f"   Preview: {content_preview}...")
                    print()
                print("="*80)
            else:
                print("\n⚠ No source documents returned")
            
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted! Continuing...")
            continue
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            print("\nContinuing to next query...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR in main(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)