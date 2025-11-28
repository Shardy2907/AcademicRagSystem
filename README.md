# Agentic RAG System with Multi-Agent Orchestration  
RAG + Web Search + General Chat + LangGraph Supervisor

This project implements an **Agentic Retrieval-Augmented Generation (RAG) system** using a **three-agent architecture** orchestrated by **LangGraph**.  
It provides a clean **Streamlit-based chat interface** with intelligent routing between:

- A RAG agent (local PDF knowledge)
- A Web agent (internet search using Tavily)
- A General agent (conversation + reasoning)

The supervisor LLM decides dynamically which agent should handle each query.

---

# 1. Introduction

## What this system is  
This is an agent-based chatbot capable of:

- Answering from **local PDFs** using Retrieval-Augmented Generation  
- Searching the **internet** for external information  
- Handling casual chat and general reasoning  
- Automatically choosing the best agent for each question

It is designed for academic, research, and technical workflows requiring both local knowledge and external information when needed.

---

## What is RAG?  
**RAG (Retrieval-Augmented Generation)** retrieves relevant information from an external knowledge base—such as PDFs or documents—and feeds it into an LLM.  
This significantly improves accuracy and reduces hallucinations.

---

## What is Agentic AI?  
**Agentic AI** consists of multiple specialized agents working together.  
A **supervisor** intelligently routes user queries to the right agent, creating:

- Better accuracy  
- Cleaner answers  
- More dynamic behavior  

This project uses **three agents**.

---

# 2. Overview of the Three Agents

### 1. RAG Agent  
- Uses Qdrant vector DB  
- Uses BGE embeddings  
- Retrieves answers from your local PDFs  
- Best for academic and technical questions

### 2. Web Agent  
- Uses Tavily Search API  
- Fetches online results  
- Uses embeddings + LLM reasoning to pick the most relevant result  
- Summarizes reliably  
- Best for geography, public figures, world info, etc.

### 3. General Agent  
- Handles greetings, goodbyes, and casual chat  
- Handles “Explain this concept” type questions  
- Explains the system when explicitly asked  
- Does not hallucinate content from PDFs or the web

---

# 3. How to Use the System

1. Place your PDFs inside the `data/` folder.
2. Run the ingestion script to index the documents.
3. Launch the Streamlit UI.
4. Ask questions freely—the supervisor will route automatically:
   - RAG → For document-based answers  
   - Web → For real-world queries  
   - General → For chat or simple conceptual answers  

The system ensures the best agent is always chosen.

---

# 4. Installation

## Step 1 — Clone the Repository
```bash
git clone <your-repo-url>
cd <your-project>
```

---

## Step 2 — Install uv (if not installed)

```bash
pip install uv
```
or
```bash
pipx install uv
```

---

## Step 3 — Install dependencies
```bash
uv sync
```
This will create and manage a virtual environment using your pyproject.toml.

---

# 5. Install Ollama and Pull Model
## Step 1 — Install Ollama:
https://ollama.com/download

## Step 2 — Pull the model used by the system:
```bash
ollama pull phi3:mini
``` 
## Step 3 — Ensure Ollama is running:
```bash
ollama serve
``` 

---

# 6. Build the RAG Vector Index (Ingest Data)
Run the ingestion script to process your PDFs:
```bash
uv run src/ingest_data.py
``` 
This will:
- Load PDFs from /data
- Split them into chunks
- Generate embeddings
- Store vectors in qdrant_local/

# 7. Set Environment Variables
Create a .env file:
```bash
TAVILY_API_KEY=your_api_key_here
``` 
Get your key from:
- https://app.tavily.com/

# 8. Run the Streamlit UI
Launch the chat interface:
```bash
uv run streamlit run src/streamlit_app.py
``` 
A browser window will open (usually at http://localhost:8501).
