import streamlit as st
import os
import sys
import time
from loguru import logger
from dotenv import load_dotenv

from src.qa_app import initialize_qa_chain
from src.web_agent import web_agent_node
from src.general_agent import general_llm_agent_node
from main import build_agent_graph, AgentState 

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Polish ---
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        margin-top: 0;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff4b4b;
        margin-top: 5px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization (Cached) ---
# We use cache_resource so we don't reload the embeddings/Qdrant on every interaction
@st.cache_resource(show_spinner="Initializing Agent System...")
def setup_agent_system():
    load_dotenv()
    logger.info("Initializing QA Chain and Graph...")
    qa_chain, qdrant_client = initialize_qa_chain()
    agent_graph = build_agent_graph(qa_chain)
    return agent_graph, qa_chain

try:
    graph, qa_chain = setup_agent_system()
except Exception as e:
    st.error(f"Failed to initialize system: {e}")
    st.stop()

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Control Center")
    st.markdown("---")
    st.markdown("**System Status:** Online")
    
    if st.button("Clear Conversation", icon="🗑️"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Agent Capabilities")
    st.caption("1. **RAG Agent:** Technical Docs & PDFs")
    st.caption("2. **Web Agent:** Real-world info")
    st.caption("3. **General Agent:** Small talk & Logic")

# --- Main Chat Interface ---
st.title("Agentic RAG System")
st.caption("Powered by LangGraph, Ollama, and Qdrant")

# 1. Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there are sources associated with a historical assistant message, show them
        if "sources" in msg:
            with st.expander("Referenced Documents"):
                for source in msg["sources"]:
                    st.markdown(f"- **{source['file']}** (Page {source['page']})")

# 2. Chat Input
if prompt := st.chat_input("Ask a question about robotics, math, or the world..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with st.spinner("Agents are coordinating..."):
            try:
                # Prepare inputs for the graph
                inputs = {
                    "messages": [{"role": "user", "content": prompt}],
                    "qa_chain": qa_chain
                }
                
                # Invoke the graph
                output = graph.invoke(inputs)
                
                # Determine which agent answered and extract data
                final_text = ""
                sources_list = []
                agent_used = "Unknown"

                if "answer" in output and output["answer"]:
                    final_text = output["answer"]
                    agent_used = "RAG Agent"
                    # Process sources
                    if output.get("sources"):
                        for d in output["sources"]:
                            src_name = os.path.basename(d.metadata.get("source", "Unknown"))
                            page = d.metadata.get("page", "N/A")
                            sources_list.append({"file": src_name, "page": page})
                            
                elif "web_result" in output and output["web_result"]:
                    final_text = output["web_result"]
                    agent_used = "Web Agent"
                    
                elif "general_answer" in output and output["general_answer"]:
                    final_text = output["general_answer"]
                    agent_used = "General Agent"
                else:
                    final_text = "I processed the request but received no valid output from the sub-agents."

                # Display Result
                status_placeholder.caption(f"Handled by: **{agent_used}**")
                message_placeholder.markdown(final_text)

                # Display Sources if RAG
                if sources_list:
                    with st.expander("Referenced Documents"):
                        for s in sources_list:
                            st.markdown(f"- 📄 `{s['file']}` (Page {s['page']})")

                # Save to History
                assistant_msg = {
                    "role": "assistant", 
                    "content": final_text
                }
                if sources_list:
                    assistant_msg["sources"] = sources_list
                
                st.session_state.messages.append(assistant_msg)

            except Exception as e:
                logger.error(f"Streamlit Error: {e}")
                st.error("An error occurred while processing your request.")