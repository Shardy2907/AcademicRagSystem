import os
import sys
from typing import TypedDict, Optional, List
from loguru import logger
from langgraph.graph import StateGraph, END
from src.qa_app import initialize_qa_chain, rag_answer
from src.web_agent import web_agent_node
from src.general_agent import general_llm_agent_node
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()


logger.remove()
logger.add(sys.stderr, level="INFO")

logger.info("Starting main.py...")


class AgentState(TypedDict):
    messages: list
    answer: Optional[str]
    sources: Optional[list]
    web_result: Optional[str]
    general_answer: Optional[str]      
    qa_chain: object



###########################################################################
######## RAG Agent Node  ###########################################
###########################################################################
def rag_agent_node(state: AgentState):
    try:
        query = state["messages"][-1]["content"]
        logger.info(f"[RAG AGENT] Query...")
        qa_chain = state["qa_chain"]

        response = rag_answer(qa_chain, query)

        new_state = dict(state)
        new_state["answer"] = response["result"]
        new_state["sources"] = response.get("source_documents", [])
        new_state["messages"] = state["messages"] + [
            {"role": "assistant", "content": response["result"]}
        ]
        return new_state
    
    except Exception as e:
        logger.exception(f"[RAG AGENT] Error: {e}")
        fallback = "Sorry, I couldn't retrieve the document-based answer."
        return _safe_state_update(state, "answer", fallback)


##############################################################################################
###### Supervisor Node ##############
##############################################################################################


def supervisor_node(state):
    try:
        query = state["messages"][-1]["content"].lower()
        logger.info(f"[SUPERVISOR] Received...")

        qa_chain = state["qa_chain"]
        retriever = qa_chain.retriever
        docs = retriever.get_relevant_documents(query)

        if docs and len(docs) > 0 and docs[0].metadata.get("score", 0) > 0.3:
            logger.info("[SUPERVISOR] → rag_agent (high similarity)")
            return _route(state, "rag_agent")

        # 3. Default → LLM-based decision (rag vs web)
        llm = Ollama(model="phi3:mini", base_url="http://localhost:11434", temperature=0.2)

        system_prompt = """
            You are the Routing Supervisor of a multi-agent RAG system. 
            Your job is to choose exactly ONE agent for the user's query.

            THE AGENTS:

            1. rag
            Use this agent when the user's query is:
            - academic 
            - technical
            - related to any robotics topics
            - connected to the content of the local PDFs
            If the user appears to ask anything that might be inside the documents, 
            prefer "rag" unless clearly not relevant.

            2. web
            Use this ONLY when the question requires real-world, external, 
            live, or factual information that is NOT available in the PDFs.
            Typical cases:
            - geography
            - world events
            - company info
            - current affairs
            - rare facts that clearly require web search
            Always use "web" when the user asks anything about any public figure or any particular human.
            Typical cases:
            - Who is Lionel Messi?
            - Who is John Doe?
            Never choose "web" if the question could possibly be answered using the PDFs.

            3. general
            Use this when the user's intent is:
            - greeting, farewell, or general conversation
            - asking about the chatbot itself, its abilities, its architecture
            - asking how the system works
            - asking a general conceptual question that does NOT require web search
            - small “LLM answerable” questions not related to the PDF
            - random chitchat conversations

            ADDITIONAL RULES:
            - If the user asks about the system (e.g., "how do you work?"), always choose "general".
            - If the user greets, always choose "general".
            - If the query can be answered by reasoning alone (without PDFs or web), choose "general".

            OUTPUT:
            Respond ONLY with one word:
            rag
            web
            general
            """

        # Avoid using web unless absolutely necessary
        # If query can be answered by general LLM, prefer general_agent

        decision = llm.invoke(
            f"{system_prompt}\nUser question: {query}\nYour answer:"
        ).strip().lower()

        if "web" in decision:
            next_agent = "web_agent"
        elif "general" in decision:
            next_agent = "general_agent"
        else:
            next_agent = "rag_agent"
        logger.info(f"[SUPERVISOR] → {next_agent} (LLM decision)")
        return _route(state, next_agent)
    
    except Exception as e:
        logger.exception(f"[SUPERVISOR] Unexpected error: {e}")
        return _route(state, "general_agent")


def _route(state, agent_name):
    new_state = dict(state)
    new_state["next"] = agent_name
    return new_state

def _safe_state_update(state, field, value):
    new_state = dict(state)
    new_state[field] = value
    new_state["messages"] = state["messages"] + [
        {"role": "assistant", "content": value}
    ]
    return new_state



########################################################################################################
############# Agent Graph #####################
########################################################################################################

def build_agent_graph(qa_chain):
    try:
        logger.info("[GRAPH] Building agent workflow...")
        workflow = StateGraph(AgentState)
        

        # Register nodes
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("rag_agent", rag_agent_node)
        workflow.add_node("web_agent", web_agent_node)
        workflow.add_node("general_agent", general_llm_agent_node)


        # Start at supervisor
        workflow.set_entry_point("supervisor")

        # Supervisor → picks rag_agent OR web_agent
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "rag_agent": "rag_agent",
                "web_agent": "web_agent",
                "general_agent": "general_agent",
            }
        )

        # Both agents → END
        workflow.add_edge("rag_agent", END)
        workflow.add_edge("web_agent", END)
        workflow.add_edge("general_agent", END)


        compiled = workflow.compile()
        logger.success("[GRAPH] Build complete.")
        return compiled.bind(qa_chain=qa_chain)
    
    except Exception as e:
        logger.exception(f"[GRAPH] Failed to build graph: {e}")
        raise

# --------------------------
# 4) Main Loop
# --------------------------
if __name__ == "__main__":
    try:
        print("Initializing RAG chain...")
        qa_chain, qdrant_client = initialize_qa_chain()

    except Exception as e:
        logger.critical(f"[MAIN] Failed to initialize QA chain: {e}")
        print("System initialization failed.")
        exit(1)

    agent_graph = build_agent_graph(qa_chain)


    print("\nAgentic RAG Ready. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("Query: ")
            if query.lower() in ["exit", "quit"]:
                qdrant_client.close()
                break

            out = agent_graph.invoke({
                "messages": [{"role": "user", "content": query}],
                "qa_chain": qa_chain
            })

            

            # Case 1: RAG agent used
            if "answer" in out:
                print("\n=== RESPONSE (from docs) ===")
                print(out["answer"])

                print("\n--- SOURCES ---")
                if out.get("sources"):
                    for d in out["sources"]:
                        source_name = d.metadata.get("source", "Unknown")
                        page = d.metadata.get("page", "N/A")
                        # Extract only the filename (no full path)
                        source_filename = os.path.basename(source_name)

                        print(f"{source_filename} (Page {page})")
                    print("\n")
                    continue
                
            # Case 2: Web agent used
            if "web_result" in out:
                print("\n=== RESPONSE (from web) ===")
                print(out["web_result"])
                print("\n")
                continue

            if "general_answer" in out:
                print("\n=== RESPONSE (from llm) ===")
                print(out["general_answer"])
                print("\n")
                continue

            
            print("No response received from any agent.")

        except Exception as loop_err:
            logger.exception(f"[MAIN LOOP] Error: {loop_err}")
            print("An error occurred while processing your request.\n")

