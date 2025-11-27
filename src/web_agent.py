import os
from typing import TypedDict, Optional, List
from tavily import TavilyClient
from langchain_community.llms import Ollama
from loguru import logger

logger.add("agent_web.log", rotation="5 MB", level="DEBUG")


import os
from tavily import TavilyClient

def internet_search(query: str):
    """Return raw Tavily results for summarization."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            logger.error("[WEB AGENT] Missing TAVILY_API_KEY.")
            return []

        tavily = TavilyClient(api_key=api_key)
        logger.info(f"[WEB AGENT] Tavily search triggered...")

        response = tavily.search(
            query=query,
            max_results=5,
            include_raw_content=False
        )

        return response.get("results", [])

    except Exception as tavily_err:
        logger.error(f"[WEB AGENT] Tavily search failed: {tavily_err}")
        return []

def web_agent_node(state):
    try:
        query = state["messages"][-1]["content"]
        logger.info(f"[WEB AGENT] Received query...")

    # Get Tavily results
        results = internet_search(query)

        if not results:
            summary = "No internet results found."
            logger.warning("[WEB AGENT] No results received from Tavily.")
            return _update_state(state, summary)
        
        try:
            # Extract text content from Tavily results
            content_blocks = [r.get("content", "") for r in results if r.get("content")]
            
            combined_text = "\n\n".join(content_blocks)
            if not combined_text.strip():
                logger.warning("[WEB AGENT] Tavily results contained no usable text.")
                summary = "No meaningful content found in search results."
                return _update_state(state, summary)

            logger.debug("[WEB AGENT] Combined Tavily text ready for summarization.")

        except Exception as extract_err:
            logger.error(f"[WEB AGENT] Failed to extract Tavily content: {extract_err}")
            summary = "Internet search returned unreadable data."
            return _update_state(state, summary)

        try:

            # Summarize using LLM
            llm = Ollama(
                model="phi3:mini",
                base_url="http://localhost:11434",
                temperature=0.3
            )

            prompt = f"""
            You are an intelligent summarizing assistant.
            Summarize the key information from the following internet search snippets 
            into ONE clear, concise answer. Avoid redundancy. 
            Do not mention that this is a summary.

            Query: {query}

            Snippets:
            {combined_text}

            Provide one clean answer:
            """

            summary = llm.invoke(prompt).strip()
            logger.info(f"[WEB AGENT] Summary generated successfully.")

        except Exception as summarization_err:
            logger.error(f"[WEB AGENT] LLM summarization failed: {summarization_err}")
            summary = "I couldn't summarize the search results due to an internal error."

        return _update_state(state, summary)
    
    except Exception as e:
        logger.exception(f"[WEB AGENT] Unexpected agent failure: {e}")
        fallback = "The web agent encountered an unexpected error."
        return _update_state(state, fallback)
    
def _update_state(state, summary: str):

    # Update state
    new_state = dict(state)
    new_state["web_result"] = summary
    new_state["messages"] = state["messages"] + [
        {
            "role": "assistant",
            "content": summary,
            "agent": "web"
        }
    ]

    return new_state