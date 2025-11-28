import os
from tavily import TavilyClient
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
import numpy as np


# Load embeddings once
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

def internet_search(query: str):
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.error("Missing Tavily API Key")
            return []

        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(
            query=query,
            max_results=5,
            include_raw_content=False
        )
        return response.get("results", [])

    except Exception as e:
        logger.error(f"Tavily error: {e}")
        return []


def choose_best_result(query: str, results: list):
    """Select the most relevant search result using embeddings + LLM validation."""
    if not results:
        return None

    try:
        q_vec = emb.embed_query(query)
        scores = []

        for r in results:
            text = r.get("content", "")
            if not text.strip():
                scores.append(-1)
                continue

            vec = emb.embed_query(text)
            score = np.dot(q_vec, vec)
            scores.append(score)

        # Take top 2 candidates
        top_idx = np.argsort(scores)[::-1][:2]

        candidates = [results[i] for i in top_idx if scores[i] > 0.1]

        # If still empty → nothing relevant
        if not candidates:
            return None

        # LLM final selection
        llm = Ollama(model="phi3:mini", base_url="http://localhost:11434")

        numbered = ""
        for i, c in enumerate(candidates):
            numbered += f"{i+1}. {c.get('content', '')[:500]}\n\n"

        selector_prompt = f"""
You are selecting the single search result that most accurately answers the question.

QUESTION:
{query}

CANDIDATES:
{numbered}

Respond ONLY with the number of the correct result (1 or 2).
If neither candidate is clearly correct, respond with "0".
"""

        selection = llm.invoke(selector_prompt).strip()

        if selection == "1":
            return candidates[0]
        elif selection == "2" and len(candidates) > 1:
            return candidates[1]
        else:
            return None

    except Exception as e:
        logger.error(f"Selection error: {e}")
        return None


def web_agent_node(state):
    query = state["messages"][-1]["content"]

    # Step 1: Get all results
    results = internet_search(query)

    if not results:
        return _update(state, "No information found online.")

    # Step 2: Choose best result
    best = choose_best_result(query, results)

    if best is None:
        return _update(state, "No reliable information found online.")

    text = best.get("content", "No information.")

    # Step 3: Summarize that result only
    llm = Ollama(model="phi3:mini", base_url="http://localhost:11434")

    prompt = f"""
Summarize the following information into a clear, correct answer.

Query: {query}

Text:
{text}

Give one accurate answer. Do not add extra information.
"""

    try:
        summary = llm.invoke(prompt).strip()
    except:
        summary = "Unable to summarize the online information."

    return _update(state, summary)


def _update(state, result):
    new_state = dict(state)
    new_state["web_result"] = result
    new_state["messages"] = state["messages"] + [
        {"role": "assistant", "content": result, "agent": "web"}
    ]
    return new_state
