from langchain_community.llms import Ollama
from loguru import logger

logger.add("agent_general.log", rotation="5 MB", level="DEBUG")

def general_llm_agent_node(state):
    """Handles greetings, chit-chat, and general queries."""
    try:
        query = state["messages"][-1]["content"]
        logger.info(f"[GENERAL AGENT] Received query...")

        try:
            llm = Ollama(
                model="phi3:mini",
                base_url="http://localhost:11434",
                temperature=0.5
            )
        except Exception as init_err:
            logger.error(f"[GENERAL AGENT] Failed to initialize LLM: {init_err}")
            reply = "Sorry, I'm unable to load the language model right now."
            return _update_state(state, reply)

        try:
            reply = llm.invoke(query).strip()
        except Exception as gen_err:
            logger.error(f"[GENERAL AGENT] LLM generation failed: {gen_err}")
            reply = "Sorry, I couldn't generate an answer at the moment."
        
        return _update_state(state, reply)
    
    except Exception as e:
        logger.exception(f"[GENERAL AGENT] Unexpected error: {e}")
        fallback = "Something went wrong in the general assistant. Please try again."
        return _update_state(state, fallback)

def _update_state(state, reply):

    new_state = dict(state)
    new_state["general_answer"] = reply
    new_state["messages"] = state["messages"] + [
        {"role": "assistant", "content": reply, "agent": "general"}
    ]
    return new_state
