from core.llm import llm
from graph.state import State
from core.logger import logger

# -------------------------
# Query Rewriting Agent
# -------------------------

def rewrite_query(state: State):
    question = state["question"]
    chat_history = state.get("chat_history", [])

    logger.info(f"Rewriting query: {question}")

    # Build history context
    history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]]
    )

    prompt = f"""
You are a query rewriting assistant.

Given the conversation history and the latest question,
rewrite the question into a clear, standalone query.

Conversation History:
{history}

Follow-up Question:
{question}

Rewritten Standalone Question:
"""

    rewritten = llm.invoke(prompt).strip()

    logger.info(f"Rewritten query: {rewritten}")

    return {"question": rewritten}