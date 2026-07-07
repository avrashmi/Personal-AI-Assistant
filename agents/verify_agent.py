from core.llm import llm
from evaluation.ragas_eval import ragas_evaluate
from graph.state import State
from core.logger import logger


def verify_answer(state: State):
    """
    Agent that verifies whether the generated
    answer is supported by the retrieved context.
    """

    answer = state["answer"]
    context = state["context"]
    question = state["question"]
    chat_history = state.get("chat_history", [])
    request_id = state.get("request_id", "N/A")
    
    contexts_list = context.split("\n")

    # ---- RAGAS Evaluation ----
    ragas_result = ragas_evaluate(question, answer, contexts_list)

    logger.info(f"[{request_id}] RAGAS: {ragas_result}")

    # ---- LLM Verification ----

    verify_prompt = f"""
Check if the answer is supported by the context.

Context:
{context}

Question:
{question}

Answer:
{answer}

If the answer is correct respond YES.
If not respond NO.
"""

    result = llm.invoke(verify_prompt)

    if "NO" in result.upper():
        logger.warning(f"[{request_id}] Answer not grounded. Replacing.")
        answer = "I could not find that information in your stored data."

    # Update memory (FINAL answer)
    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return {
        "answer": answer,
        "chat_history": chat_history
    }
