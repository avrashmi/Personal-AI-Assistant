#Not in use
def evaluate_response(query: str, context: str, answer: str) -> dict:
    """
    Evaluates the quality of a generated RAG response.

    Parameters:
    - query (str): User input question
    - context (str): Retrieved documents used for generation
    - answer (str): LLM-generated response

    Returns:
    - dict:
        {
            "score": float (0 to 1),
            "status": "good" | "needs_improvement"
        }

    Evaluation Criteria:
    - Context Presence (0.3): Were relevant docs retrieved?
    - Query Match (0.3): Does answer relate to query?
    - Answer Quality (0.4): Is answer sufficiently informative?

    Why important:
    - Measures RAG effectiveness
    - Detects hallucinations / weak answers
    - Enables future auto-retry/self-improvement
    """

    score = 0

    if context.strip():
        score += 0.3

    if query.lower() in answer.lower():
        score += 0.3

    if len(answer) > 20:
        score += 0.4

    return {
        "score": round(score, 2),
        "status": "good" if score >= 0.7 else "needs_improvement"
    }