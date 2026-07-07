import uuid
from core.llm import llm
from core.embedding import retriever
from utilis.reranker import rerank_docs
from langchain_core.prompts import PromptTemplate
from graph.state import State

from core.logger import logger
from core.metrics import start_timer, end_timer

# from core.logger import logger
# from core.metrics import update_metrics

# -------------------------
# Prompt Templates
# -------------------------

prompt = PromptTemplate(
    input_variables=["context", "question", "history"],
    template="""
You are a helpful personal assistant.

Conversation history:
{history}

Use the conversation history and the provided context to answer the question.
Context:
{context}

Question:
{question}

Give a short and direct answer.
"""
)

def rag_node(state: State):
    '''
    Core RAG pipeline step.

    Responsibilities:
    - Retrieve relevant documents
    - Rerank documents
    - Construct prompt using context + history
    - Generate response using LLM

    Observability:
    - Assigns unique request_id
    - Logs each step (retrieval, generation)
    - Measures latency

    Monitoring:
    - Tracks response time

    Returns:
    - Updated state with answer, context, request_id
    '''

    request_id = str(uuid.uuid4())
    start = start_timer()

    question = state["question"]
    chat_history = state.get("chat_history", [])

    logger.info(f"[{request_id}] Question: {question}")

    # ---- Retrieval ----
    docs = retriever.invoke(question)
    logger.info(f"[{request_id}] Retrieved docs: {len(docs)}")

    # ---- Rerank ----
    docs = rerank_docs(question, docs)
    logger.info(f"[{request_id}] Reranked docs: {len(docs)}")


    context = "\n".join([d.page_content for d in docs])

    history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]]
    )

    formatted_prompt = prompt.format(
        context=context,
        question=question,
        history=history
    )

    # ---- LLM ----
    response = llm.invoke(formatted_prompt)

    latency = end_timer(start)
    logger.info(f"[{request_id}] Response generated in {latency}s")

    return {
        "answer": response.strip(),
        "context": context,
        "question": question,
        "chat_history": chat_history,
        "request_id": request_id,
        "latency": latency
    }