from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph

from preprocessing import load_and_preprocess
from typing import TypedDict, List


print("DEBUG 1: Starting agent setup")


# -------------------------
# Load data
# -------------------------

print("DEBUG 2: Loading knowledge file")

docs = load_and_preprocess("data/knowledge_db.txt")

print("DEBUG 3: Loaded docs:", len(docs))

documents = [Document(page_content=d) for d in docs]

print("DEBUG 4: Converted to LangChain Documents")


# -------------------------
# Embeddings
# -------------------------

print("DEBUG 5: Loading embedding model")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("DEBUG 6: Creating FAISS vector store")

vectorstore = FAISS.from_documents(documents, embedding_model)

print("DEBUG 7: Vectorstore created")

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

print("DEBUG 8: Retriever ready")


# -------------------------
# LLM (Mistral)
# -------------------------

print("DEBUG 9: Loading Mistral model")

llm = Ollama(
    model="mistral",
    temperature=0
)

print("DEBUG 10: LLM ready")


# -------------------------
# Conversation Memory
# -------------------------

"""
Simple in-memory conversation history.

Stores last interactions so the assistant can
understand follow-up questions.
"""

# chat_history = []


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


# -------------------------
# Query Rewriting Agent
# -------------------------

def rewrite_query(state):
    """
    Agent that rewrites vague user queries into
    clearer search queries for the vector database.
    """

    question = state["question"]

    rewrite_prompt = f"""
Rewrite the following question so it is clearer for searching a personal calendar database.

Original Question:
{question}

Rewritten Question:
"""

    rewritten = llm.invoke(rewrite_prompt).strip()

    return {"question": rewritten}


# -------------------------
# Document Reranker
# -------------------------

def rerank_docs(query, docs):
    """Simple keyword-based reranking.

    Scores retrieved documents based on
    keyword overlap with the user question."""

    scored_docs = []

    query_words = set(query.lower().split())

    for doc in docs:

        text_words = set(doc.page_content.lower().split())
        score = len(query_words.intersection(text_words))

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:2]]


# -------------------------
# State
# -------------------------

class State(TypedDict):
    question: str
    answer: str
    context: str
    chat_history: List[dict]


print("DEBUG 18: Creating LangGraph")


# -------------------------
# RAG Answer Generator
# -------------------------

def rag_node(state: State):

    question = state["question"]
    chat_history = state.get("chat_history", [])

    # Retrieve documents
    docs = retriever.invoke(question)

    # Rerank
    docs = rerank_docs(question, docs)

    # Build context
    context = "\n".join([d.page_content for d in docs])

    # Build history text
    history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]]
    )

    # Build prompt
    formatted_prompt = prompt.format(
        context=context,
        question=question,
        history=history
    )

    # LLM answer
    response = llm.invoke(formatted_prompt)

    answer = response.strip()

    return {
        "answer": answer,
        "context": context,
        "question": question,
        "chat_history": chat_history
    }



# -------------------------
# Hallucination Checker
# -------------------------

def verify_answer(state: State):
    """
    Agent that verifies whether the generated
    answer is supported by the retrieved context.
    """

    answer = state["answer"]
    context = state["context"]
    question = state["question"]
    chat_history = state.get("chat_history", [])

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

# -------------------------
# LangGraph Workflow
# -------------------------

builder = StateGraph(State)

builder.add_node("rewrite", rewrite_query)
builder.add_node("rag", rag_node)
builder.add_node("verify", verify_answer)

builder.set_entry_point("rewrite")

builder.add_edge("rewrite", "rag")
builder.add_edge("rag", "verify")

builder.set_finish_point("verify")

graph = builder.compile()

print("DEBUG 19: Graph compiled successfully")
