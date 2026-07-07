This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: node_modules, .git, venv, .env, .venv, __pycache__, dist, build
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
agents/rag_agent.py
agents/rewrite_agent.py
agents/verify_agent.py
app.py
core/embedding.py
core/llm.py
core/logger.py
core/metrics.py
data/knowledge_db.txt
evaluation/evaluation.py
evaluation/ragas_eval.py
graph/builder.py
graph/state.py
preprocessing.py
requirements.txt
single_file/agent.py
single_file/app.py
utilis/reranker.py
```

# Files

## File: agents/rag_agent.py
```python
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
```

## File: agents/rewrite_agent.py
```python
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
```

## File: agents/verify_agent.py
```python
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
```

## File: app.py
```python
import streamlit as st
from graph.builder import graph
from dotenv import load_dotenv
from core.logger import logger


load_dotenv()

st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖")

st.title("🤖 Personal AI Assistant")
st.write("Ask questions about your calendar or to-do list.")

# -------------------------
# Session Memory
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask something like: What meetings do I have tomorrow?")

if user_input:

    with st.chat_message("user"):
        st.write(user_input)


    try:
        result = graph.invoke({
            "question": user_input,
            "chat_history": st.session_state.messages
        })

        answer = result["answer"]

        st.session_state.messages = result["chat_history"]

        with st.chat_message("assistant"):
            st.write(answer)

        # ---- Monitoring ----
        if "latency" in result:
            st.caption(f"⏱ Response time: {result['latency']}s")

    except Exception as e:
        logger.error(f"App error: {e}")
        st.error("Something went wrong.")
```

## File: core/embedding.py
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from preprocessing import load_and_preprocess

# Load data
docs = load_and_preprocess("data/knowledge_db.txt")
documents = [Document(page_content=d) for d in docs]

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(documents, embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

## File: core/llm.py
```python
'''from langchain_community.llms import Ollama

llm = Ollama(
    model="mistral",
    temperature=0
)'''


from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Initialize LLM
llm = Ollama(
    model="mistral",
    temperature=0
)

# Test call
#response = llm.invoke("Explain RAG in simple terms")
#print(response)
```

## File: core/logger.py
```python
import logging

def setup_logger():
    """
    Initializes and configures the application-wide logger.

    Features:
    - Logs messages to both console and a file (app.log)
    - Includes timestamp, log level, and message
    - Helps debug RAG pipeline (retrieval, LLM, evaluation)

    Why important:
    - Tracks internal system behavior
    - Essential for debugging production issues
    - Provides traceability for each user request
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("agentic-rag")


logger = setup_logger()


'''import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    return logging.getLogger("agentic-rag")

logger = setup_logger()'''
```

## File: core/metrics.py
```python
metrics = {
    "total_requests": 0,
    "failed_requests": 0,
    "response_times": []
}


'''def update_metrics(success: bool, response_time: float):
    """
    Updates system-level monitoring metrics.

    Parameters:
    - success (bool): Whether the request was processed successfully
    - response_time (float): Time taken to process the request (in seconds)

    Tracks:
    - Total number of requests
    - Failed requests
    - Response time distribution

    Why important:
    - Helps monitor system health
    - Detects failures and latency issues
    - Useful for dashboards (Grafana/Prometheus later)
    """

    metrics["total_requests"] += 1

    if not success:
        metrics["failed_requests"] += 1

    metrics["response_times"].append(response_time)
'''

'''
def get_avg_response_time():
    """
    Computes average response time.

    Returns:
    - float: Average latency in seconds

    Why important:
    - Measures system performance
    - Helps identify bottlenecks (LLM vs retrieval)
    """

    if not metrics["response_times"]:
        return 0

    return sum(metrics["response_times"]) / len(metrics["response_times"])'''



import time

def start_timer():
    return time.time()

def end_timer(start_time):
    return round(time.time() - start_time, 3)
```

## File: data/knowledge_db.txt
```
I have a dentist appointment on May 26 at 3 PM at Bright Smile Dental Clinic.

Tomorrow I have a project planning meeting with the product team from 9 AM to 10 AM.

Tomorrow I have a client status meeting from 2 PM to 3 PM.

Tomorrow I have a performance review meeting with my manager from 6 PM to 6:30 PM.

Tomorrow morning my schedule includes a project planning meeting from 9 AM to 10 AM.

This week I also have a gym appointment on May 28 at 6 PM.

My to-do list for tomorrow includes finishing the project report, preparing slides for the team meeting, and calling the bank about my credit card issue.

I need to submit the tax document tomorrow before 5 PM.

My current to-do list includes finishing the project report, preparing presentation slides, updating the project documentation, and responding to pending emails.

I still have some pending tasks including updating the project documentation and replying to three important emails from the client.

Today I should complete the task of reviewing the project code, sending the weekly status report, and organizing files for the upcoming team meeting.
```

## File: evaluation/evaluation.py
```python
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
```

## File: evaluation/ragas_eval.py
```python
from ragas import evaluate
from datasets import Dataset

def ragas_evaluate(query, answer, contexts):
    """
    Runs RAGAS evaluation on a single query.

    Metrics:
    - faithfulness
    - answer_relevancy
    - context_precision

    Returns:
    - dict of evaluation scores
    """

    dataset = Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [contexts]
    })

    result = evaluate(dataset)

    return result
```

## File: graph/builder.py
```python
from langgraph.graph import StateGraph

from graph.state import State

from agents.rewrite_agent import rewrite_query
from agents.rag_agent import rag_node
from agents.verify_agent import verify_answer


builder = StateGraph(State)

builder.add_node("rewrite", rewrite_query)
builder.add_node("rag", rag_node)
builder.add_node("verify", verify_answer)

builder.set_entry_point("rewrite")

builder.add_edge("rewrite", "rag")
builder.add_edge("rag", "verify")

builder.set_finish_point("verify")

graph = builder.compile()
```

## File: graph/state.py
```python
from typing import TypedDict, List

class State(TypedDict):
    question: str
    answer: str
    context: str
    chat_history: List[dict]
```

## File: preprocessing.py
```python
from datetime import datetime, timedelta
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.logger import logger


def normalize_dates(text):

    today = datetime.today()

    if "tomorrow" in text.lower():
        tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        text = re.sub(r"tomorrow", tomorrow, text, flags=re.I)

    if "today" in text.lower():
        today_str = today.strftime("%Y-%m-%d")
        text = re.sub(r"today", today_str, text, flags=re.I)

    return text


def load_and_preprocess(file_path):

    logger.info("PREPROCESS: Opening file")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    logger.info("PREPROCESS: File loaded")

    # Basic cleaning
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    logger.info("PREPROCESS: Text cleaned")


    # -------------------------
    # Chunking
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    documents = splitter.split_text(text)

    logger.info(f"PREPROCESS: Total chunks created: {len(documents)}")

    return documents
```

## File: requirements.txt
```
langchain
langgraph
langchain-community
sentence-transformers
faiss-cpu
transformers
accelerate
python-dateutil
```

## File: single_file/agent.py
```python
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
```

## File: single_file/app.py
```python
import streamlit as st
from single_file.agent import graph

st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖")

st.title("🤖 Personal AI Assistant")
st.write("Ask questions about your calendar or to-do list.")


# -------------------------
# Session Memory
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------
# Display chat
# -------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------
# User input
# -------------------------

user_input = st.chat_input("Ask something like: What meetings do I have tomorrow?")


if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    # Call graph with memory
    result = graph.invoke({
        "question": user_input,
        "chat_history": st.session_state.messages
    })

    answer = result["answer"]

    # Update session memory
    st.session_state.messages = result["chat_history"]

    with st.chat_message("assistant"):
        st.write(answer)
```

## File: utilis/reranker.py
```python
def rerank_docs(query, docs):
    """Simple keyword-based reranking.

    Scores retrieved documents based on
    keyword overlap with the user question."""

    query_words = set(query.lower().split())

    scored_docs = []
    for doc in docs:
        text_words = set(doc.page_content.lower().split())
        score = len(query_words.intersection(text_words))
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:2]]
```
