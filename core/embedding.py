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