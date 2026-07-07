# evaluation/ragas_config.py

'''from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper

# 1. Load your local model
llm = Ollama(model="mistral")

# 2. Wrap it for RAGAS (THIS IS YOUR STEP)
ragas_llm = LangchainLLMWrapper(llm)'''


from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

#llm = Ollama(model="mistral")
#Evaluation model (USED BY RAGAS)


eval_llm = Ollama(
    model="gemma2:2b",
    temperature=0
)


ragas_llm = LangchainLLMWrapper(eval_llm)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    embeddings
)