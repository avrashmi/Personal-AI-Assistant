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