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

#graph = builder.compile()

compiled_graph = builder.compile()