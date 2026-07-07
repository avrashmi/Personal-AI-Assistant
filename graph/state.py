from typing import TypedDict, List

class State(TypedDict):
    question: str
    answer: str
    context: str
    chat_history: List[dict]