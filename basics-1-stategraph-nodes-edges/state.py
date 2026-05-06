from typing import TypedDict


class QAState(TypedDict):
    """Shared state that flows through every node in the graph.

    LangGraph passes this dict from node to node. Each node reads what it needs
    and returns only the keys it changed — LangGraph merges those changes back.
    """

    question: str  # The user's input question
    answer: str    # The LLM's response, populated by answer_node
