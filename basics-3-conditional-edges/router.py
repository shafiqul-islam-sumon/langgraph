from typing import Literal

from state import SupportState


def route_by_category(
    state: SupportState,
) -> Literal["billing_support", "technical_support", "general_support"]:
    """Routes execution to the support node that matches the classified category.

    Called by LangGraph after classify_node runs. Reads state["category"] and
    returns the name of the next node as a string. The Literal type hint
    documents every valid return value explicitly.
    """
    mapping = {
        "billing":   "billing_support",
        "technical": "technical_support",
        "general":   "general_support",
    }
    # Falls back to general_support if the LLM returns an unexpected value
    return mapping.get(state["category"], "general_support")
