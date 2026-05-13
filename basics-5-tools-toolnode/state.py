from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class FinanceState(TypedDict):
    # add_messages appends every new message (HumanMessage, AIMessage, ToolMessage)
    # instead of replacing the list. This gives the agent the full conversation
    # history — including tool results — on every subsequent turn.
    messages: Annotated[list[BaseMessage], add_messages]
