from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class StudyState(TypedDict):
    # add_messages reducer appends each new message instead of overwriting the list.
    # This gives the study buddy access to the full conversation history on every turn.
    messages: Annotated[list[BaseMessage], add_messages]
