import operator
from typing import Annotated, TypedDict


class TopicState(TypedDict):
    """Shared state that flows through every node in the Topic Expander graph.

    - topic:   The input subject the user wants to explore.
    - points:  Accumulates across nodes. Annotated with operator.add so that
               each node's returned list is appended rather than replaced.
    - summary: Plain string written once by the final node.
    """

    topic: str
    points: Annotated[list[str], operator.add]  # reducer: concatenates lists
    summary: str
