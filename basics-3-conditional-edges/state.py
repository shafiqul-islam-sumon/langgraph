from typing import TypedDict


class SupportState(TypedDict):
    """Shared state that flows through every node in the Customer Support Router graph.

    - message:   The customer's support message.
    - category:  Populated by classify_node; one of "billing", "technical", "general".
    - response:  The support reply written by whichever answer node runs.
    """

    message: str
    category: str  # set by classify_node; drives the conditional edge
    response: str
