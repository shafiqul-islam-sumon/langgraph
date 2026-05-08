import os

from llm import GeminiLLM
from state import SupportState

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    """Reads a prompt template from the prompts/ folder."""
    with open(os.path.join(_PROMPTS_DIR, filename), "r") as f:
        return f.read()


class SupportNodes:
    """Contains all node functions for the Customer Support Router graph.

    classify_node decides which branch to take; the three support nodes
    each handle one category of customer message with a tailored prompt.
    """

    def __init__(self):
        self.llm               = GeminiLLM().get_llm()
        self.classify_prompt   = _load_prompt("classify.txt")
        self.billing_prompt    = _load_prompt("billing.txt")
        self.technical_prompt  = _load_prompt("technical.txt")
        self.general_prompt    = _load_prompt("general.txt")

    def classify_node(self, state: SupportState) -> dict:
        """Classifies the customer message as billing, technical, or general.

        Returns only the 'category' key. LangGraph merges this into state
        before calling the router function.
        """
        prompt   = self.classify_prompt.format(message=state["message"])
        response = self.llm.invoke(prompt)
        raw      = self._extract_text(response).strip().lower()
        # Guard: accept only expected values, fall back to "general"
        category = raw if raw in ("billing", "technical", "general") else "general"
        return {"category": category}

    def billing_node(self, state: SupportState) -> dict:
        """Handles billing and payment-related customer messages."""
        prompt = self.billing_prompt.format(message=state["message"])
        return {"response": self._extract_text(self.llm.invoke(prompt)).strip()}

    def technical_node(self, state: SupportState) -> dict:
        """Handles technical issues and product troubleshooting messages."""
        prompt = self.technical_prompt.format(message=state["message"])
        return {"response": self._extract_text(self.llm.invoke(prompt)).strip()}

    def general_node(self, state: SupportState) -> dict:
        """Handles general inquiries and all other customer messages."""
        prompt = self.general_prompt.format(message=state["message"])
        return {"response": self._extract_text(self.llm.invoke(prompt)).strip()}

    def _extract_text(self, response) -> str:
        """Handles both plain string and list-of-dicts response formats from Gemini."""
        content = response.content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return content
