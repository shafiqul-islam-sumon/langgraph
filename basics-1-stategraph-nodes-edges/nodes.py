from llm import GeminiLLM
from state import QAState


class QANodes:
    """Contains all node functions for the Q&A graph.

    The LLM is initialised once in __init__ and reused across calls,
    avoiding the overhead of creating a new client on every invocation.
    """

    def __init__(self):
        self.llm = GeminiLLM().get_llm()

    def answer_node(self, state: QAState) -> dict:
        """Sends the question to Gemini and returns the answer.

        Returns a partial state update (only the 'answer' key) — not the full
        QAState. LangGraph merges this dict back into the existing state.

        langchain-google-genai 4.x returns response.content as a list of
        content blocks instead of a plain string, so both formats are handled.
        """
        response = self.llm.invoke(state["question"])
        content = response.content

        # Flatten list of content blocks into a single string (4.x API format)
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        return {"answer": content}
