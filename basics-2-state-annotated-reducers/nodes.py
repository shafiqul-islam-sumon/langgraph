import os

from llm import GeminiLLM
from state import TopicState

# Prompts live in the prompts/ subfolder — one text file per node
_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    """Reads a prompt template from the prompts/ folder."""
    with open(os.path.join(_PROMPTS_DIR, filename), "r") as f:
        return f.read()


class TopicNodes:
    """Contains all node functions for the Topic Expander graph.

    The LLM and prompts are initialised once in __init__ and reused across
    calls, avoiding repeated file I/O and client creation.
    """

    def __init__(self):
        self.llm = GeminiLLM().get_llm()
        self.expand_prompt = _load_prompt("expand_node.txt")
        self.refine_prompt = _load_prompt("refine_node.txt")

    def expand_node(self, state: TopicState) -> dict:
        """Generates three key points about the given topic.

        Returns a partial state update — only the 'points' key. Items are
        prefixed with [Expand] so the output clearly shows which node produced
        them. LangGraph calls operator.add(current_points, new_points) to
        merge the list rather than replacing it.
        """
        prompt = self.expand_prompt.format(topic=state["topic"])
        response = self.llm.invoke(prompt)
        content = self._extract_text(response)
        lines = [
            f"[Expand] {ln.strip()}"
            for ln in content.strip().splitlines()
            if ln.strip()
        ]
        return {"points": lines[:3]}

    def refine_node(self, state: TopicState) -> dict:
        """Adds a deeper insight for each point produced by expand_node.

        Reads the accumulated points from state, generates one insight per
        point, and returns them prefixed with [Refine]. The operator.add
        reducer appends these to the existing points — they do not overwrite.
        Also sets the final summary string.
        """
        existing = "\n".join(f"- {p}" for p in state["points"])
        prompt = self.refine_prompt.format(topic=state["topic"], existing=existing)
        response = self.llm.invoke(prompt)
        content = self._extract_text(response)
        insights = [
            f"[Refine] {ln.strip()}"
            for ln in content.strip().splitlines()
            if ln.strip()
        ]
        return {
            "points": insights[: len(state["points"])],
            "summary": (
                f"Explored '{state['topic']}' across {len(state['points'])} "
                "key aspects with deeper insights."
            ),
        }

    def _extract_text(self, response) -> str:
        """Handles both plain string and list-of-dicts response formats.

        langchain-google-genai 4.x returns response.content as a list of
        content blocks instead of a plain string, so both formats are handled.
        """
        content = response.content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return content
