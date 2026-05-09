import os

from langchain_core.messages import SystemMessage

from llm import GeminiLLM
from state import StudyState

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), "r") as f:
        return f.read()


class StudyBuddyNodes:
    def __init__(self):
        self.llm            = GeminiLLM().get_llm()
        self.system_prompt  = _load_prompt("study_buddy.txt")

    def study_buddy_node(self, state: StudyState) -> dict:
        # Prepend the system prompt so Gemini always has the persona context.
        # state["messages"] contains the full conversation history from all prior
        # turns — the checkpointer loaded it before this node ran.
        system   = SystemMessage(content=self.system_prompt)
        messages = [system] + state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}
