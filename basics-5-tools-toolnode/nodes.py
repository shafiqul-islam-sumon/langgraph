import os

from langchain_core.messages import SystemMessage

from llm import GeminiLLM
from tools import TOOLS

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), "r") as f:
        return f.read()


class FinanceNodes:
    def __init__(self):
        llm = GeminiLLM().get_llm()
        # Bind all four tools so the LLM knows what it can call.
        # bind_tools() injects the tool schemas into the LLM's context.
        self.llm_with_tools = llm.bind_tools(TOOLS)
        self.system_prompt  = _load_prompt("system.txt")

    def agent_node(self, state: dict) -> dict:
        # Prepend the system prompt on every turn so the LLM always has its
        # role and tool-usage instructions — even mid-loop after a tool call.
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
