import os

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from nodes import FinanceNodes
from state import FinanceState
from tools import TOOLS

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figure")


class FinanceGraph:
    def __init__(self):
        self.nodes          = FinanceNodes()
        self.compiled_graph = self._build()

    def _build(self):
        graph = StateGraph(FinanceState)

        # Register nodes
        graph.add_node("agent", self.nodes.agent_node)
        graph.add_node("tools", ToolNode(TOOLS))

        # Wire the ReAct loop — three edges
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)  # agent → tools | END
        graph.add_edge("tools", "agent")                       # after tools, back to agent

        return graph.compile()

    def save_figure(self):
        """Export the graph structure as a Mermaid diagram (.mmd) and PNG."""
        os.makedirs(FIGURE_DIR, exist_ok=True)

        mermaid_str = self.compiled_graph.get_graph().draw_mermaid()
        mmd_path    = os.path.join(FIGURE_DIR, "graph.mmd")
        with open(mmd_path, "w") as f:
            f.write(mermaid_str)

        png_bytes = self.compiled_graph.get_graph().draw_mermaid_png()
        png_path  = os.path.join(FIGURE_DIR, "graph.png")
        with open(png_path, "wb") as f:
            f.write(png_bytes)

        print(f"  Graph saved → {mmd_path}")
        print(f"  Graph saved → {png_path}")

    def get_compiled_graph(self):
        return self.compiled_graph
