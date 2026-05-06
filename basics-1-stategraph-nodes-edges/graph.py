import os

from langgraph.graph import END, START, StateGraph

from nodes import QANodes
from state import QAState

# Output directory for Mermaid diagram exports (.mmd and .png)
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figure")


class QAGraph:
    """Builds and compiles the LangGraph StateGraph for the Q&A bot.

    Graph structure: START → answer → END

    The compiled graph is a CompiledStateGraph, which implements LangChain's
    Runnable interface and exposes .invoke(), .stream(), and .ainvoke().
    """

    def __init__(self):
        self.nodes = QANodes()
        self.compiled_graph = self._build()

    def _build(self):
        """Registers nodes, wires edges, and compiles the graph.

        Called once in __init__. compile() validates the full graph structure
        (reachable nodes, START/END paths) and returns a Runnable.
        """
        graph = StateGraph(QAState)

        # Register the answer node under the name "answer"
        # The string name is what edges reference — not the function name
        graph.add_node("answer", self.nodes.answer_node)

        graph.add_edge(START, "answer")  # graph entry point
        graph.add_edge("answer", END)    # graph exit point

        return graph.compile()

    def save_figure(self):
        """Exports the graph structure as a Mermaid diagram (.mmd) and PNG.

        draw_mermaid_png() requires an internet connection on first call —
        it uses the Mermaid.ink API to render the PNG.
        """
        os.makedirs(FIGURE_DIR, exist_ok=True)

        mermaid_str = self.compiled_graph.get_graph().draw_mermaid()
        mmd_path = os.path.join(FIGURE_DIR, "graph.mmd")
        with open(mmd_path, "w") as f:
            f.write(mermaid_str)

        png_bytes = self.compiled_graph.get_graph().draw_mermaid_png()
        png_path = os.path.join(FIGURE_DIR, "graph.png")
        with open(png_path, "wb") as f:
            f.write(png_bytes)

        print(f"  Graph saved → {mmd_path}")
        print(f"  Graph saved → {png_path}")

    def get_compiled_graph(self):
        """Returns the compiled graph so callers can invoke it directly."""
        return self.compiled_graph
