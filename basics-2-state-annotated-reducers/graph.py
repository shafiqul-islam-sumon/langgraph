import os

from langgraph.graph import END, START, StateGraph

from nodes import TopicNodes
from state import TopicState

# Output directory for Mermaid diagram exports (.mmd and .png)
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figure")


class TopicGraph:
    """Builds and compiles the Topic Expander StateGraph.

    Graph structure (linear — two nodes):
        START → expand → refine → END

    Both nodes write to state['points']. Because the field is declared with
    Annotated[list[str], operator.add], LangGraph concatenates each node's
    returned list instead of replacing the previous value.
    """

    def __init__(self):
        self.nodes = TopicNodes()
        # _build() is called once; the compiled graph is stored and reused
        self.compiled_graph = self._build()

    def _build(self):
        """Wires nodes and edges, then compiles to a CompiledStateGraph."""
        graph = StateGraph(TopicState)

        # Register node functions under string names
        graph.add_node("expand", self.nodes.expand_node)
        graph.add_node("refine", self.nodes.refine_node)

        # Connect the edges: START → expand → refine → END
        graph.add_edge(START, "expand")
        graph.add_edge("expand", "refine")
        graph.add_edge("refine", END)

        # compile() returns a CompiledStateGraph (a LangChain Runnable)
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
        """Exposes the CompiledStateGraph for the runner to call invoke() on."""
        return self.compiled_graph
