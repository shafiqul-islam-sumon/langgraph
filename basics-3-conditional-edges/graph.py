import os

from langgraph.graph import END, START, StateGraph

from nodes import SupportNodes
from router import route_by_category
from state import SupportState

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figure")


class SupportGraph:
    """Builds and compiles the Customer Support Router StateGraph.

    Graph structure (conditional):
        START → classify → [route_by_category] → billing_support  → END
                                                → technical_support → END
                                                → general_support   → END

    The dashed arrows (conditional edges) mean execution follows exactly one
    branch per run, based on the category stored in state after classify runs.
    """

    def __init__(self):
        self.nodes          = SupportNodes()
        self.compiled_graph = self._build()

    def _build(self):
        """Wires nodes and edges, then compiles to a CompiledStateGraph."""
        graph = StateGraph(SupportState)

        # Register all four node functions
        graph.add_node("classify",          self.nodes.classify_node)
        graph.add_node("billing_support",   self.nodes.billing_node)
        graph.add_node("technical_support", self.nodes.technical_node)
        graph.add_node("general_support",   self.nodes.general_node)

        # Fixed edge: every run starts with classify
        graph.add_edge(START, "classify")

        # Conditional edge: route_by_category picks which support node runs
        graph.add_conditional_edges(
            "classify",
            route_by_category,
            {
                "billing_support":   "billing_support",
                "technical_support": "technical_support",
                "general_support":   "general_support",
            },
        )

        # All three support nodes converge at END
        graph.add_edge("billing_support",   END)
        graph.add_edge("technical_support", END)
        graph.add_edge("general_support",   END)

        return graph.compile()

    def save_figure(self):
        """Exports the graph structure as a Mermaid diagram (.mmd) and PNG."""
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
        """Exposes the CompiledStateGraph for the runner to call invoke() on."""
        return self.compiled_graph
