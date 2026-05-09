import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes import StudyBuddyNodes
from state import StudyState

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figure")


class StudyBuddyGraph:
    def __init__(self, checkpointer=None):
        self.nodes = StudyBuddyNodes()
        # MemorySaver by default. Pass SqliteSaver to persist across restarts.
        self.checkpointer = checkpointer or MemorySaver()
        self.compiled_graph = self._build()

    def _build(self):
        graph = StateGraph(StudyState)

        graph.add_node("study_buddy", self.nodes.study_buddy_node)

        graph.add_edge(START, "study_buddy")
        graph.add_edge("study_buddy", END)

        # The ONLY change from a stateless graph — one argument at compile time.
        return graph.compile(checkpointer=self.checkpointer)

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
        return self.compiled_graph
