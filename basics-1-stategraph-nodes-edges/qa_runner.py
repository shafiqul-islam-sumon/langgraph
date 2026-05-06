from graph import QAGraph


class QARunner:
    """Thin wrapper around QAGraph that exposes a clean run() interface.

    Separates graph construction (QAGraph) from execution — callers only
    need to call run(question) and get a string back.
    """

    def __init__(self):
        self.qa_graph = QAGraph()
        # get_compiled_graph() returns the CompiledStateGraph Runnable
        self.app = self.qa_graph.get_compiled_graph()

    def save_figure(self):
        self.qa_graph.save_figure()

    def run(self, question: str) -> str:
        """Invokes the compiled graph with the question and returns the answer.

        invoke() runs the graph synchronously: START → answer_node → END.
        It returns the final state dict; we extract just the 'answer' field.
        """
        result = self.app.invoke({"question": question})
        return result["answer"]


if __name__ == "__main__":
    runner = QARunner()

    print("\n" + "=" * 60)
    print("         LangGraph Basics — Q&A Bot Demo")
    print("=" * 60)

    print("\n Saving graph architecture...")
    runner.save_figure()

    questions = [
        "What is LangGraph in one sentence?",
        "What is the difference between a node and an edge in LangGraph?",
        "Why do we need state in LangGraph?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}] {question}")
        print("-" * 60)
        print(runner.run(question))
        print("=" * 60)
