from graph import SupportGraph


class SupportRunner:
    """Entry point for running the Customer Support Router graph.

    Wraps SupportGraph and provides a clean run() interface. The runner calls
    invoke() on the CompiledStateGraph — not on SupportGraph itself.
    """

    def __init__(self):
        self.support_graph = SupportGraph()
        self.app           = self.support_graph.get_compiled_graph()

    def save_figure(self):
        self.support_graph.save_figure()

    def run(self, message: str) -> dict:
        """Runs the full graph for a given customer message and returns the final state."""
        return self.app.invoke({"message": message, "category": "", "response": ""})

    def format_output(self, result: dict) -> str:
        """Formats the final state into a human-readable string."""
        lines = [
            f"📩 Message  : {result['message']}",
            f"🏷️  Category : {result['category'].upper()}",
            "─" * 60,
            f"💬 Response :\n{result['response']}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    runner = SupportRunner()

    print("\n" + "=" * 60)
    print("    LangGraph Basics — Customer Support Router Demo")
    print("=" * 60)

    print("\n  Saving graph architecture...")
    runner.save_figure()

    messages = [
        "I was charged twice for my subscription this month.",
        "My app keeps crashing whenever I try to upload a file.",
        "What are your customer support hours?",
    ]
    for message in messages:
        print()
        result = runner.run(message)
        print(runner.format_output(result))
        print("=" * 60)
