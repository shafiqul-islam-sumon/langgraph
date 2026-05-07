from graph import TopicGraph


class TopicRunner:
    """Entry point for running the Topic Expander graph.

    Wraps TopicGraph and provides a clean run() interface. The runner calls
    invoke() on the CompiledStateGraph — not on TopicGraph itself.
    """

    def __init__(self):
        self.topic_graph = TopicGraph()
        # self.app holds the CompiledStateGraph returned by get_compiled_graph()
        self.app = self.topic_graph.get_compiled_graph()

    def save_figure(self):
        self.topic_graph.save_figure()

    def run(self, topic: str) -> dict:
        """Runs the full graph for a given topic and returns the final state."""
        # points must be initialised as an empty list; operator.add will append to it
        return self.app.invoke({"topic": topic, "points": [], "summary": ""})

    def format_output(self, result: dict) -> str:
        """Formats the final state into a human-readable string."""
        lines = [f"📌 Topic: {result['topic']}", "─" * 60, ""]
        lines.append("Points collected:")
        for i, point in enumerate(result["points"], 1):
            lines.append(f"  {i}. {point}")
        lines.append("")
        lines.append(f"📝 Summary: {result['summary']}")
        return "\n".join(lines)


if __name__ == "__main__":
    runner = TopicRunner()

    print("\n" + "=" * 60)
    print("    LangGraph Basics — Topic Expander Demo")
    print("=" * 60)

    print("\n Saving graph architecture...")
    runner.save_figure()

    topics = ["Machine Learning", "Climate Change"]
    for topic in topics:
        print()
        result = runner.run(topic)
        print(runner.format_output(result))
        print("=" * 60)
