from langchain_core.messages import HumanMessage

from graph import FinanceGraph


class FinanceRunner:
    def __init__(self):
        self.finance_graph = FinanceGraph()
        self.app           = self.finance_graph.get_compiled_graph()

    def save_figure(self):
        self.finance_graph.save_figure()

    def chat(self, message: str) -> str:
        """Send one message and return the finance assistant's complete reply."""
        result  = self.app.invoke({"messages": [HumanMessage(content=message)]})
        content = result["messages"][-1].content
        # langchain-google-genai 4.x returns content as a list of typed blocks
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return content


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SEP = "=" * 60

    print(SEP)
    print("   LangGraph Basics — Personal Finance Assistant Demo")
    print(SEP)

    runner = FinanceRunner()

    print("\n  Saving graph architecture...")
    runner.save_figure()

    queries = [
        "What would my monthly payment be on a $20,000 car loan at 6% interest for 48 months?",
        "If I invest $5,000 at 7% annual interest, how much will I have after 10 years?",
        "How much is 1,000 USD in EUR?",
        "I earn $4,500 per month. Split my budget: 30% rent, 20% food, and 20% savings.",
    ]

    print()
    for query in queries:
        print(f"💬 User:     {query}")
        print(f"🤖 Assistant: {runner.chat(query)}")
        print("-" * 60)

    print(SEP)
