from langchain_core.messages import HumanMessage

from graph import StudyBuddyGraph


class StudyRunner:
    def __init__(self, checkpointer=None):
        self.study_graph = StudyBuddyGraph(checkpointer=checkpointer)
        self.app = self.study_graph.get_compiled_graph()

    def save_figure(self):
        self.study_graph.save_figure()

    def chat(self, message: str, thread_id: str) -> str:
        """Send one message and return the study buddy's complete reply."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        content = result["messages"][-1].content
        # langchain-google-genai 4.x returns content as a list of typed blocks
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return content

    def stream_chat(self, message: str, thread_id: str):
        """Yield the study buddy's reply one token at a time."""
        config = {"configurable": {"thread_id": thread_id}}
        for chunk, _ in self.app.stream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="messages",
        ):
            if not hasattr(chunk, "content"):
                continue
            content = chunk.content
            # langchain-google-genai 4.x returns content as a list of typed blocks
            if isinstance(content, str) and content:
                yield content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            yield text

    def get_history(self, thread_id: str) -> list:
        """Return all messages stored for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.app.get_state(config)
        return state.values.get("messages", [])

    def get_history_snapshots(self, thread_id: str) -> list:
        """Return every checkpoint snapshot ever saved for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return list(self.app.get_state_history(config))


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SEP = "=" * 60

    print(SEP)
    print("   LangGraph Basics — Personal Study Buddy Demo")
    print(SEP)

    runner = StudyRunner()

    print("\n  Saving graph architecture...")
    runner.save_figure()

    # ── Alice's multi-turn study session ─────────────────────────────────────
    alice_thread = "alice-session-001"

    alice_questions = [
        "Can you explain recursion to me? I'm a complete beginner.",
        "That makes sense! Can you show me a Python code example?",
        "How is recursion different from a regular for loop?",
    ]

    print("\n👩‍🎓  Alice's Study Session (thread: alice-session-001)")
    print("-" * 60)

    for question in alice_questions:
        print(f"\n🙋  Alice: {question}")
        reply = runner.chat(question, alice_thread)
        print(f"\n🤖  Study Buddy:\n{reply}")
        print("-" * 60)

    # ── Bob starts a completely separate session ──────────────────────────────
    bob_thread = "bob-session-001"

    print("\n\n👨‍🎓  Bob's Study Session (thread: bob-session-001)")
    print("-" * 60)

    bob_question = "What is a Python list comprehension?"
    print(f"\n🙋  Bob: {bob_question}")
    reply = runner.chat(bob_question, bob_thread)
    print(f"\n🤖  Study Buddy:\n{reply}")
    print("-" * 60)

    # ── Thread isolation proof ────────────────────────────────────────────────
    print("\n\n🔍  Thread Isolation Proof")
    print("-" * 60)
    follow_up = "Can you remind me — what analogy did you use earlier when explaining recursion?"
    print(f"\n🙋  Alice (follow-up): {follow_up}")
    reply = runner.chat(follow_up, alice_thread)
    print(f"\n🤖  Study Buddy:\n{reply}")
    print("-" * 60)

    # ── Inspect saved state ───────────────────────────────────────────────────
    print("\n\n📋  Alice's saved message history:")
    print("-" * 60)
    history = runner.get_history(alice_thread)
    for i, msg in enumerate(history, 1):
        role    = type(msg).__name__
        content = msg.content
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        preview = content[:80].replace("\n", " ")
        print(f"  [{i}] {role}: {preview}...")

    print(f"\n  Total messages stored for Alice: {len(history)}")
    print(SEP)
