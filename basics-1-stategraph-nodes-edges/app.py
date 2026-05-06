import gradio as gr

from qa_runner import QARunner


class QAApp:
    """Gradio web interface for the LangGraph Q&A bot.

    Wraps QARunner in a ChatInterface so the same graph that runs in the
    console (qa_runner.py) can be used interactively in the browser.
    """

    def __init__(self):
        self.runner = QARunner()

    def respond(self, message: str, _history: list) -> str:
        """Gradio ChatInterface callback.

        _history is required by the ChatInterface signature but not used here —
        each question is stateless (LangGraph state resets on every invoke call).
        """
        if not message.strip():
            return ""
        return self.runner.run(message)

    def launch(self):
        gr.ChatInterface(
            fn=self.respond,
            title="🤖 LangGraph Chatbot",
        ).launch()


if __name__ == "__main__":
    QAApp().launch()
