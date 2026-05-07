import gradio as gr

from topic_runner import TopicRunner


class TopicApp:
    """Gradio ChatInterface wrapping the Topic Expander runner."""

    def __init__(self):
        self.runner = TopicRunner()

    def respond(self, message: str, _history: list) -> str:
        # _history is required by Gradio's ChatInterface signature but not used here
        if not message.strip():
            return ""
        result = self.runner.run(message)
        return self.runner.format_output(result)

    def launch(self):
        gr.ChatInterface(fn=self.respond, title="🌐 Topic Expander").launch()


if __name__ == "__main__":
    TopicApp().launch()
