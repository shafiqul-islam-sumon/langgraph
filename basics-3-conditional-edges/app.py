import gradio as gr

from support_runner import SupportRunner


class SupportApp:
    """Gradio ChatInterface wrapping the Customer Support Router runner."""

    def __init__(self):
        self.runner = SupportRunner()

    def respond(self, message: str, _history: list) -> str:
        # _history is required by Gradio's ChatInterface signature but not used here
        if not message.strip():
            return ""
        result = self.runner.run(message)
        return self.runner.format_output(result)

    def launch(self):
        gr.ChatInterface(fn=self.respond, title="🎧 Customer Support Router").launch()


if __name__ == "__main__":
    SupportApp().launch()
