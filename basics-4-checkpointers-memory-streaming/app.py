import uuid

import gradio as gr

from study_runner import StudyRunner


class StudyBuddyApp:
    def __init__(self):
        self.runner = StudyRunner()

    def respond(self, message: str, _history: list, thread_id: str):
        """Stream the study buddy reply into the chat UI."""
        if not message.strip():
            yield ""
            return
        # gr.ChatInterface expects each yield to be the full response so far
        accumulated = ""
        for token in self.runner.stream_chat(message, thread_id):
            accumulated += token
            yield accumulated

    def launch(self):
        with gr.Blocks(title="📚 Personal Study Buddy") as demo:
            thread_state = gr.State(value=str(uuid.uuid4()))

            chat = gr.ChatInterface(
                fn=self.respond,
                title="📚 Personal Study Buddy",
                additional_inputs=[thread_state],
            )

            gr.ClearButton(
                [chat.chatbot, chat.textbox],
                value="🔄 New Session",
                variant="primary",
            ).click(
                fn=lambda: str(uuid.uuid4()),
                outputs=[thread_state],
            )

        demo.launch()


if __name__ == "__main__":
    StudyBuddyApp().launch()
