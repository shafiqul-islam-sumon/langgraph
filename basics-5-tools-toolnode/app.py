import gradio as gr

from finance_runner import FinanceRunner


class FinanceApp:
    def __init__(self):
        self.runner = FinanceRunner()

    def respond(self, message: str, _history: list) -> str:
        if not message.strip():
            return ""
        return self.runner.chat(message)

    def launch(self):
        demo = gr.ChatInterface(
            fn=self.respond,
            title="💰 Personal Finance Assistant",
            description=(
                "Ask me about loan payments, savings growth, currency conversion, "
                "or budget planning. Powered by LangGraph + Google Gemini."
            ),
            examples=[
                "What's my monthly payment on a $20,000 car loan at 6% for 48 months?",
                "If I save $10,000 at 5% annual interest for 15 years, how much will I have?",
                "Convert 500 EUR to USD.",
                "I earn $5,000/month. Split my budget: 30% rent, 15% food, 25% savings.",
            ],
        )
        demo.launch()


if __name__ == "__main__":
    FinanceApp().launch()
