from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config


class GeminiLLM:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_retries=Config.MAX_RETRIES,
        )

    def get_llm(self):
        return self.llm
