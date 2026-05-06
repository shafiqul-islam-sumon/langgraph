from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config


class GeminiLLM:
    """Wraps ChatGoogleGenerativeAI with settings from Config.

    Keeping LLM construction here means nodes only import GeminiLLM —
    they don't need to know how the model is configured.
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_retries=Config.MAX_RETRIES,
        )

    def get_llm(self):
        """Returns the underlying LangChain Runnable LLM instance."""
        return self.llm
