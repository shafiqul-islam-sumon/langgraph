from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config


class GeminiLLM:
    """Wraps ChatGoogleGenerativeAI with settings loaded from Config.

    Instantiate once and call get_llm() to retrieve the underlying LLM object.
    Keeping instantiation here avoids scattering ChatGoogleGenerativeAI calls
    across the codebase.
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_retries=Config.MAX_RETRIES,
        )

    def get_llm(self):
        return self.llm
