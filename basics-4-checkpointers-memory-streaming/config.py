import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class Config:
    MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
    TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", 0.7))
    MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", 2))
