import os

from dotenv import load_dotenv

# .env lives one level up in the shared langgraph/ folder, not inside this series folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class Config:
    """Centralises all environment-driven settings.

    Class-level attributes are read once at import time. Any module that needs
    a setting imports Config directly — no instantiation required.
    """

    # Falls back to gemini-2.0-flash if GEMINI_MODEL_NAME is not set
    MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
    TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", 0.7))
    # MAX_RETRIES controls how many times the SDK retries on transient API errors
    MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", 2))
