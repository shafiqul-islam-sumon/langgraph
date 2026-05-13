import os
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

_RATES_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "BDT": 0.0091,
    "CAD": 0.74,
    "AUD": 0.65,
    "INR": 0.012,
}


def _load_prompt(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), "r") as f:
        return f.read().strip()


class ConvertCurrencyInput(BaseModel):
    amount: float
    from_currency: str
    to_currency: str


class ConvertCurrencyTool(BaseTool):
    name: str = "convert_currency"
    description: str = _load_prompt("convert_currency.txt")
    args_schema: Type[BaseModel] = ConvertCurrencyInput

    def _run(self, amount: float, from_currency: str, to_currency: str) -> str:
        src = from_currency.upper()
        dst = to_currency.upper()
        if src not in _RATES_TO_USD:
            return f"Unsupported currency: {src}. Supported: {', '.join(_RATES_TO_USD)}."
        if dst not in _RATES_TO_USD:
            return f"Unsupported currency: {dst}. Supported: {', '.join(_RATES_TO_USD)}."
        # convert via USD as intermediate
        amount_in_usd = amount * _RATES_TO_USD[src]
        converted = amount_in_usd / _RATES_TO_USD[dst]
        return f"{amount:.2f} {src} = {converted:.2f} {dst}."

    async def _arun(self, amount: float, from_currency: str, to_currency: str) -> str:
        return self._run(amount, from_currency, to_currency)
