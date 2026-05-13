import os
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), "r") as f:
        return f.read().strip()


class LoanPaymentInput(BaseModel):
    principal: float
    annual_rate_pct: float
    months: int


class LoanPaymentTool(BaseTool):
    name: str = "calculate_loan_payment"
    description: str = _load_prompt("loan_payment.txt")
    args_schema: Type[BaseModel] = LoanPaymentInput

    def _run(self, principal: float, annual_rate_pct: float, months: int) -> str:
        if annual_rate_pct == 0:
            monthly_payment = principal / months
        else:
            r = annual_rate_pct / 100 / 12
            monthly_payment = principal * (r * (1 + r) ** months) / ((1 + r) ** months - 1)
        total_paid = monthly_payment * months
        return (
            f"Monthly payment: ${monthly_payment:.2f} for {months} months. "
            f"Total paid: ${total_paid:.2f}."
        )

    async def _arun(self, principal: float, annual_rate_pct: float, months: int) -> str:
        return self._run(principal, annual_rate_pct, months)
