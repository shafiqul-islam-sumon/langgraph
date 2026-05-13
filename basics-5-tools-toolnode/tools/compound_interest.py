import os
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), "r") as f:
        return f.read().strip()


class CompoundInterestInput(BaseModel):
    principal: float
    annual_rate_pct: float
    years: int


class CompoundInterestTool(BaseTool):
    name: str = "compound_interest"
    description: str = _load_prompt("compound_interest.txt")
    args_schema: Type[BaseModel] = CompoundInterestInput

    def _run(self, principal: float, annual_rate_pct: float, years: int) -> str:
        rate = annual_rate_pct / 100
        final_amount = principal * (1 + rate) ** years
        interest_earned = final_amount - principal
        return (
            f"Final amount after {years} years: ${final_amount:.2f}. "
            f"Interest earned: ${interest_earned:.2f}."
        )

    async def _arun(self, principal: float, annual_rate_pct: float, years: int) -> str:
        return self._run(principal, annual_rate_pct, years)
