import os
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), "r") as f:
        return f.read().strip()


class SplitBudgetInput(BaseModel):
    monthly_income: float
    rent_pct: float
    food_pct: float
    savings_pct: float


class SplitBudgetTool(BaseTool):
    name: str = "split_budget"
    description: str = _load_prompt("split_budget.txt")
    args_schema: Type[BaseModel] = SplitBudgetInput

    def _run(
        self,
        monthly_income: float,
        rent_pct: float,
        food_pct: float,
        savings_pct: float,
    ) -> str:
        total_pct = rent_pct + food_pct + savings_pct
        if total_pct > 100:
            return (
                f"Percentages sum to {total_pct:.1f}%, which exceeds 100%. "
                "Please adjust the values."
            )
        rent      = monthly_income * rent_pct    / 100
        food      = monthly_income * food_pct    / 100
        savings   = monthly_income * savings_pct / 100
        misc_pct  = 100 - total_pct
        misc      = monthly_income * misc_pct    / 100
        return (
            f"Monthly budget breakdown for ${monthly_income:.2f}: "
            f"Rent ${rent:.2f} ({rent_pct:.0f}%), "
            f"Food ${food:.2f} ({food_pct:.0f}%), "
            f"Savings ${savings:.2f} ({savings_pct:.0f}%), "
            f"Discretionary ${misc:.2f} ({misc_pct:.0f}%)."
        )

    async def _arun(
        self,
        monthly_income: float,
        rent_pct: float,
        food_pct: float,
        savings_pct: float,
    ) -> str:
        return self._run(monthly_income, rent_pct, food_pct, savings_pct)
