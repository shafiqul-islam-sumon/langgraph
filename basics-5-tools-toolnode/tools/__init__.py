from .compound_interest import CompoundInterestTool
from .convert_currency import ConvertCurrencyTool
from .loan_payment import LoanPaymentTool
from .split_budget import SplitBudgetTool

TOOLS = [
    LoanPaymentTool(),
    CompoundInterestTool(),
    ConvertCurrencyTool(),
    SplitBudgetTool(),
]
