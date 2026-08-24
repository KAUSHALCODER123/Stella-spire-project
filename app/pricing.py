"""What a run actually cost, in the currency the person reading it thinks in.

Token counts are the honest unit but not a useful one: "412,000 in / 18,000
out" tells a recruiter nothing about whether to run this over a hundred CVs.

Two rules here, both about not inventing numbers:

* A model whose price is not in the table returns None rather than a guess.
  The interface then says so. A confidently wrong rupee figure is worse than
  an absent one, because it will be repeated to a client.
* Free models return exactly zero, not a notional "what this would have cost
  on GPT-4o". That comparison is interesting, but it is a different claim and
  it belongs next to a label that says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# USD per one million tokens, (input, output). Published list prices; caching
# and batch discounts make the real figure lower, so this is an upper bound.
#
# Models absent from this table are deliberately absent: the pipeline offers a
# few whose pricing I cannot state with confidence, and guessing is the one
# thing this project does not do anywhere else.
USD_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
}

# Approximate, and stale the moment it is written. Only ever used for a figure
# the interface labels "approx.", never for anything anyone is invoiced from.
INR_PER_USD = 88.0


@dataclass(frozen=True)
class RunCost:
    """None for usd/inr means "not priced", which the template must render."""

    usd: Optional[float]
    inr: Optional[float]
    free: bool
    model: str

    @property
    def known(self) -> bool:
        return self.usd is not None

    @property
    def inr_display(self) -> str:
        if self.free:
            return "₹0"
        if self.inr is None:
            return "—"
        if self.inr < 1:
            return "under ₹1"
        return "₹{:,.2f}".format(self.inr)


def is_free(model: str) -> bool:
    """OpenRouter marks its no-cost tier with a `:free` suffix."""
    return (model or "").endswith(":free")


def price_run(input_tokens: int, output_tokens: int, model: str) -> RunCost:
    if is_free(model):
        return RunCost(usd=0.0, inr=0.0, free=True, model=model)

    rates = USD_PER_MTOK.get(model)
    if rates is None:
        return RunCost(usd=None, inr=None, free=False, model=model)

    per_in, per_out = rates
    usd = (input_tokens / 1_000_000) * per_in + (output_tokens / 1_000_000) * per_out
    return RunCost(usd=usd, inr=usd * INR_PER_USD, free=False, model=model)


def price_usage(usage, model: str) -> RunCost:
    """Convenience for the Usage object the pipeline threads through."""
    return price_run(
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
        model,
    )
