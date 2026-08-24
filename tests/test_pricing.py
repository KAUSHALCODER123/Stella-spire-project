"""Cost reporting, including the cases where the honest answer is "unknown"."""

from __future__ import annotations

import pytest

from app.config import MODEL_CHOICES
from app.pricing import USD_PER_MTOK, is_free, price_run, price_usage


class FakeUsage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


def test_a_free_model_costs_exactly_nothing():
    c = price_run(500_000, 40_000, "dots-studio/dots-3-note-preview:free")
    assert c.free and c.usd == 0.0
    assert c.inr_display == "₹0"


def test_an_unpriced_model_reports_unknown_rather_than_guessing():
    """A confident wrong figure gets repeated to a client. A dash does not."""
    c = price_run(500_000, 40_000, "gpt-5.5")
    assert not c.known
    assert c.usd is None and c.inr is None
    assert c.inr_display == "—"


def test_a_priced_model_computes_from_the_published_rate():
    c = price_run(1_000_000, 1_000_000, "gpt-4o")
    assert c.usd == pytest.approx(12.50)
    assert c.inr == pytest.approx(12.50 * 88.0)
    assert c.inr_display.startswith("₹1,100")


def test_a_tiny_run_does_not_render_as_zero():
    """₹0.00 and 'free' must not look the same to someone scanning the page."""
    c = price_run(100, 10, "gpt-4o-mini")
    assert not c.free
    assert c.inr_display == "under ₹1"


def test_zero_tokens_on_a_paid_model_is_still_not_free():
    c = price_run(0, 0, "gpt-4o")
    assert c.usd == 0.0 and not c.free


def test_price_usage_reads_the_pipeline_object():
    assert price_usage(FakeUsage(1_000_000, 0), "gpt-4o").usd == pytest.approx(2.50)


def test_price_usage_survives_a_usage_with_nothing_recorded():
    class Empty:
        pass

    assert price_usage(Empty(), "gpt-4o").usd == 0.0


@pytest.mark.parametrize("model", [m[0] for m in MODEL_CHOICES if m[0].endswith(":free")])
def test_every_free_choice_in_the_picker_is_detected_as_free(model):
    assert is_free(model)


@pytest.mark.parametrize("model", [m[0] for m in MODEL_CHOICES if not m[0].endswith(":free")])
def test_every_paid_choice_is_either_priced_or_honestly_unpriced(model):
    """No silent third state: it is priced, or it renders as a dash."""
    c = price_run(1000, 100, model)
    assert (model in USD_PER_MTOK) == c.known
    assert c.known or c.inr_display == "—"
