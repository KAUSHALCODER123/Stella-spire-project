"""Error classification.

Regression origin: OpenAI returns `insufficient_quota` as HTTP 429, which the
SDK surfaces as RateLimitError. The pipeline retried it twice and then
reported "the job description could not be parsed" -- so an account with no
credit looked like a broken parser, and sent the reader debugging the wrong
layer entirely.
"""

from __future__ import annotations

import pytest

from app.extract.llm import LLMError, _classify_rate_limit


class FakeRateLimit(Exception):
    pass


# --- permanent vs retryable ------------------------------------------------


@pytest.mark.parametrize("message", [
    "Error code: 429 - {'error': {'message': 'You have no credits remaining.', 'code': 'insufficient_quota'}}",
    "You have no credits remaining. Add credits to continue using the API.",
    "Your credit balance is too low to access the API",
    "You exceeded your current quota, please check your plan and billing details",
])
def test_billing_failures_are_permanent(message):
    err = _classify_rate_limit(FakeRateLimit(message))
    assert err is not None, "billing failure was treated as retryable"
    assert err.kind == "quota"


@pytest.mark.parametrize("message", [
    "Rate limit reached for gpt-4o in organization org-x. Please try again in 20s.",
    "Requests to the API have exceeded the rate limit for this minute.",
    "429 Too Many Requests",
])
def test_ordinary_rate_limits_stay_retryable(message):
    assert _classify_rate_limit(FakeRateLimit(message)) is None


def test_classification_is_case_insensitive():
    assert _classify_rate_limit(FakeRateLimit("NO CREDITS REMAINING")) is not None


# --- message quality -------------------------------------------------------


def test_quota_message_says_what_to_do():
    err = _classify_rate_limit(FakeRateLimit("insufficient_quota"))
    text = str(err).lower()
    assert "credit" in text
    assert "billing" in text or "platform.openai.com" in text


def test_quota_message_does_not_blame_the_document():
    """The old message accused the job description. It is a billing problem."""
    text = str(_classify_rate_limit(FakeRateLimit("insufficient_quota"))).lower()
    for wrong in ("job description", "could not be parsed", "not a cv", "resume"):
        assert wrong not in text


# --- the error type itself -------------------------------------------------


def test_llmerror_carries_a_kind():
    assert LLMError("x").kind == "error"
    assert LLMError("x", kind="auth").kind == "auth"


def test_llmerror_is_still_an_exception():
    with pytest.raises(RuntimeError):
        raise LLMError("boom", kind="quota")
