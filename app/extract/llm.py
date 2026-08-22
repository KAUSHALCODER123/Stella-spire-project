"""The three model calls that make up the pipeline.

Each one uses `client.responses.parse(text_format=<PydanticModel>)`, so the API
constrains generation to the schema and the SDK hands back a validated
instance. That removes the entire class of "the model returned prose instead of
JSON" failure without any hand-rolled parsing or regex repair.

This is the ONLY module in the project that knows which vendor we call. The
schemas, the timeline arithmetic, the prompts, the ingestion layer and the
renderer are all provider-agnostic, so swapping vendors is a change to this
file alone.

What is deliberately NOT here: anything the model does not need to decide.
Dates, gaps and tenure are computed in app/analysis.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.analysis import TimelineAnalysis
from app.config import settings
from app.extract.prompts import ASSESSMENT_SYSTEM, EXTRACTION_SYSTEM, JOB_BRIEF_SYSTEM
from app.schemas import Assessment, CandidateProfile, JobBrief
from app.toon import encode_model

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: Optional[OpenAI] = None


class LLMError(RuntimeError):
    """Raised when the model could not produce usable output.

    `kind` lets the interface say something specific and actionable instead of
    a generic failure. "quota" and "auth" in particular are account problems,
    not application problems, and telling the user they are the same thing
    sends them debugging entirely the wrong layer.
    """

    def __init__(self, message: str, kind: str = "error") -> None:
        super().__init__(message)
        self.kind = kind


def _classify_rate_limit(exc: Exception) -> Optional[LLMError]:
    """A 429 is not always a rate limit.

    OpenAI returns `insufficient_quota` as a 429, which the SDK surfaces as
    RateLimitError. That is permanent until someone adds credit, so retrying
    it just delays a failure that was never going to succeed.
    """
    text = str(exc).lower()
    if "insufficient_quota" in text or "credit balance" in text or "no credits remaining" in text:
        return LLMError(
            "This OpenAI account has no credits remaining, so the analysis could not run. "
            "Add credit at platform.openai.com/settings/organization/billing, then try again.",
            kind="quota",
        )
    if "exceeded your current quota" in text:
        return LLMError(
            "This OpenAI account has exceeded its quota. Check your plan and billing limits "
            "at platform.openai.com, then try again.",
            kind="quota",
        )
    return None


@dataclass
class Usage:
    """Token accounting, surfaced in the UI so cost per dossier is never a mystery."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    call_log: List[str] = field(default_factory=list)

    cached_tokens: int = 0

    def add(self, label: str, response) -> None:
        u = getattr(response, "usage", None)
        if u is not None:
            self.input_tokens += getattr(u, "input_tokens", 0) or 0
            self.output_tokens += getattr(u, "output_tokens", 0) or 0
            details = getattr(u, "input_tokens_details", None)
            self.cached_tokens += getattr(details, "cached_tokens", 0) or 0
        self.calls += 1
        self.call_log.append(label)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise LLMError(
                "No OPENAI_API_KEY found. Copy .env.example to .env, add your key, and restart the server.",
                kind="auth",
            )
        kwargs = {"api_key": settings.openai_api_key, "timeout": 180.0}
        if settings.openai_base_url:
            # OpenRouter, NVIDIA NIM, or anything else speaking the same
            # protocol. Everything downstream is unchanged.
            kwargs["base_url"] = settings.openai_base_url
        _client = OpenAI(**kwargs)
    return _client


def _parse(
    *,
    label: str,
    system: str,
    user: str,
    output_format: type[T],
    usage: Optional[Usage] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> T:
    """One structured call, retried on transient and malformed-output failures.

    Providers differ in how they enforce a schema. OpenAI constrains decoding,
    so the JSON is valid by construction. OpenRouter's free models mostly
    prompt for it, and intermittently emit something that does not parse -- a
    doubled opening brace was the first failure seen in practice. That is
    stochastic rather than systematic, so it is worth another attempt; the
    request is otherwise identical and the next sample is usually clean.
    """
    client = get_client()
    model = model or settings.model
    last_error: Optional[Exception] = None

    for attempt in (1, 2, 3):
        try:
            response = client.responses.parse(
                model=model,
                instructions=system,
                input=user,
                max_output_tokens=max_tokens or settings.max_tokens,
                text_format=output_format,
            )
        except openai.RateLimitError as exc:
            fatal = _classify_rate_limit(exc)
            if fatal is not None:
                # Permanent until billing changes. Fail immediately.
                log.error("%s: %s", label, fatal)
                raise fatal from exc
            last_error = exc
            log.warning("%s: rate limited on attempt %d, retrying", label, attempt)
            continue
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            last_error = exc
            log.warning("%s: transient failure on attempt %d (%s)", label, attempt, type(exc).__name__)
            continue
        except openai.AuthenticationError as exc:
            raise LLMError(
                "The OpenAI API key was rejected. Check OPENAI_API_KEY in your .env file, "
                "then restart the server and run `python -m scripts.check_setup`.",
                kind="auth",
            ) from exc
        except openai.NotFoundError as exc:
            raise LLMError(
                "The model '{}' is not available to this API key. Run "
                "`python -m scripts.check_setup` to see which models you can use.".format(model),
                kind="model",
            ) from exc
        except openai.LengthFinishReasonError as exc:
            # The schema was not finished before the token ceiling. Retrying at
            # the same ceiling would fail identically, so fail loudly instead.
            raise LLMError(
                "The response was cut off before it was complete. Raise MAX_TOKENS in .env "
                "(currently {}) and try again.".format(max_tokens or settings.max_tokens),
                kind="truncated",
            ) from exc
        except openai.APIStatusError as exc:
            raise LLMError("The API returned an error ({}): {}".format(exc.status_code, exc), kind="api") from exc
        except ValidationError as exc:
            # The provider returned something that is not the schema. Common on
            # providers that prompt for JSON rather than constrain decoding.
            last_error = exc
            log.warning("%s: malformed output on attempt %d (%s), retrying",
                        label, attempt, str(exc).splitlines()[0][:80])
            continue

        # A refusal or a truncated response comes back as HTTP 200. Guard
        # before touching the parsed output.
        if getattr(response, "status", None) == "incomplete":
            detail = getattr(response, "incomplete_details", None)
            raise LLMError(
                "The response was incomplete ({}). Try again, or raise MAX_TOKENS in .env.".format(
                    getattr(detail, "reason", "unknown")),
                kind="truncated",
            )

        if usage is not None:
            usage.add(label, response)

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            last_error = LLMError(
                "The model returned no structured output. This usually means the uploaded file "
                "is not a CV.", kind="not_a_cv",
            )
            continue
        return parsed

    if isinstance(last_error, ValidationError):
        raise LLMError(
            "The model '{}' returned output that did not match the required schema, three times "
            "running. Some providers only prompt for JSON rather than enforcing it. Try a larger "
            "model, or one whose provider supports strict structured output.".format(model),
            kind="schema",
        )
    raise LLMError(
        "The model did not respond after three attempts. This is usually a temporary network or "
        "rate-limit problem — wait a moment and try again. ({}: {})".format(label, last_error),
        kind="transient",
    )


# --------------------------------------------------------------------------


def extract_profile(cv_text: str, usage: Optional[Usage] = None, model: Optional[str] = None) -> CandidateProfile:
    user = "Extract this CV into the schema.\n\n<cv>\n{}\n</cv>".format(cv_text)
    return _parse(
        label="extract_profile",
        system=EXTRACTION_SYSTEM,
        user=user,
        output_format=CandidateProfile,
        usage=usage,
        model=model,
    )


def extract_job_brief(jd_text: str, usage: Optional[Usage] = None, model: Optional[str] = None) -> JobBrief:
    user = "Parse this job description into a structured brief.\n\n<job_description>\n{}\n</job_description>".format(jd_text)
    return _parse(
        label="extract_job_brief",
        system=JOB_BRIEF_SYSTEM,
        user=user,
        output_format=JobBrief,
        usage=usage,
        model=model,
    )


def _timeline_block(timeline: TimelineAnalysis) -> str:
    lines = [
        "Total experience (union of roles, overlaps not double counted): {} years".format(timeline.total_experience_years),
        "Average tenure per role: {} years".format(timeline.average_tenure_years),
        "Career start: {}".format(timeline.career_start.strftime("%b %Y") if timeline.career_start else "unknown"),
    ]
    if timeline.gaps:
        for start, end, months in timeline.gaps:
            lines.append(
                "Employment gap: {} months ({} to {})".format(months, start.strftime("%b %Y"), end.strftime("%b %Y"))
            )
    else:
        lines.append("Employment gaps: none over the reporting threshold")
    return "\n".join(lines)


def assess(
    *,
    profile: CandidateProfile,
    timeline: TimelineAnalysis,
    brief: JobBrief,
    cv_text: str,
    usage: Optional[Usage] = None,
    model: Optional[str] = None,
) -> Assessment:
    # Block order here is a cost decision, not a style one. OpenAI caches the
    # longest common PREFIX of a request (1024 tokens and up, roughly half
    # price on the cached part), so whatever is shared between calls has to
    # come first. Assessing twenty candidates against one role shares the
    # brief; putting it ahead of the per-candidate blocks lets all twenty hit
    # the same cached prefix instead of every call being a miss.
    user = (
        # The model is told the format explicitly. A compact encoding it has to
        # infer is a decoding risk, and the explanation costs a dozen tokens
        # against the hundreds the encoding saves.
        "The brief and profile below are TOON: indentation shows nesting, and a "
        "line like `skills[3]{{name,category}}:` introduces a table whose next 3 "
        "lines are rows of comma-separated values in that field order.\n\n"
        "<client_brief>\n{brief}\n</client_brief>\n\n"
        "<computed_timeline>\n{timeline}\n</computed_timeline>\n\n"
        "<extracted_profile>\n{profile}\n</extracted_profile>\n\n"
        "<raw_cv_text>\n{cv}\n</raw_cv_text>\n\n"
        "Write the assessment. Every 'strong' and 'partial' verdict needs a verbatim quote "
        "from the raw CV text above."
    ).format(
        # TOON rather than JSON: these two blocks are mostly uniform arrays
        # (requirements, positions, skills), which collapse into tables that
        # name their fields once instead of per row. Measured at 32% fewer
        # tokens on the real payloads. Input only -- the response stays
        # schema-constrained JSON.
        brief=encode_model(brief),
        timeline=_timeline_block(timeline),
        profile=encode_model(profile),
        cv=cv_text,
    )
    return _parse(
        label="assess",
        system=ASSESSMENT_SYSTEM,
        user=user,
        output_format=Assessment,
        usage=usage,
        model=model,
    )
