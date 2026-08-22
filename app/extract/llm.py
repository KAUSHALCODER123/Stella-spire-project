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
from pydantic import BaseModel

from app.analysis import TimelineAnalysis
from app.config import settings
from app.extract.prompts import ASSESSMENT_SYSTEM, EXTRACTION_SYSTEM, JOB_BRIEF_SYSTEM
from app.schemas import Assessment, CandidateProfile, JobBrief

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: Optional[OpenAI] = None


class LLMError(RuntimeError):
    """Raised when the model could not produce usable output."""


@dataclass
class Usage:
    """Token accounting, surfaced in the UI so cost per dossier is never a mystery."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    call_log: List[str] = field(default_factory=list)

    def add(self, label: str, response) -> None:
        u = getattr(response, "usage", None)
        if u is not None:
            self.input_tokens += getattr(u, "input_tokens", 0) or 0
            self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.calls += 1
        self.call_log.append(label)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise LLMError(
                "No OPENAI_API_KEY found. Copy .env.example to .env and put your key in it, "
                "then restart the server."
            )
        _client = OpenAI(api_key=settings.openai_api_key, timeout=180.0)
    return _client


def _parse(
    *,
    label: str,
    system: str,
    user: str,
    output_format: type[T],
    usage: Optional[Usage] = None,
    max_tokens: Optional[int] = None,
) -> T:
    """One structured call, with one retry on a transient failure."""
    client = get_client()
    last_error: Optional[Exception] = None

    for attempt in (1, 2):
        try:
            response = client.responses.parse(
                model=settings.model,
                instructions=system,
                input=user,
                max_output_tokens=max_tokens or settings.max_tokens,
                text_format=output_format,
            )
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as exc:
            last_error = exc
            log.warning("%s: transient failure on attempt %d (%s)", label, attempt, type(exc).__name__)
            continue
        except openai.LengthFinishReasonError as exc:
            # The schema was not finished before the token ceiling. Retrying at
            # the same ceiling would fail identically, so fail loudly instead.
            raise LLMError(
                "{}: ran out of output tokens before completing the schema. "
                "Raise MAX_TOKENS in .env (currently {}).".format(label, max_tokens or settings.max_tokens)
            ) from exc
        except openai.APIStatusError as exc:
            raise LLMError("{} failed: {}".format(label, exc)) from exc

        # A refusal or a truncated response comes back as HTTP 200. Guard
        # before touching the parsed output.
        if getattr(response, "status", None) == "incomplete":
            detail = getattr(response, "incomplete_details", None)
            raise LLMError("{}: response incomplete ({}).".format(label, getattr(detail, "reason", "unknown")))

        if usage is not None:
            usage.add(label, response)

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            last_error = LLMError(
                "{}: the model returned no structured output. This usually means the uploaded "
                "document is not a CV.".format(label)
            )
            continue
        return parsed

    raise LLMError("{} failed after 2 attempts: {}".format(label, last_error))


# --------------------------------------------------------------------------


def extract_profile(cv_text: str, usage: Optional[Usage] = None) -> CandidateProfile:
    user = "Extract this CV into the schema.\n\n<cv>\n{}\n</cv>".format(cv_text)
    return _parse(
        label="extract_profile",
        system=EXTRACTION_SYSTEM,
        user=user,
        output_format=CandidateProfile,
        usage=usage,
    )


def extract_job_brief(jd_text: str, usage: Optional[Usage] = None) -> JobBrief:
    user = "Parse this job description into a structured brief.\n\n<job_description>\n{}\n</job_description>".format(jd_text)
    return _parse(
        label="extract_job_brief",
        system=JOB_BRIEF_SYSTEM,
        user=user,
        output_format=JobBrief,
        usage=usage,
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
) -> Assessment:
    user = (
        "<computed_timeline>\n{timeline}\n</computed_timeline>\n\n"
        "<extracted_profile>\n{profile}\n</extracted_profile>\n\n"
        "<client_brief>\n{brief}\n</client_brief>\n\n"
        "<raw_cv_text>\n{cv}\n</raw_cv_text>\n\n"
        "Write the assessment. Every 'strong' and 'partial' verdict needs a verbatim quote "
        "from the raw CV text above."
    ).format(
        timeline=_timeline_block(timeline),
        profile=profile.model_dump_json(indent=2, exclude_none=True),
        brief=brief.model_dump_json(indent=2, exclude_none=True),
        cv=cv_text,
    )
    return _parse(
        label="assess",
        system=ASSESSMENT_SYSTEM,
        user=user,
        output_format=Assessment,
        usage=usage,
    )
