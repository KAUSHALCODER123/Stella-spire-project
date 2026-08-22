"""Locate each evidence quote inside the source CV.

The prompt instructs the model to quote verbatim. Nothing was checking. This
module checks: every quote in the assessment is searched for in the raw CV
text, and one of three things is recorded.

    exact       found character-for-character
    normalised  found once whitespace and quote marks are regularised, which
                is the common case -- PDF extraction inserts line breaks the
                model tidies up when it quotes
    not_found   the quote is not in the document

A `not_found` is the honest name for a fabricated quote, and the interface
shows it as UNVERIFIED rather than hiding it. This turns the product's central
promise from an instruction in a prompt into a measurement.

Offsets are returned against the ORIGINAL text so the source pane can
highlight the real span, not a normalised copy of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# A quote shorter than this matches too easily to mean anything.
MIN_QUOTE_CHARS = 12

# Below this similarity we call it not found rather than guess at a span.
FUZZY_THRESHOLD = 0.82

_SMART = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}


@dataclass
class Trace:
    """Where one quote came from."""

    quote: str
    status: str                      # "exact" | "normalised" | "not_found"
    start: Optional[int] = None      # offsets into the ORIGINAL source text
    end: Optional[int] = None
    similarity: float = 0.0
    source: str = "cv"               # "cv" | "brief" | "none"

    @property
    def verified(self) -> bool:
        return self.status in ("exact", "normalised")


def _fold(text: str) -> Tuple[str, List[int]]:
    """Lowercase, regularise punctuation, collapse whitespace.

    Returns the folded string plus a map from each folded character back to
    its index in the original, so a match can be reported against the source.
    """
    out: List[str] = []
    index: List[int] = []
    prev_space = True  # leading whitespace is dropped

    for i, ch in enumerate(text):
        ch = _SMART.get(ch, ch)
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            index.append(i)
            prev_space = True
            continue
        prev_space = False
        out.append(ch.lower())
        index.append(i)

    return "".join(out), index


def find_quote(quote: Optional[str], source: str) -> Optional[Trace]:
    """Locate one quote in the source text."""
    if not quote:
        return None

    cleaned = quote.strip().strip('"“”').strip()
    if len(cleaned) < MIN_QUOTE_CHARS:
        # Too short to verify meaningfully. Report it rather than pretend.
        return Trace(quote=cleaned, status="not_found", similarity=0.0)

    # 1. exact, as written
    at = source.find(cleaned)
    if at != -1:
        return Trace(quote=cleaned, status="exact", start=at, end=at + len(cleaned), similarity=1.0)

    # 2. normalised
    folded_src, index = _fold(source)
    folded_q, _ = _fold(cleaned)
    if not folded_q:
        return Trace(quote=cleaned, status="not_found")

    at = folded_src.find(folded_q)
    if at != -1:
        return Trace(
            quote=cleaned,
            status="normalised",
            start=index[at],
            end=index[min(at + len(folded_q) - 1, len(index) - 1)] + 1,
            similarity=1.0,
        )

    # 3. near miss -- the model paraphrased slightly. Locate the best window so
    #    the recruiter can still see roughly where it came from, but do not
    #    call it verified.
    matcher = SequenceMatcher(None, folded_src, folded_q, autojunk=False)
    block = matcher.find_longest_match(0, len(folded_src), 0, len(folded_q))
    if block.size >= max(MIN_QUOTE_CHARS, len(folded_q) * 0.5):
        window_start = max(0, block.a - block.b)
        window_end = min(len(folded_src), window_start + len(folded_q))
        ratio = SequenceMatcher(None, folded_src[window_start:window_end], folded_q).ratio()
        if ratio >= FUZZY_THRESHOLD:
            return Trace(
                quote=cleaned,
                status="normalised",
                start=index[window_start],
                end=index[min(window_end - 1, len(index) - 1)] + 1,
                similarity=round(ratio, 3),
            )
        return Trace(quote=cleaned, status="not_found", similarity=round(ratio, 3))

    return Trace(quote=cleaned, status="not_found", similarity=0.0)


@dataclass
class VerificationReport:
    traces: List[Trace]
    # Keyed by the evidence string rather than by index: risk flags are later
    # merged with the computed ones, which would break any positional mapping.
    by_quote: Dict[str, int] = field(default_factory=dict)

    def trace_for(self, quote: Optional[str]) -> Optional[Trace]:
        if not quote:
            return None
        idx = self.by_quote.get(quote)
        return self.traces[idx] if idx is not None else None

    @property
    def total(self) -> int:
        return len(self.traces)

    @property
    def verified(self) -> int:
        return sum(1 for t in self.traces if t.verified)

    @property
    def unverified(self) -> List[Trace]:
        return [t for t in self.traces if not t.verified]

    @property
    def rate(self) -> float:
        return self.verified / self.total if self.total else 1.0

    def spans(self) -> List[Tuple[int, int, int]]:
        """Non-overlapping CV spans as (start, end, trace_index), earliest first.

        Overlaps are dropped rather than merged: two quotes covering the same
        line would otherwise produce nested marks the renderer cannot express.
        """
        found = sorted(
            ((t.start, t.end, i) for i, t in enumerate(self.traces)
             if t.start is not None and t.end is not None and t.source == "cv"),
            key=lambda s: (s[0], -(s[1] - s[0])),
        )
        kept: List[Tuple[int, int, int]] = []
        for start, end, idx in found:
            if kept and start < kept[-1][1]:
                continue
            kept.append((start, end, idx))
        return kept


def verify_assessment(assessment, cv_text: str, brief_text: str = "") -> VerificationReport:
    """Trace every model-produced quote back to a source document.

    The CV is searched first, then the client brief -- the
    jd_market_impossibility flag quotes the brief rather than the candidate,
    and calling that unverified would be wrong.

    Only the model's own quotes are checked. Computed flags cite calculated
    dates rather than document text, so running them through here would
    report false failures.
    """
    traces: List[Trace] = []
    by_quote: Dict[str, int] = {}

    def _add(quote: Optional[str]) -> None:
        if not quote or quote in by_quote:
            return
        trace = find_quote(quote, cv_text)
        if trace is None:
            return
        if not trace.verified and brief_text:
            from_brief = find_quote(quote, brief_text)
            if from_brief is not None and from_brief.verified:
                from_brief.source = "brief"
                trace = from_brief
        traces.append(trace)
        by_quote[quote] = len(traces) - 1

    for match in assessment.requirement_matches:
        _add(match.evidence)
    for flag in assessment.risk_flags:
        _add(flag.evidence)

    return VerificationReport(traces=traces, by_quote=by_quote)
