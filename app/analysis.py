"""Deterministic analysis over an extracted CandidateProfile.

Everything in this module is plain date arithmetic and string matching. No LLM
call is made here, and that is the whole point: a recruiter who disputes a
number in the dossier can be walked through the exact rule that produced it.

The LLM is only trusted with judgement that genuinely requires reading
comprehension. Arithmetic is not that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

from app.schemas import CandidateProfile, Position, RiskFlag

# A gap shorter than this is normal notice-period / joining slack in the Indian
# market and is not worth a recruiter's attention.
GAP_THRESHOLD_MONTHS = 4
SHORT_TENURE_MONTHS = 13

_SENIORITY_LADDER: List[Tuple[int, Tuple[str, ...]]] = [
    (5, ("chief", "cto", "cio", "ceo", "founder", "vp ", "vice president", "president")),
    (4, ("head of", "director", "general manager")),
    (3, ("principal", "staff", "architect", "senior manager", "group manager")),
    (2, ("lead", "manager", "senior", "sr.", "sr ")),
    (1, ("engineer", "analyst", "developer", "associate", "consultant", "specialist")),
    (0, ("intern", "trainee", "junior", "jr.", "graduate")),
]


def _parse_month(value: str, today: Optional[date] = None) -> Optional[date]:
    """'2021-07' -> date(2021, 7, 1). 'present' -> today. Anything else -> None."""
    today = today or date.today()
    if not value:
        return None
    v = value.strip().lower()
    if v in {"present", "current", "now", "ongoing"}:
        return date(today.year, today.month, 1)
    m = re.match(r"^(\d{4})-(\d{2})$", v)
    if not m:
        m = re.match(r"^(\d{4})$", v)
        if m:
            return date(int(m.group(1)), 1, 1)
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return None
    return date(year, month, 1)


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _seniority_rank(title: str) -> int:
    t = " " + title.lower().strip() + " "
    for rank, keywords in _SENIORITY_LADDER:
        if any(k in t for k in keywords):
            return rank
    return 1


@dataclass
class Interval:
    start: date
    end: date
    position: Position


@dataclass
class TimelineAnalysis:
    """Everything derivable from the dates on the CV."""

    intervals: List[Interval] = field(default_factory=list)
    unparseable: List[Position] = field(default_factory=list)

    total_experience_months: int = 0
    average_tenure_months: float = 0.0
    career_start: Optional[date] = None
    gaps: List[Tuple[date, date, int]] = field(default_factory=list)
    short_tenures: List[Tuple[Position, int]] = field(default_factory=list)

    @property
    def total_experience_years(self) -> float:
        return round(self.total_experience_months / 12, 1)

    @property
    def average_tenure_years(self) -> float:
        return round(self.average_tenure_months / 12, 1)


def build_timeline(profile: CandidateProfile, today: Optional[date] = None) -> TimelineAnalysis:
    today = today or date.today()
    result = TimelineAnalysis()

    for pos in profile.positions:
        start = _parse_month(pos.start, today)
        end = _parse_month(pos.end, today)
        if start is None or end is None or end < start:
            result.unparseable.append(pos)
            continue
        result.intervals.append(Interval(start=start, end=end, position=pos))

    if not result.intervals:
        return result

    result.intervals.sort(key=lambda i: i.start)
    result.career_start = result.intervals[0].start

    # Total experience is the union of the intervals, so concurrent roles
    # (a consulting gig alongside a day job) are not double counted.
    merged: List[List[date]] = []
    for iv in result.intervals:
        if merged and iv.start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], iv.end)
        else:
            merged.append([iv.start, iv.end])
    result.total_experience_months = sum(_months_between(s, e) for s, e in merged)

    # Gaps sit between the merged blocks, so overlapping roles cannot create a
    # phantom gap.
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        gap = _months_between(prev_end, next_start)
        if gap >= GAP_THRESHOLD_MONTHS:
            result.gaps.append((prev_end, next_start, gap))

    tenures = [_months_between(iv.start, iv.end) for iv in result.intervals]
    result.average_tenure_months = sum(tenures) / len(tenures)

    for iv, months in zip(result.intervals, tenures):
        # An ongoing role being short is not job-hopping, it is just current.
        is_current = iv.end >= date(today.year, today.month, 1)
        if months < SHORT_TENURE_MONTHS and not is_current and iv.position.employment_type != "internship":
            result.short_tenures.append((iv.position, months))

    return result


def _fmt(d: date) -> str:
    return d.strftime("%b %Y")


def derive_risk_flags(profile: CandidateProfile, timeline: TimelineAnalysis) -> List[RiskFlag]:
    """Risk flags that follow from arithmetic alone.

    The LLM produces a separate set of judgement-based flags; the two lists are
    concatenated downstream. Every flag here cites the dates it came from so a
    recruiter can check it against the CV in seconds.
    """
    flags: List[RiskFlag] = []

    for start, end, months in timeline.gaps:
        severity = "high" if months >= 12 else "medium" if months >= 7 else "low"
        flags.append(
            RiskFlag(
                kind="employment_gap",
                severity=severity,
                summary=(
                    str(months) + "-month gap between roles (" + _fmt(start) + " to " + _fmt(end) + "). "
                    "Not explained on the CV."
                ),
                evidence="Previous role ended " + _fmt(start) + "; next role began " + _fmt(end) + ".",
            )
        )

    if len(timeline.short_tenures) >= 2:
        detail = "; ".join(p.title + " at " + p.company + " (" + str(m) + " months)" for p, m in timeline.short_tenures[:3])
        flags.append(
            RiskFlag(
                kind="short_tenure",
                severity="high" if len(timeline.short_tenures) >= 3 else "medium",
                summary=(
                    str(len(timeline.short_tenures)) + " roles held under " + str(SHORT_TENURE_MONTHS)
                    + " months. Average tenure across all roles is " + str(timeline.average_tenure_years) + " years."
                ),
                evidence=detail,
            )
        )

    # Title inflation: a senior-management title reached unusually early, with
    # no team size stated anywhere on the CV to back it up.
    if timeline.intervals and timeline.career_start:
        top = max(timeline.intervals, key=lambda i: (_seniority_rank(i.position.title), i.start))
        rank = _seniority_rank(top.position.title)
        years_at_start = _months_between(timeline.career_start, top.start) / 12
        expected = {5: 12.0, 4: 9.0, 3: 7.0}.get(rank)
        has_team_evidence = any(p.team_size for p in profile.positions)
        if expected and years_at_start < expected and not has_team_evidence:
            flags.append(
                RiskFlag(
                    kind="title_inflation",
                    severity="medium" if years_at_start >= expected * 0.6 else "high",
                    summary=(
                        "Reached '" + top.position.title + "' after " + format(years_at_start, ".1f")
                        + " years of experience (typical for this level: " + format(expected, ".0f")
                        + "+). No team size or reporting line stated on the CV."
                    ),
                    evidence=(
                        "Career start " + _fmt(timeline.career_start) + "; '" + top.position.title
                        + "' began " + _fmt(top.start) + "."
                    ),
                )
            )

    if timeline.unparseable:
        titles = ", ".join(p.title + " at " + p.company for p in timeline.unparseable[:3])
        flags.append(
            RiskFlag(
                kind="employment_gap",
                severity="low",
                summary=(
                    str(len(timeline.unparseable))
                    + " role(s) had dates that could not be read. Timeline may be incomplete."
                ),
                evidence=titles,
            )
        )

    if profile.notice_period_days and profile.notice_period_days >= 90:
        flags.append(
            RiskFlag(
                kind="logistics",
                severity="medium",
                summary=(
                    str(profile.notice_period_days)
                    + "-day notice period. Long exposure to counter-offers before joining."
                ),
                evidence="Stated notice period: " + str(profile.notice_period_days) + " days.",
            )
        )

    return flags


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def sort_flags(flags: List[RiskFlag]) -> List[RiskFlag]:
    return sorted(flags, key=lambda f: _SEVERITY_ORDER.get(f.severity, 3))
