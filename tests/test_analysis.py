"""Unit tests for the deterministic layer.

These run with no API key and no network. That is the point: the parts of the
dossier a recruiter is most likely to challenge are the parts covered by
ordinary tests.

`today` is pinned in every test so that "present" roles do not make results
drift as the calendar moves.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.analysis import (
    GAP_THRESHOLD_MONTHS,
    build_timeline,
    derive_risk_flags,
    sort_flags,
)
from app.schemas import CandidateProfile, Position, RiskFlag

TODAY = date(2026, 8, 1)


def prof(*positions: Position, **kwargs) -> CandidateProfile:
    return CandidateProfile(positions=list(positions), **kwargs)


def pos(company: str, title: str, start: str, end: str, **kwargs) -> Position:
    return Position(company=company, title=title, start=start, end=end, **kwargs)


# --- total experience -----------------------------------------------------


def test_total_experience_sums_sequential_roles():
    t = build_timeline(
        prof(
            pos("A", "Engineer", "2019-01", "2021-01"),  # 24 months
            pos("B", "Engineer", "2021-01", "2023-01"),  # 24 months
        ),
        today=TODAY,
    )
    assert t.total_experience_months == 48
    assert t.total_experience_years == 4.0


def test_overlapping_roles_are_not_double_counted():
    """A consulting gig alongside a day job is not extra career experience."""
    t = build_timeline(
        prof(
            pos("Day Job", "Engineer", "2020-01", "2024-01"),          # 48 months
            pos("Side Consulting", "Consultant", "2021-01", "2022-01"),  # inside the above
        ),
        today=TODAY,
    )
    assert t.total_experience_months == 48


def test_partially_overlapping_roles_merge():
    t = build_timeline(
        prof(
            pos("A", "Engineer", "2020-01", "2022-06"),
            pos("B", "Engineer", "2022-01", "2023-01"),
        ),
        today=TODAY,
    )
    # Jan 2020 -> Jan 2023 is 36 months, not 30 + 12.
    assert t.total_experience_months == 36


def test_present_resolves_to_today():
    t = build_timeline(prof(pos("A", "Engineer", "2026-02", "present")), today=TODAY)
    assert t.total_experience_months == 6


# --- gaps -----------------------------------------------------------------


def test_gap_above_threshold_is_reported():
    t = build_timeline(
        prof(
            pos("A", "Engineer", "2019-07", "2020-08"),
            pos("B", "Engineer", "2021-03", "2023-12"),  # 7-month gap
        ),
        today=TODAY,
    )
    assert len(t.gaps) == 1
    start, end, months = t.gaps[0]
    assert months == 7
    assert (start, end) == (date(2020, 8, 1), date(2021, 3, 1))


def test_short_gap_is_ignored_as_notice_period_slack():
    t = build_timeline(
        prof(
            pos("A", "Engineer", "2020-01", "2022-01"),
            pos("B", "Engineer", "2022-03", "2024-01"),  # 2 months
        ),
        today=TODAY,
    )
    assert t.gaps == []


def test_gap_threshold_is_inclusive():
    t = build_timeline(
        prof(
            pos("A", "Engineer", "2020-01", "2022-01"),
            pos("B", "Engineer", "2022-05", "2024-01"),  # exactly 4 months
        ),
        today=TODAY,
    )
    assert len(t.gaps) == 1
    assert t.gaps[0][2] == GAP_THRESHOLD_MONTHS


def test_overlapping_roles_cannot_create_a_phantom_gap():
    """Regression guard: gaps are computed on merged blocks, not raw pairs."""
    t = build_timeline(
        prof(
            pos("Long Role", "Engineer", "2018-01", "2024-01"),
            pos("Short Overlap", "Advisor", "2019-01", "2019-06"),
        ),
        today=TODAY,
    )
    assert t.gaps == []


# --- tenure ---------------------------------------------------------------


def test_short_tenures_flagged_only_when_repeated():
    one_short = prof(
        pos("A", "Engineer", "2019-01", "2019-09"),   # 8 months
        pos("B", "Engineer", "2019-09", "2024-01"),
    )
    flags = derive_risk_flags(one_short, build_timeline(one_short, today=TODAY))
    assert not [f for f in flags if f.kind == "short_tenure"]

    two_short = prof(
        pos("A", "Engineer", "2019-01", "2019-09"),   # 8 months
        pos("B", "Engineer", "2019-10", "2020-06"),   # 8 months
        pos("C", "Engineer", "2020-08", "2024-01"),
    )
    flags = derive_risk_flags(two_short, build_timeline(two_short, today=TODAY))
    assert [f for f in flags if f.kind == "short_tenure"]


def test_current_short_role_is_not_job_hopping():
    p = prof(
        pos("A", "Engineer", "2019-01", "2019-08"),
        pos("B", "Engineer", "2026-05", "present"),  # 3 months, but ongoing
    )
    t = build_timeline(p, today=TODAY)
    assert len(t.short_tenures) == 1  # only the historical one


def test_internships_are_not_short_tenure():
    p = prof(
        pos("A", "Intern", "2018-05", "2018-08", employment_type="internship"),
        pos("B", "Intern", "2018-09", "2018-12", employment_type="internship"),
        pos("C", "Engineer", "2019-01", "2024-01"),
    )
    t = build_timeline(p, today=TODAY)
    assert t.short_tenures == []


# --- title inflation ------------------------------------------------------


def test_early_head_of_title_without_team_size_is_flagged():
    p = prof(
        pos("Startup", "Head of AI Engineering", "2023-01", "present"),
        pos("Prior", "Engineer", "2020-01", "2023-01"),
    )
    flags = derive_risk_flags(p, build_timeline(p, today=TODAY))
    assert [f for f in flags if f.kind == "title_inflation"]


def test_stated_team_size_suppresses_the_title_flag():
    """An explicit team size is exactly the evidence the flag asks for."""
    p = prof(
        pos("Startup", "Head of AI Engineering", "2023-01", "present", team_size=14),
        pos("Prior", "Engineer", "2020-01", "2023-01"),
    )
    flags = derive_risk_flags(p, build_timeline(p, today=TODAY))
    assert not [f for f in flags if f.kind == "title_inflation"]


def test_long_career_before_senior_title_is_not_flagged():
    p = prof(
        pos("Now", "Head of Engineering", "2024-01", "present"),
        pos("Before", "Engineer", "2010-01", "2024-01"),
    )
    flags = derive_risk_flags(p, build_timeline(p, today=TODAY))
    assert not [f for f in flags if f.kind == "title_inflation"]


# --- robustness -----------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "unknown", "2021", "2021-13", "Jan 2021", "n/a"])
def test_unparseable_dates_do_not_crash(bad: str):
    p = prof(pos("A", "Engineer", bad, "2024-01"))
    t = build_timeline(p, today=TODAY)
    # "2021" is recoverable as 2021-01; the rest are not. Either way: no crash.
    assert len(t.intervals) + len(t.unparseable) == 1


def test_reversed_dates_are_treated_as_unparseable():
    p = prof(pos("A", "Engineer", "2024-01", "2020-01"))
    t = build_timeline(p, today=TODAY)
    assert len(t.unparseable) == 1
    assert t.total_experience_months == 0


def test_empty_profile_is_safe():
    t = build_timeline(prof(), today=TODAY)
    assert t.total_experience_months == 0
    assert t.gaps == []
    assert derive_risk_flags(prof(), t) == []


def test_long_notice_period_is_flagged():
    p = prof(pos("A", "Engineer", "2020-01", "present"), notice_period_days=90)
    flags = derive_risk_flags(p, build_timeline(p, today=TODAY))
    assert [f for f in flags if f.kind == "logistics"]


def test_flags_sort_high_severity_first():
    flags = [
        RiskFlag(kind="logistics", severity="low", summary="c"),
        RiskFlag(kind="logistics", severity="high", summary="a"),
        RiskFlag(kind="logistics", severity="medium", summary="b"),
    ]
    assert [f.summary for f in sort_flags(flags)] == ["a", "b", "c"]


# --- the sample CV, end to end through the deterministic layer ------------


def test_sample_profile_produces_the_expected_flags():
    """Mirrors data/samples/cv_arjun_menon.txt, hand-transcribed.

    If the extraction pass is working, the real pipeline should reach the same
    conclusions this test asserts.
    """
    p = prof(
        pos("Fintrail Technologies", "Head of AI Engineering", "2024-01", "present"),
        pos("Razorpay", "Senior Machine Learning Engineer", "2021-03", "2023-12"),
        pos("Mu Sigma", "Machine Learning Engineer", "2019-07", "2020-08"),
        pos("Tata Consultancy Services", "Data Analyst", "2017-06", "2019-06"),
        notice_period_days=90,
    )
    t = build_timeline(p, today=TODAY)
    kinds = {f.kind for f in derive_risk_flags(p, t)}

    assert "employment_gap" in kinds     # Aug 2020 -> Mar 2021
    assert "title_inflation" in kinds    # Head of AI at ~6.5 years, no team size
    assert "logistics" in kinds          # 90-day notice
    assert "short_tenure" not in kinds   # tenures are all healthy
