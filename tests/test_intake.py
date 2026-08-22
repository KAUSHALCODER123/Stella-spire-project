"""Structured intake and the free constraint gate.

Nothing here touches a model. That is the point: every rule in this file
eliminates a pair for zero tokens, and the numbers behind each decision are
reported so a recruiter can disagree with them.
"""

from __future__ import annotations

import pytest

from app.intake import (
    CandidatePreferences,
    RoleConstraints,
    brief_from_constraints,
    check_constraints,
    preferences_from_profile,
    role_text,
)


def prefs(**kw) -> CandidatePreferences:
    return CandidatePreferences(**kw)


def role(**kw) -> RoleConstraints:
    kw.setdefault("role_title", "Engineer")
    return RoleConstraints(**kw)


# --- a brief without a model call -----------------------------------------


def test_brief_is_built_from_the_form_with_no_model_call():
    rc = role(role_title="GenAI Platform Lead", client_name="GCC",
              must_have_skills=["Kubernetes", "RAG systems"],
              nice_to_have_skills=["Vector databases"],
              min_years=8, domain="Insurance")
    brief = brief_from_constraints(rc)

    assert brief.role_title == "GenAI Platform Lead"
    assert brief.stated_min_years == 8
    musts = [r.text for r in brief.requirements if r.kind == "must_have"]
    nice = [r.text for r in brief.requirements if r.kind == "nice_to_have"]
    assert "Kubernetes" in musts and "RAG systems" in musts
    assert "Insurance domain experience" in musts
    assert "8+ years of relevant experience" in musts
    assert nice == ["Vector databases"]


def test_blank_skills_are_dropped():
    brief = brief_from_constraints(role(must_have_skills=["", "  ", "Python"]))
    assert [r.text for r in brief.requirements] == ["Python"]


def test_compensation_note_renders_a_band():
    brief = brief_from_constraints(role(ctc_min_lpa=40, ctc_max_lpa=60))
    assert "40-60" in brief.compensation_note


def test_role_text_is_readable_and_covers_the_answers():
    text = role_text(role(role_title="VP Finance", client_name="NBFC", location="Mumbai",
                          ctc_min_lpa=60, ctc_max_lpa=90, must_have_skills=["IFRS"]))
    for expected in ["VP Finance", "NBFC", "Mumbai", "IFRS", "90"]:
        assert expected in text


# --- compensation ----------------------------------------------------------


def test_a_hard_salary_floor_above_the_band_blocks():
    c = check_constraints(prefs(min_acceptable_ctc_lpa=90), role(ctc_max_lpa=60))
    assert c.blocked
    assert "90" in c.blocks[0] and "60" in c.blocks[0]


def test_an_expectation_above_the_band_only_warns():
    """An ask is negotiable; a stated floor is not."""
    c = check_constraints(prefs(expected_ctc_lpa=90), role(ctc_max_lpa=60))
    assert not c.blocked
    assert c.warnings


def test_salary_within_band_is_clean():
    c = check_constraints(prefs(min_acceptable_ctc_lpa=50), role(ctc_min_lpa=40, ctc_max_lpa=60))
    assert not c.blocked and not c.warnings


def test_missing_salary_information_never_blocks():
    assert not check_constraints(prefs(), role()).blocked
    assert not check_constraints(prefs(min_acceptable_ctc_lpa=90), role()).blocked


# --- notice period ---------------------------------------------------------


def test_a_wildly_long_notice_blocks():
    c = check_constraints(prefs(notice_period_days=120), role(max_notice_days=30))
    assert c.blocked


def test_a_slightly_long_notice_only_warns():
    """Thirty days over is the kind of thing recruiters negotiate every week."""
    c = check_constraints(prefs(notice_period_days=90), role(max_notice_days=60))
    assert not c.blocked and c.warnings


def test_notice_inside_the_limit_is_clean():
    assert not check_constraints(prefs(notice_period_days=30), role(max_notice_days=60)).warnings


# --- location and work mode ------------------------------------------------


def test_wrong_city_without_relocation_blocks():
    c = check_constraints(prefs(current_location="Chennai", open_to_relocate=False),
                          role(location="Bengaluru"))
    assert c.blocked


def test_wrong_city_with_relocation_only_warns():
    c = check_constraints(prefs(current_location="Chennai", open_to_relocate=True),
                          role(location="Bengaluru"))
    assert not c.blocked and c.warnings


def test_remote_role_survives_a_location_mismatch():
    c = check_constraints(prefs(current_location="Chennai"), role(location="Bengaluru", work_mode="remote"))
    assert not c.blocked


def test_location_matching_is_forgiving_about_wording():
    c = check_constraints(prefs(current_location="Bengaluru"), role(location="Bengaluru (hybrid, 3 days)"))
    assert not c.blocked


def test_preferred_location_counts_even_if_not_current():
    c = check_constraints(prefs(current_location="Pune", preferred_locations=["Bengaluru"]),
                          role(location="Bengaluru"))
    assert not c.blocked


def test_remote_candidate_against_onsite_role_blocks():
    c = check_constraints(prefs(work_mode="remote", current_location="Bengaluru"),
                          role(work_mode="onsite", location="Bengaluru"))
    assert c.blocked


def test_no_stated_location_never_blocks():
    assert not check_constraints(prefs(), role(location="Bengaluru")).blocked


# --- experience ------------------------------------------------------------


def test_far_too_junior_blocks():
    c = check_constraints(prefs(years_experience=3), role(min_years=12))
    assert c.blocked


def test_slightly_junior_only_warns():
    c = check_constraints(prefs(years_experience=10), role(min_years=12))
    assert not c.blocked and c.warnings


def test_heavy_over_qualification_warns_but_does_not_block():
    """Over-qualification is a retention risk, not a disqualification."""
    c = check_constraints(prefs(years_experience=25), role(min_years=4, max_years=8))
    assert not c.blocked
    assert any("retention" in w.lower() for w in c.warnings)


# --- reporting -------------------------------------------------------------


def test_every_block_quotes_the_numbers_behind_it():
    c = check_constraints(
        prefs(min_acceptable_ctc_lpa=90, notice_period_days=180, years_experience=2),
        role(ctc_max_lpa=50, max_notice_days=30, min_years=10),
    )
    assert len(c.blocks) >= 3
    assert all(any(ch.isdigit() for ch in b) for b in c.blocks), c.blocks


def test_summary_prefers_a_block_over_a_warning():
    c = check_constraints(prefs(min_acceptable_ctc_lpa=90, years_experience=11), role(ctc_max_lpa=50, min_years=12))
    assert c.summary == c.blocks[0]


def test_summary_is_friendly_when_nothing_conflicts():
    assert "no declared conflicts" in check_constraints(prefs(), role()).summary.lower()


# --- falling back to the CV ------------------------------------------------


def test_preferences_fall_back_to_what_the_cv_stated():
    """The ad-hoc upload path gets free filtering too."""
    from app.analysis import build_timeline
    from tests.fixtures import sample_profile

    profile = sample_profile()
    p = preferences_from_profile(profile, build_timeline(profile))
    assert p.notice_period_days == 90
    assert p.expected_ctc_lpa == 90.0
    assert p.current_location == "Bengaluru, Karnataka"
    assert p.years_experience and p.years_experience > 0


def test_cv_derived_preferences_still_drive_the_gate():
    from app.analysis import build_timeline
    from tests.fixtures import sample_profile

    profile = sample_profile()
    p = preferences_from_profile(profile, build_timeline(profile))
    # The CV asks 90 LPA; this role tops out at 45.
    c = check_constraints(p, role(ctc_max_lpa=45))
    assert c.warnings or c.blocked


def test_an_empty_profile_produces_a_gate_that_blocks_nothing():
    from app.schemas import CandidateProfile
    p = preferences_from_profile(CandidateProfile())
    assert not check_constraints(p, role(ctc_max_lpa=10, min_years=30, location="Mars")).blocked
