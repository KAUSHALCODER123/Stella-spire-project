"""The eval scorer is itself measured here, offline.

A scorer nobody checked is worse than no scorer: it produces a number that
looks like evidence. These tests feed it a profile built directly from the
ground truth (which must score 100%) and then a series of single-field
mutations (each of which must be caught, in the right field). No model call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas import CandidateProfile, Position, Skill
from eval.run_eval import Scorecard, companies_match, digits, score_case, titles_match

GROUND_TRUTH = json.loads(
    (Path(__file__).resolve().parent.parent / "eval" / "ground_truth.json").read_text(encoding="utf-8")
)["cases"]

BY_ID = {c["id"]: c for c in GROUND_TRUTH}


def perfect_profile(truth: dict) -> CandidateProfile:
    """Exactly what a flawless extractor would return for this case."""
    return CandidateProfile(
        full_name=truth["full_name"],
        email=truth["email"],
        phone=truth.get("phone_digits") and "+{}".format(truth["phone_digits"]),
        notice_period_days=truth.get("notice_period_days"),
        current_ctc_lpa=truth.get("current_ctc_lpa"),
        expected_ctc_lpa=truth.get("expected_ctc_lpa"),
        positions=[Position(**p) for p in truth["positions"]],
        skills=[Skill(name=s, category="domain") for s in truth.get("skills_must_include", [])],
        extraction_notes=["Career break noted between roles."] if truth.get("expect_break_noted") else [],
    )


def run(truth: dict, profile: CandidateProfile) -> Scorecard:
    card = Scorecard()
    score_case(card, truth, profile)
    return card


def totals(card: Scorecard) -> tuple:
    return (sum(t.correct for t in card.fields.values()),
            sum(t.checked for t in card.fields.values()))


# --- the scorer must accept a correct answer -------------------------------


@pytest.mark.parametrize("case_id", list(BY_ID))
def test_a_perfect_extraction_scores_full_marks(case_id):
    """Any check that a flawless profile fails is a bug in the scorer."""
    truth = BY_ID[case_id]
    card = run(truth, perfect_profile(truth))
    correct, checked = totals(card)
    assert checked > 0
    assert correct == checked, card.case_notes


def test_every_case_is_actually_checked():
    """A case with no assertions would silently inflate the average."""
    for case_id, truth in BY_ID.items():
        _, checked = totals(run(truth, perfect_profile(truth)))
        assert checked >= 8, "{} only makes {} checks".format(case_id, checked)


# --- and it must reject a wrong one ----------------------------------------


def test_a_wrong_name_is_caught():
    truth = BY_ID["01_clean"]
    p = perfect_profile(truth)
    p.full_name = "Rahul Deshpandey"
    assert run(truth, p).fields["name"].correct == 0


def test_a_dropped_role_is_caught_as_both_count_and_company():
    truth = BY_ID["03_abbrev_dates"]
    p = perfect_profile(truth)
    p.positions = p.positions[:-1]
    card = run(truth, p)
    assert card.fields["position count"].correct == 0
    assert card.fields["company"].correct == len(truth["positions"]) - 1


def test_a_hallucinated_extra_role_is_caught():
    """Inventing a job is the failure that matters most; it must not pass."""
    truth = BY_ID["01_clean"]
    p = perfect_profile(truth)
    p.positions.append(Position(company="Imaginary Ltd", title="CFO", start="2024-01", end="present"))
    assert run(truth, p).fields["position count"].correct == 0


def test_an_off_by_one_month_start_date_is_caught():
    truth = BY_ID["01_clean"]
    p = perfect_profile(truth)
    p.positions[0].start = "2021-05"
    assert run(truth, p).fields["start date"].correct == len(truth["positions"]) - 1


def test_a_missing_key_skill_is_caught():
    truth = BY_ID["01_clean"]
    p = perfect_profile(truth)
    p.skills = [s for s in p.skills if "SAP" not in s.name]
    card = run(truth, p)
    assert card.fields["key skills"].correct == len(truth["skills_must_include"]) - 1


def test_an_unreported_career_break_is_caught():
    truth = BY_ID["06_career_break"]
    p = perfect_profile(truth)
    p.extraction_notes = []
    assert run(truth, p).fields["career break noted"].correct == 0


def test_a_null_field_is_a_miss_not_a_crash():
    truth = BY_ID["01_clean"]
    card = run(truth, CandidateProfile())
    correct, checked = totals(card)
    assert correct == 0 and checked > 0


def test_a_wrong_ctc_is_caught_but_rounding_is_forgiven():
    truth = BY_ID["01_clean"]
    p = perfect_profile(truth)
    p.current_ctc_lpa = 42.4          # 42 rendered as 42.4 is the same figure
    assert run(truth, p).fields["current_ctc_lpa"].correct == 1
    p.current_ctc_lpa = 45.0          # 45 is a different figure
    assert run(truth, p).fields["current_ctc_lpa"].correct == 0


# --- the fuzzy comparators must stay honest --------------------------------


@pytest.mark.parametrize("expected,got", [
    ("VP, Financial Reporting", "Vice President Financial Reporting"),
    ("Head of FP&A", "Head of FP&A"),
    ("Assistant Manager, Audit", "Assistant Manager - Audit"),
])
def test_title_wording_differences_are_forgiven(expected, got):
    assert titles_match(expected, got)


@pytest.mark.parametrize("expected,got", [
    ("Financial Controller", "Financial Analyst"),
    ("Head of Finance", "Head of Marketing"),
    ("Director, FP&A", "Manager, FP&A"),
])
def test_genuinely_different_titles_are_not_forgiven(expected, got):
    assert not titles_match(expected, got)


@pytest.mark.parametrize("expected,got", [
    ("Arvind Limited", "Arvind Ltd"),
    ("Deloitte India", "Deloitte"),
    ("BSR & Co.", "BSR and Co"),
])
def test_company_suffix_differences_are_forgiven(expected, got):
    assert companies_match(expected, got)


def test_different_companies_are_not_forgiven():
    assert not companies_match("Kotak Mahindra Bank", "IndusInd Bank")


def test_phone_formatting_is_ignored_but_the_number_is_not():
    assert digits("+91 98220 11447") == digits("(+91) 98220-11447") == "919822011447"
    assert digits("+91 98220 11448") != "919822011447"


# --- promotions: the case most likely to be mis-scored ---------------------


def test_promotions_at_one_employer_are_matched_to_the_right_rows():
    """Four roles, three sharing an employer -- start date must disambiguate."""
    truth = BY_ID["08_promotions"]
    p = perfect_profile(truth)
    card = run(truth, p)
    assert card.fields["title"].correct == 4, card.case_notes


def test_swapping_two_titles_within_one_employer_is_caught():
    truth = BY_ID["08_promotions"]
    p = perfect_profile(truth)
    p.positions[0].title, p.positions[1].title = p.positions[1].title, p.positions[0].title
    assert run(truth, p).fields["title"].correct < 4


@pytest.mark.parametrize("expected,got", [
    ("Chief Financial Officer", "Officer"),
    ("Financial Controller", "Controller"),
    ("Manager, Business Finance", "Manager"),
    ("Director, Financial Planning & Analysis", "Director"),
])
def test_a_truncated_title_is_not_forgiven(expected, got):
    """"Officer" sits inside "Chief Financial Officer"; that is a miss."""
    assert not titles_match(expected, got)
