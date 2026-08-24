"""Score properties, including the degenerate inputs that divide by zero.

Every score shown in the interface is computed here, so every one of these
needs to survive an empty or malformed dossier without raising.
"""

from __future__ import annotations

import copy

import pytest

from app.extract.documents import DocumentText
from app.extract.llm import Usage
from app.pipeline import Dossier
from app.analysis import build_timeline
from app.schemas import Assessment, CandidateProfile, JobBrief, RequirementMatch, Requirement, RiskFlag
from tests.fixtures import sample_dossier


def empty_dossier(**kw) -> Dossier:
    profile = kw.pop("profile", CandidateProfile())
    brief = kw.pop("brief", JobBrief(role_title="Role"))
    assessment = kw.pop("assessment", Assessment(executive_summary="", fit_rationale=""))
    return Dossier(
        profile=profile,
        timeline=build_timeline(profile),
        brief=brief,
        assessment=assessment,
        flags=[],
        document=DocumentText(text="", page_count=0, source_format="txt"),
        usage=Usage(),
        **kw,
    )


# --- empty everything ------------------------------------------------------


def test_empty_dossier_computes_every_score_without_raising():
    d = empty_dossier()
    assert d.must_have_coverage is None
    assert d.match_counts == {"strong": 0, "partial": 0, "absent": 0, "unclear": 0}
    assert d.matched_requirements == []
    assert d.missing_requirements == []
    assert d.skill_stats["ratio"] == 0.0
    assert d.experience_match["required"] is None
    s = d.suitability
    assert 0 <= s["percent"] <= 100
    assert s["band"] in ("Strong fit", "Possible fit", "Weak fit")


def test_brief_with_no_must_haves_does_not_divide_by_zero():
    brief = JobBrief(role_title="R", requirements=[
        Requirement(text="Nice thing", kind="nice_to_have", category="technical")])
    d = empty_dossier(brief=brief)
    assert d.must_have_coverage is None
    assert d.suitability["percent"] >= 0


def test_zero_stated_min_years_is_treated_as_unspecified():
    """0 is falsy; it must not become a divide-by-zero."""
    d = empty_dossier(brief=JobBrief(role_title="R", stated_min_years=0))
    exp = d.experience_match
    assert exp["required"] is None
    assert exp["ratio"] == 1.0


# --- experience ------------------------------------------------------------


@pytest.mark.parametrize("actual_years,required,verdict", [
    (12.0, 12.0, "meets"),
    (15.0, 12.0, "meets"),
    (10.0, 12.0, "close"),     # ratio .83
    (5.0, 12.0, "short"),      # ratio .42
])
def test_experience_verdicts_on_an_unbroken_career(actual_years, required, verdict):
    """No break, so worked years and calendar span are the same number."""
    d = empty_dossier(brief=JobBrief(role_title="R", stated_min_years=required))
    d.timeline.total_experience_months = int(actual_years * 12)
    assert d.experience_match["verdict"] == verdict


def test_a_career_break_does_not_cost_someone_the_experience_bar():
    """Two people who started work on the same day are equally senior.

    Measuring only billed months quietly penalises anyone who took time out,
    which is precisely the group this firm places.
    """
    d = sample_dossier()
    d.brief.stated_min_years = 15
    exp = d.experience_match

    assert exp["months_out"] > 0, "the sample candidate should have a break"
    assert exp["actual"] < 15, "worked years alone fall short of the bar"
    assert exp["span"] >= 15, "the calendar span clears it"
    assert exp["verdict"] == "meets"


def test_both_numbers_are_reported_so_nothing_is_hidden():
    exp = sample_dossier().experience_match
    assert exp["actual"] != exp["span"]
    assert exp["months_out"] == 19


def test_experience_ratio_is_capped_at_one():
    d = sample_dossier()
    d.brief.stated_min_years = 2.0
    d.timeline.total_experience_months = 240
    assert d.experience_match["ratio"] == 1.0
    assert d.experience_match["shortfall"] == 0.0


def test_shortfall_is_never_negative():
    d = sample_dossier()
    d.brief.stated_min_years = 1.0
    assert d.experience_match["shortfall"] == 0.0


# --- suitability -----------------------------------------------------------


def test_suitability_stays_within_bounds_under_heavy_penalty():
    d = empty_dossier()
    d.flags = [RiskFlag(kind="logistics", severity="high", summary="x") for _ in range(20)]
    s = d.suitability
    assert 0.0 <= s["score"] <= 1.0
    assert s["percent"] >= 0


def test_high_flags_lower_the_score():
    clean = sample_dossier()
    flagged = copy.deepcopy(clean)
    flagged.flags = flagged.flags + [
        RiskFlag(kind="logistics", severity="high", summary="x") for _ in range(3)]
    assert flagged.suitability["score"] < clean.suitability["score"]


def test_suitability_components_are_exposed():
    """The number is only defensible if its parts are shown next to it."""
    s = sample_dossier().suitability
    for key in ("coverage", "strength", "experience", "penalty", "band", "tone"):
        assert key in s


def test_band_thresholds():
    d = sample_dossier()
    for pct, band in [(0.9, "Strong fit"), (0.5, "Possible fit"), (0.1, "Weak fit")]:
        d.brief.stated_min_years = None
        # drive coverage directly through the matches
        must = [r.text for r in d.brief.requirements if r.kind == "must_have"]
        n_strong = int(len(must) * pct)
        d.assessment.requirement_matches = (
            [RequirementMatch(requirement=t, verdict="strong", evidence="x") for t in must[:n_strong]]
            + [RequirementMatch(requirement=t, verdict="absent") for t in must[n_strong:]]
        )
        assert d.suitability["band"] == band, (pct, d.suitability)


# --- matched vs missing partition -----------------------------------------


def test_matched_and_missing_partition_every_requirement():
    d = sample_dossier()
    total = len(d.assessment.requirement_matches)
    assert len(d.matched_requirements) + len(d.missing_requirements) == total


def test_unclear_counts_as_missing_not_matched():
    """An unclear verdict is not something to tell a client is covered."""
    d = empty_dossier()
    d.assessment.requirement_matches = [RequirementMatch(requirement="x", verdict="unclear", evidence="q")]
    assert d.missing_requirements and not d.matched_requirements


# --- skills ----------------------------------------------------------------


def test_skill_stats_split_evidenced_from_listed():
    st = sample_dossier().skill_stats
    assert st["evidenced"] + st["listed_only"] == st["total"]
    assert 0.0 <= st["ratio"] <= 1.0


# --- the break must survive all the way to the client's copy ---------------


def test_effective_is_present_even_when_the_brief_states_no_minimum():
    """Templates read this on every path; a missing key renders blank."""
    exp = empty_dossier().experience_match
    assert "effective" in exp
    assert exp["effective"] == exp["actual"]


def test_effective_is_the_figure_the_verdict_was_reached_on():
    d = sample_dossier()
    d.brief.stated_min_years = 15
    exp = d.experience_match
    assert exp["effective"] == exp["span"] > exp["actual"]
    assert exp["effective"] >= exp["required"]


def test_the_client_facing_document_states_span_not_billed_months():
    """The PDF is where a career break is most costly if it is misreported.

    Showing worked months alone re-imposes the penalty the arithmetic removes,
    at the one step the candidate never sees.
    """
    from app.render.dossier import build_context, render_html

    d = sample_dossier()
    d.brief.stated_min_years = 15
    exp = d.experience_match

    ctx = build_context(d, anonymise=True)
    assert ctx["experience"]["effective"] == exp["span"]

    html = render_html(d, anonymise=True)
    assert str(exp["span"]) in html, "career span missing from the dossier"
    assert "{}-mo break".format(exp["months_out"]) in html
    assert "not seniority lost" in html, "the policy is not stated anywhere a client can read it"
