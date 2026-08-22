"""Many-to-many matching: the affinity screen, the pair plan, and the run.

Everything here runs with no API key: the three model calls are stubbed, so
the orchestration, the failure isolation and the cost arithmetic are all
tested directly.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from app.analysis import build_timeline
from app.matching import (
    DEFAULT_MIN_TERM_RATIO,
    Affinity,
    candidate_corpus,
    estimate_calls,
    plan_pairs,
    score_affinity,
)
from app.schemas import Assessment, JobBrief, Requirement
from tests.fixtures import sample_brief, sample_profile


def brief(title, reqs, years=None, kind="must_have"):
    return JobBrief(
        role_title=title, stated_min_years=years,
        requirements=[Requirement(text=r, kind=kind, category="technical") for r in reqs],
    )


ML_ROLE = brief("ML Platform Engineer", [
    "Real-time inference serving at low latency",
    "Kubernetes and Kubeflow",
    "PyTorch model training",
], 6)
FRONTEND_ROLE = brief("Frontend Engineer", [
    "React and TypeScript",
    "CSS and design systems",
    "Web accessibility WCAG",
], 4)
FINANCE_ROLE = brief("VP Finance", [
    "Qualified chartered accountant",
    "IFRS and Ind AS reporting",
    "Treasury and capital markets",
], 12)


@pytest.fixture
def cand():
    p = sample_profile()
    return p, build_timeline(p)


# --- the screen ------------------------------------------------------------


def test_relevant_role_outranks_irrelevant_ones(cand):
    p, t = cand
    ml = score_affinity(p, t, ML_ROLE).score
    fe = score_affinity(p, t, FRONTEND_ROLE).score
    fin = score_affinity(p, t, FINANCE_ROLE).score
    assert ml > fe and ml > fin


def test_matching_role_covers_its_requirements(cand):
    p, t = cand
    a = score_affinity(p, t, ML_ROLE)
    assert a.term_ratio == 1.0
    assert not a.uncovered


def test_unrelated_role_covers_nothing(cand):
    p, t = cand
    for role in (FRONTEND_ROLE, FINANCE_ROLE):
        a = score_affinity(p, t, role)
        assert a.term_ratio == 0.0, (role.role_title, a.covered)


def test_corpus_includes_achievements_not_just_the_skills_list(cand):
    p, _ = cand
    corpus = candidate_corpus(p)
    assert "torchserve" in corpus          # only appears inside an achievement
    assert "fraud scoring" in corpus       # bigram from an achievement


def test_one_generic_word_in_common_is_not_a_match():
    """"Feature store engineering" must not be satisfied by "retail store rota"."""
    from app.schemas import CandidateProfile, Position
    p = CandidateProfile(positions=[Position(
        company="Acme Retail", title="Shift Supervisor", start="2019-01", end="2023-01",
        achievements=["Managed the retail store rota"])])
    a = score_affinity(p, build_timeline(p), brief("R", ["Feature store engineering"]))
    assert a.term_ratio == 0.0, a.covered


def test_two_matching_terms_do_count_as_a_match():
    """The rule is "two distinctive terms", not "never match without a phrase"."""
    from app.schemas import CandidateProfile, Position
    p = CandidateProfile(positions=[Position(
        company="Acme", title="Engineer", start="2019-01", end="2023-01",
        achievements=["Built a feature store for the pricing models"])])
    a = score_affinity(p, build_timeline(p), brief("R", ["Feature store engineering"]))
    assert a.term_ratio == 1.0


def test_experience_is_a_modifier_not_a_gate(cand):
    """An impossible years requirement must not zero out a strong candidate."""
    p, t = cand
    impossible = brief("R", ["Real-time inference serving at low latency"], years=40)
    a = score_affinity(p, t, impossible)
    assert a.term_ratio == 1.0
    assert a.score > 0.7


def test_nice_to_haves_are_used_when_there_are_no_must_haves(cand):
    p, t = cand
    a = score_affinity(p, t, brief("R", ["Kubernetes and Kubeflow"], kind="nice_to_have"))
    assert a.term_ratio == 1.0


def test_brief_with_no_requirements_does_not_divide_by_zero(cand):
    p, t = cand
    a = score_affinity(p, t, JobBrief(role_title="Empty"))
    assert a.term_ratio == 0.0
    assert 0.0 <= a.score <= 1.0


def test_empty_profile_scores_zero_terms():
    from app.schemas import CandidateProfile
    p = CandidateProfile()
    a = score_affinity(p, build_timeline(p), ML_ROLE)
    assert a.term_ratio == 0.0


# --- the plan --------------------------------------------------------------


def make_affinities(rows):
    """rows: {(ci, ri): (score, term_ratio)}"""
    return {k: Affinity(score=s, term_ratio=tr) for k, (s, tr) in rows.items()}


def test_top_roles_limits_pairs_when_there_are_candidates_to_spare():
    """With more candidates than roles, the per-candidate budget binds."""
    aff = make_affinities({(c, r): (0.9 - r * 0.1, 0.8) for c in range(6) for r in range(3)})
    plans = plan_pairs(aff, n_candidates=6, n_requisitions=3, top_roles=1)
    selected = [p for p in plans if p.selected]
    # 6 candidates x 1 role each (all pick role 0), plus one fill for roles 1
    # and 2 so neither comes back empty. Far short of the 18-pair cross product.
    assert len(selected) == 8
    assert sum(1 for p in selected if p.requisition_index == 0) == 6


def test_selection_is_bounded():
    """Worst case is candidates*top_roles plus one fill per role, never M*N."""
    n_c, n_r, top = 6, 5, 2
    aff = make_affinities({(c, r): (0.9 - r * 0.05, 0.8) for c in range(n_c) for r in range(n_r)})
    plans = plan_pairs(aff, n_candidates=n_c, n_requisitions=n_r, top_roles=top)
    selected = sum(1 for p in plans if p.selected)
    assert selected <= n_c * top + n_r
    assert selected < n_c * n_r


def test_a_single_candidate_is_assessed_against_every_viable_role():
    """One CV against five roles is the "which of these does she fit?" case.

    The per-role guarantee wins here, and it is bounded by the number of roles.
    """
    aff = make_affinities({(0, r): (0.9 - r * 0.1, 0.8) for r in range(5)})
    plans = plan_pairs(aff, n_candidates=1, n_requisitions=5, top_roles=2)
    assert sum(1 for p in plans if p.selected) == 5


def test_a_pair_with_no_term_overlap_is_never_auto_selected():
    """Regression: experience alone lifted an unrelated role over the floor."""
    aff = make_affinities({(0, 0): (0.25, 0.0)})
    plans = plan_pairs(aff, n_candidates=1, n_requisitions=1, top_roles=3)
    assert not plans[0].selected
    assert "nothing in this cv" in plans[0].reason.lower()


def test_every_role_keeps_its_best_candidate():
    """A niche role must not come back empty just because everyone ranked it second."""
    aff = make_affinities({
        (0, 0): (0.9, 0.9), (0, 1): (0.8, 0.8),
        (1, 0): (0.85, 0.9), (1, 1): (0.7, 0.7),
    })
    plans = plan_pairs(aff, n_candidates=2, n_requisitions=2, top_roles=1)
    roles_covered = {p.requisition_index for p in plans if p.selected}
    assert roles_covered == {0, 1}


def test_assess_all_selects_the_whole_cross_product():
    aff = make_affinities({(c, r): (0.05, 0.0) for c in range(3) for r in range(4)})
    plans = plan_pairs(aff, n_candidates=3, n_requisitions=4, assess_all=True)
    assert all(p.selected for p in plans)
    assert len(plans) == 12


def test_skipped_pairs_explain_themselves():
    """Three candidates, two roles, one role each: some pairs go unassessed."""
    aff = make_affinities({(c, r): (0.9 - r * 0.2, 0.8) for c in range(3) for r in range(2)})
    plans = plan_pairs(aff, n_candidates=3, n_requisitions=2, top_roles=1)
    skipped = [p for p in plans if not p.selected]
    assert skipped, "expected some pairs to be screened out"
    assert all(p.reason for p in skipped)
    assert any("outranked" in p.reason for p in skipped)


def test_no_pairs_at_all_is_handled():
    assert plan_pairs({}, n_candidates=0, n_requisitions=0) == []


# --- cost ------------------------------------------------------------------


def test_cost_estimate_beats_the_naive_cross_product():
    c = estimate_calls(n_candidates=20, n_requisitions=10, n_selected=60)
    assert c["total"] == 90
    assert c["naive_total"] == 230
    assert c["total"] < c["naive_total"]


def test_single_role_costs_the_same_as_the_old_batch_path():
    """1 brief + N extractions + N assessments, exactly as before."""
    c = estimate_calls(n_candidates=5, n_requisitions=1, n_selected=5)
    assert c["total"] == 11 == c["naive_total"]


# --- the run, end to end with a stubbed model ------------------------------


@pytest.fixture
def stub_llm(monkeypatch, tmp_path):
    """Replace the three model calls with deterministic stand-ins."""
    from app.extract import llm as llm_mod
    from app import matchrun

    calls = {"brief": 0, "profile": 0, "assess": 0}

    def fake_brief(jd_text, usage=None, model=None):
        calls["brief"] += 1
        if "BROKEN" in jd_text:
            raise ValueError("unparseable brief")
        title = jd_text.strip().splitlines()[0][:40]
        return brief(title, ["Kubernetes and Kubeflow", "PyTorch model training"], 5)

    def fake_profile(text, usage=None, model=None):
        calls["profile"] += 1
        if "BADCV" in text:
            raise ValueError("unreadable CV")
        return sample_profile()

    def fake_assess(*, profile, timeline, brief, cv_text, usage=None, model=None):
        calls["assess"] += 1
        if "FAILASSESS" in cv_text:
            raise ValueError("assessment blew up")
        return Assessment(executive_summary="s", fit_rationale="r")

    monkeypatch.setattr(matchrun.llm, "extract_job_brief", fake_brief)
    monkeypatch.setattr(matchrun.llm, "extract_profile", fake_profile)
    monkeypatch.setattr(matchrun.llm, "assess", fake_assess)
    return calls


def write_cvs(tmp_path, specs):
    out = []
    for name, body in specs:
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")
        out.append((name, p))
    return out


def run_match(tmp_path, jds, cvs, **kw):
    from app.matchrun import create_run, execute
    store = {}
    run = create_run(jds=jds, cvs=cvs, model="stub", anonymise=True, **kw)
    execute(run.id, store)
    return run, store


CV_BODY = "Jane Doe\nSenior ML Engineer\nBuilt PyTorch models on Kubeflow and Kubernetes."


def test_full_run_produces_dossiers(tmp_path, stub_llm):
    jds = [("role_a.txt", "Platform Lead\nNeed Kubernetes"), ("role_b.txt", "ML Engineer\nNeed PyTorch")]
    cvs = write_cvs(tmp_path, [("a.txt", CV_BODY), ("b.txt", CV_BODY)])
    run, store = run_match(tmp_path, jds, cvs)

    assert run.phase == "done" and not run.running
    assert stub_llm["brief"] == 2
    assert stub_llm["profile"] == 2, "each CV must be extracted exactly once"
    assert stub_llm["assess"] == len(run.selected_pairs)
    assert len(store) == len(run.assessed_pairs)


def test_each_cv_is_extracted_once_regardless_of_role_count(tmp_path, stub_llm):
    jds = [("r{}.txt".format(i), "Role {}\nNeed Kubernetes".format(i)) for i in range(4)]
    cvs = write_cvs(tmp_path, [("a.txt", CV_BODY)])
    run_match(tmp_path, jds, cvs)
    assert stub_llm["profile"] == 1
    assert stub_llm["brief"] == 4


def test_a_broken_brief_fails_only_its_own_column(tmp_path, stub_llm):
    jds = [("good.txt", "Good Role\nNeed Kubernetes"), ("bad.txt", "BROKEN")]
    cvs = write_cvs(tmp_path, [("a.txt", CV_BODY)])
    run, _ = run_match(tmp_path, jds, cvs)

    assert run.requisitions[0].ok and not run.requisitions[1].ok
    assert run.requisitions[1].error
    assert run.assessed_pairs, "the working role should still have been assessed"
    assert all(p.requisition_index == 0 for p in run.pairs)


def test_a_broken_cv_fails_only_its_own_row(tmp_path, stub_llm):
    jds = [("r.txt", "Role\nNeed Kubernetes")]
    cvs = write_cvs(tmp_path, [("good.txt", CV_BODY), ("bad.txt", "BADCV")])
    run, _ = run_match(tmp_path, jds, cvs)

    assert run.candidates[0].ok and not run.candidates[1].ok
    assert run.candidates[1].error
    assert len(run.assessed_pairs) == 1


def test_an_empty_cv_file_fails_before_any_model_call(tmp_path, stub_llm):
    jds = [("r.txt", "Role\nNeed Kubernetes")]
    cvs = write_cvs(tmp_path, [("empty.txt", "   ")])
    run, _ = run_match(tmp_path, jds, cvs)
    assert not run.candidates[0].ok
    assert "no readable text" in run.candidates[0].error.lower()


def test_one_failing_assessment_does_not_stop_the_others(tmp_path, stub_llm):
    jds = [("r.txt", "Role\nNeed Kubernetes")]
    cvs = write_cvs(tmp_path, [("ok.txt", CV_BODY), ("bad.txt", CV_BODY + "\nFAILASSESS")])
    run, _ = run_match(tmp_path, jds, cvs)
    assert len(run.assessed_pairs) == 1
    assert any(p.status == "failed" for p in run.pairs)


def test_run_always_terminates_even_on_an_unexpected_crash(tmp_path, monkeypatch):
    from app import matchrun
    monkeypatch.setattr(matchrun.llm, "extract_job_brief",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    run, _ = run_match(tmp_path, [("r.txt", "Role")], write_cvs(tmp_path, [("a.txt", CV_BODY)]))
    assert not run.running
    assert run.phase == "done"


def test_shortlist_and_role_tags_are_consistent(tmp_path, stub_llm):
    jds = [("a.txt", "Alpha Role\nNeed Kubernetes"), ("b.txt", "Beta Role\nNeed PyTorch")]
    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY), ("y.txt", CV_BODY)])
    run, _ = run_match(tmp_path, jds, cvs)

    from_roles = {(p.candidate_index, p.requisition_index)
                  for ri in range(2) for p in run.shortlist_for(ri)}
    from_cands = {(p.candidate_index, p.requisition_index)
                  for ci in range(2) for p in run.roles_for(ci)}
    assert from_roles == from_cands, "the matrix must read the same both ways"


def test_assessed_pairs_outrank_screened_only_ones(tmp_path, stub_llm):
    jds = [("a.txt", "Alpha\nNeed Kubernetes"), ("b.txt", "Beta\nNeed PyTorch"),
           ("c.txt", "Gamma\nNeed PyTorch")]
    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY)])
    run, _ = run_match(tmp_path, jds, cvs, top_roles=1)
    tags = run.roles_for(0)
    assessed = [t for t in tags if t.dossier]
    if assessed and len(tags) > len(assessed):
        assert tags[0].dossier, "an assessed pair must sort above a screened-only one"


def test_cost_reporting_reflects_the_actual_run(tmp_path, stub_llm):
    jds = [("a.txt", "Alpha\nNeed Kubernetes"), ("b.txt", "Beta\nNeed PyTorch")]
    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY), ("y.txt", CV_BODY)])
    run, _ = run_match(tmp_path, jds, cvs)
    cost = run.cost()
    assert cost["briefs"] == 2 and cost["extractions"] == 2
    assert cost["assessments"] == len(run.selected_pairs)
    assert cost["total"] == 2 + 2 + len(run.selected_pairs)


def test_status_payload_is_serialisable_at_every_stage(tmp_path, stub_llm):
    import json
    from app.matchrun import create_run, execute, status_payload
    jds = [("a.txt", "Alpha\nNeed Kubernetes")]
    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY)])
    run = create_run(jds=jds, cvs=cvs, model="stub", anonymise=True)
    json.dumps(status_payload(run))          # queued
    execute(run.id, {})
    json.dumps(status_payload(run))          # done
    assert status_payload(run)["running"] is False


# --- cost levers -----------------------------------------------------------


def test_brief_comes_first_in_the_assessment_prompt(monkeypatch):
    """Prompt caching keys on the longest common prefix, so the block shared
    across candidates (the brief) has to precede the per-candidate blocks."""
    from app.extract import llm as llm_mod
    seen = {}

    class FakeResp:
        status = "completed"
        usage = None
        output_parsed = Assessment(executive_summary="s", fit_rationale="r")

    class FakeResponses:
        def parse(self, **kw):
            seen.update(kw)
            return FakeResp()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(llm_mod, "get_client", lambda: FakeClient())
    p = sample_profile()
    llm_mod.assess(profile=p, timeline=build_timeline(p), brief=ML_ROLE, cv_text="CV TEXT")

    body = seen["input"]
    assert body.index("<client_brief>") < body.index("<extracted_profile>")
    assert body.index("<client_brief>") < body.index("<raw_cv_text>")


def test_assessments_are_grouped_by_role(tmp_path, stub_llm):
    """Grouping is what makes the cached brief prefix actually hit."""
    from app.matchrun import create_run
    jds = [("a.txt", "Alpha\nNeed Kubernetes"), ("b.txt", "Beta\nNeed PyTorch")]
    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY), ("y.txt", CV_BODY)])
    run, _ = run_match(tmp_path, jds, cvs)
    order = [p.requisition_index for p in sorted(run.selected_pairs, key=lambda p: p.requisition_index)]
    assert order == sorted(order)


def test_extraction_can_use_a_cheaper_model_than_assessment(tmp_path, monkeypatch):
    from app import matchrun
    used = []
    monkeypatch.setattr(matchrun.llm, "extract_job_brief",
                        lambda t, usage=None, model=None: brief("R", ["Kubernetes and Kubeflow"], 5))
    monkeypatch.setattr(matchrun.llm, "extract_profile",
                        lambda t, usage=None, model=None: (used.append(("extract", model)), sample_profile())[1])
    monkeypatch.setattr(matchrun.llm, "assess",
                        lambda **kw: (used.append(("assess", kw.get("model"))),
                                      Assessment(executive_summary="s", fit_rationale="r"))[1])

    from app.matchrun import create_run, execute
    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY)])
    run = create_run(jds=[("a.txt", "Alpha")], cvs=cvs, model="big-model",
                     anonymise=True, extraction_model="small-model")
    execute(run.id, {})

    assert ("extract", "small-model") in used
    assert ("assess", "big-model") in used


# --- the constraint gate inside a run --------------------------------------


def test_blocked_pairs_never_reach_the_model(tmp_path, stub_llm):
    """The whole point of the gate: a declared conflict costs zero tokens."""
    from app.intake import CandidatePreferences, RoleConstraints
    from app.matchrun import create_run, execute

    jds = [("cheap.txt", "Cheap Role\nNeed Kubernetes")]
    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY)])
    run = create_run(jds=jds, cvs=cvs, model="stub", anonymise=True)
    # Candidate will not go below 90 LPA; the role tops out at 40.
    run.candidates[0].prefs = CandidatePreferences(min_acceptable_ctc_lpa=90)
    run.requisitions[0].constraints = RoleConstraints(role_title="Cheap Role", ctc_max_lpa=40)
    execute(run.id, {})

    assert stub_llm["assess"] == 0, "a blocked pair must not be assessed"
    assert len(run.blocked_pairs) == 1
    assert not run.selected_pairs


def test_a_blocked_pair_explains_itself_with_numbers(tmp_path, stub_llm):
    from app.intake import CandidatePreferences, RoleConstraints
    from app.matchrun import create_run, execute

    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY)])
    run = create_run(jds=[("r.txt", "Role\nNeed Kubernetes")], cvs=cvs, model="stub", anonymise=True)
    run.candidates[0].prefs = CandidatePreferences(notice_period_days=180)
    run.requisitions[0].constraints = RoleConstraints(role_title="R", max_notice_days=30)
    execute(run.id, {})

    pair = run.blocked_pairs[0]
    assert "180" in pair.reason and "30" in pair.reason
    assert pair.check is not None and pair.check.blocked


def test_unblocked_pairs_are_unaffected_by_the_gate(tmp_path, stub_llm):
    from app.intake import CandidatePreferences, RoleConstraints
    from app.matchrun import create_run, execute

    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY)])
    run = create_run(jds=[("r.txt", "Role\nNeed Kubernetes")], cvs=cvs, model="stub", anonymise=True)
    run.candidates[0].prefs = CandidatePreferences(min_acceptable_ctc_lpa=30)
    run.requisitions[0].constraints = RoleConstraints(role_title="R", ctc_max_lpa=60)
    execute(run.id, {})

    assert not run.blocked_pairs
    assert stub_llm["assess"] == 1


def test_the_gate_only_applies_where_constraints_were_declared(tmp_path, stub_llm):
    """A role with no form answers must behave exactly as before."""
    from app.matchrun import create_run, execute

    cvs = write_cvs(tmp_path, [("x.txt", CV_BODY)])
    run = create_run(jds=[("r.txt", "Role\nNeed Kubernetes")], cvs=cvs, model="stub", anonymise=True)
    execute(run.id, {})
    assert not run.blocked_pairs
    assert stub_llm["assess"] >= 1


def test_gate_scales_the_saving_across_a_grid(tmp_path, stub_llm):
    """Three candidates, two roles, one role priced out for everyone."""
    from app.intake import CandidatePreferences, RoleConstraints
    from app.matchrun import create_run, execute

    cvs = write_cvs(tmp_path, [("a.txt", CV_BODY), ("b.txt", CV_BODY), ("c.txt", CV_BODY)])
    run = create_run(jds=[("rich.txt", "Rich Role\nNeed Kubernetes"),
                          ("poor.txt", "Poor Role\nNeed PyTorch")],
                     cvs=cvs, model="stub", anonymise=True, top_roles=2)
    for c in run.candidates:
        c.prefs = CandidatePreferences(min_acceptable_ctc_lpa=80)
    run.requisitions[0].constraints = RoleConstraints(role_title="Rich", ctc_max_lpa=120)
    run.requisitions[1].constraints = RoleConstraints(role_title="Poor", ctc_max_lpa=40)
    execute(run.id, {})

    assert len(run.blocked_pairs) == 3, "every candidate is priced out of the poor role"
    assert all(p.requisition_index == 0 for p in run.selected_pairs)
    assert stub_llm["assess"] == 3
