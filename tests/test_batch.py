"""Tests for batch ranking and progress accounting.

No API key and no network: BatchItems are given hand-built dossiers, so the
ordering rules are tested directly.
"""

from __future__ import annotations

import copy

from app.batch import Batch, BatchItem, status_payload
from app.schemas import RequirementMatch, RiskFlag
from tests.fixtures import sample_dossier


def item(name: str, *, strong: int = 0, partial: int = 0, absent: int = 0, high_flags: int = 0) -> BatchItem:
    """A finished item whose dossier yields the requested coverage shape."""
    d = copy.deepcopy(sample_dossier())

    must = [r.text for r in d.brief.requirements if r.kind == "must_have"]
    verdicts = (["strong"] * strong) + (["partial"] * partial) + (["absent"] * absent)
    verdicts += ["absent"] * (len(must) - len(verdicts))

    d.assessment.requirement_matches = [
        RequirementMatch(requirement=text, verdict=v, evidence=None)
        for text, v in zip(must, verdicts)
    ]
    d.flags = [RiskFlag(kind="logistics", severity="high", summary="x") for _ in range(high_flags)]

    it = BatchItem(filename=name, path=None, index=0)  # type: ignore[arg-type]
    it.dossier = d
    it.status = "done"
    it.stage = "done"
    it.dossier_id = name
    return it


def batch_of(*items: BatchItem) -> Batch:
    b = Batch(id="t", jd_text="", model="gpt-4o", anonymise=True, items=list(items))
    for n, it in enumerate(b.items):
        it.index = n
    return b


# --- ranking ---------------------------------------------------------------


def test_higher_coverage_ranks_first():
    low = item("low.pdf", strong=2)
    high = item("high.pdf", strong=7)
    order = [i.filename for i in batch_of(low, high).ranked()]
    assert order == ["high.pdf", "low.pdf"]


def test_partial_counts_toward_coverage():
    """A partial match is still coverage; ignoring it would misrank."""
    only_strong = item("a.pdf", strong=3)
    mixed = item("b.pdf", strong=3, partial=2)
    order = [i.filename for i in batch_of(only_strong, mixed).ranked()]
    assert order == ["b.pdf", "a.pdf"]


def test_strong_breaks_a_coverage_tie():
    mostly_partial = item("partial.pdf", strong=1, partial=4)
    mostly_strong = item("strong.pdf", strong=4, partial=1)
    order = [i.filename for i in batch_of(mostly_partial, mostly_strong).ranked()]
    assert order == ["strong.pdf", "partial.pdf"]


def test_high_flags_break_a_remaining_tie():
    clean = item("clean.pdf", strong=3, high_flags=0)
    flagged = item("flagged.pdf", strong=3, high_flags=3)
    order = [i.filename for i in batch_of(flagged, clean).ranked()]
    assert order == ["clean.pdf", "flagged.pdf"]


def test_ranking_is_stable_for_identical_candidates():
    a, b = item("b.pdf", strong=3), item("a.pdf", strong=3)
    order = [i.filename for i in batch_of(a, b).ranked()]
    assert order == ["a.pdf", "b.pdf"]  # falls back to filename, not input order


def test_failed_items_are_excluded_from_the_ranking():
    ok = item("ok.pdf", strong=3)
    bad = BatchItem(filename="bad.pdf", path=None, index=1, status="failed", error="unreadable")  # type: ignore[arg-type]
    b = batch_of(ok, bad)
    assert [i.filename for i in b.ranked()] == ["ok.pdf"]
    assert [i.filename for i in b.failed] == ["bad.pdf"]


# --- progress accounting ---------------------------------------------------


def test_completed_counts_failures_as_finished():
    """A failed CV must not leave the progress bar stuck short of 100%."""
    ok = item("ok.pdf", strong=1)
    bad = BatchItem(filename="bad.pdf", path=None, index=1, status="failed")  # type: ignore[arg-type]
    b = batch_of(ok, bad)
    assert b.completed == 2
    assert b.percent == 100


def test_queued_items_are_not_complete():
    b = batch_of(item("done.pdf", strong=1), BatchItem(filename="q.pdf", path=None, index=1))  # type: ignore[arg-type]
    assert b.completed == 1
    assert b.percent == 50


def test_status_payload_shape():
    b = batch_of(item("a.pdf", strong=5))
    payload = status_payload(b)
    assert payload["total"] == 1
    assert payload["items"][0]["filename"] == "a.pdf"
    assert payload["items"][0]["coverage"] is not None
    assert "role_title" in payload
    assert payload["items"][0]["decision"] == "unreviewed"


def test_recruiter_decision_does_not_change_algorithmic_rank():
    high = item("high.pdf", strong=7)
    low = item("low.pdf", strong=2)
    low.decision = "shortlist"
    assert [i.filename for i in batch_of(low, high).ranked()] == ["high.pdf", "low.pdf"]


def test_status_payload_survives_an_unparsed_brief():
    """The heading polls role_title before the brief has been read."""
    b = Batch(id="t", jd_text="", model="gpt-4o", anonymise=True, items=[])
    payload = status_payload(b)
    assert payload["role_title"] is None
    assert payload["percent"] == 100  # an empty batch is not "0% forever"
