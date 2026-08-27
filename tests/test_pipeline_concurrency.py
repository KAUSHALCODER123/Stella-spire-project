"""build_dossier's two independent model calls run concurrently.

CV extraction and brief parsing read different documents and neither needs
the other's output, so a single-CV run should not pay for them serially.
This exercises that directly rather than trusting a wall-clock number.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.extract import llm
from app.pipeline import build_dossier
from app.schemas import Assessment
from tests.fixtures import sample_brief, sample_profile

CV_TEXT = "Jane Doe\nHead of Finance\nOwned month-end close and controllership."


@pytest.fixture
def slow_stub_llm(monkeypatch, tmp_path):
    """Each call sleeps, so overlap is only possible if they run concurrently."""
    calls = {"profile": [], "brief": []}

    def fake_profile(text, usage=None, model=None):
        calls["profile"].append(time.perf_counter())
        time.sleep(0.2)
        return sample_profile()

    def fake_brief(jd_text, usage=None, model=None):
        calls["brief"].append(time.perf_counter())
        time.sleep(0.2)
        return sample_brief()

    def fake_assess(*, profile, timeline, brief, cv_text, usage=None, model=None):
        return Assessment(executive_summary="s", fit_rationale="r")

    monkeypatch.setattr(llm, "extract_profile", fake_profile)
    monkeypatch.setattr(llm, "extract_job_brief", fake_brief)
    monkeypatch.setattr(llm, "assess", fake_assess)

    cv_path = tmp_path / "cv.txt"
    cv_path.write_text(CV_TEXT, encoding="utf-8")
    return calls, cv_path


def test_profile_and_brief_extraction_overlap_on_a_single_cv_run(slow_stub_llm):
    calls, cv_path = slow_stub_llm
    started = time.perf_counter()
    dossier = build_dossier(cv_path=cv_path, jd_text="Head of Finance role, 5+ years.", model="stub")
    elapsed = time.perf_counter() - started

    assert dossier.profile.full_name == "Meera Ramanathan"
    assert dossier.brief.role_title == "Chief Financial Officer"
    # Serial would be >= 0.4s; concurrent should land close to the single 0.2s sleep.
    assert elapsed < 0.35, "profile and brief extraction did not run concurrently"
    # Both calls should have started within the same short window.
    assert abs(calls["profile"][0] - calls["brief"][0]) < 0.1


def test_a_precomputed_brief_skips_the_concurrent_path(slow_stub_llm):
    """Batch mode already parsed the brief once; passing it in must not
    trigger a second brief call or spin up an executor for nothing."""
    calls, cv_path = slow_stub_llm
    brief = sample_brief()

    dossier = build_dossier(cv_path=cv_path, jd_text="ignored", model="stub", brief=brief)

    assert dossier.brief is brief
    assert calls["brief"] == []
    assert len(calls["profile"]) == 1
