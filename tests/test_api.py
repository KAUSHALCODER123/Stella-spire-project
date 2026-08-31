"""Route-level tests through FastAPI's TestClient.

These exercise the real endpoints without a running server and without
spending a token: the only analysis path used is /demo, which renders the
stored fixture.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

HTML = {"accept": "text/html"}
JSON = {"accept": "application/json"}


@pytest.fixture(scope="module")
def client():
    """Signed in as the agency account.

    Every workspace surface is behind the single sign-in now, so these route
    tests need a session. Public pages are covered in test_accounts.py.
    """
    from app.accounts import ADMIN_EMAIL, ADMIN_PASSWORD, seed
    seed()
    with TestClient(app) as c:
        r = c.post("/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
                   follow_redirects=False)
        assert r.status_code == 303, "could not sign in for route tests"
        yield c


@pytest.fixture(scope="module")
def demo_id(client):
    r = client.post("/demo", follow_redirects=False)
    assert r.status_code == 303
    return r.headers["location"].rsplit("/", 1)[-1]


# --- pages render ----------------------------------------------------------


@pytest.mark.parametrize("path", ["/dashboard", "/candidates", "/shortlists", "/roles"])
def test_workspace_pages_render(client, path):
    r = client.get(path, headers=HTML)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "SpireDossier" in r.text


def test_landing_renders_for_everyone(client):
    r = client.get("/", headers=HTML)
    assert r.status_code == 200
    assert "Stellaspire" in r.text


def test_health(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert "model" in body and "dossiers_in_memory" in body


def test_every_page_declares_a_mobile_viewport(client):
    for path in ["/", "/login", "/register", "/apply", "/dashboard", "/candidates", "/shortlists"]:
        assert 'name="viewport"' in client.get(path, headers=HTML).text


# --- the sample flow -------------------------------------------------------


def test_demo_creates_a_report(client, demo_id):
    r = client.get("/dossier/{}".format(demo_id), headers=HTML)
    assert r.status_code == 200
    for expected in ["Overall suitability", "Matched requirements", "Missing requirements", "Experience match"]:
        assert expected in r.text, expected


def test_blind_and_named_views(client, demo_id):
    blind = client.get("/dossier/{}?blind=1".format(demo_id), headers=HTML).text.lower()
    named = client.get("/dossier/{}?blind=0".format(demo_id), headers=HTML).text.lower()
    assert "meera" not in blind and "ramanathan" not in blind
    assert "meera" in named
    assert "blind client view" in blind and "client-safe" in blind
    assert "named · internal view" in named and "do not share" in named
    assert "download blind client pdf" in blind
    assert "review named export" in named


def test_embed_renders_the_document(client, demo_id):
    r = client.get("/dossier/{}/embed".format(demo_id))
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text or "<html" in r.text


def test_pdf_downloads(client, demo_id):
    r = client.get("/dossier/{}/pdf?blind=1".format(demo_id))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"
    assert "BLIND.pdf" in r.headers["content-disposition"]


def test_named_pdf_requires_a_separate_confirmation(client, demo_id):
    warning = client.get("/dossier/{}/pdf?blind=0".format(demo_id), headers=HTML)
    assert warning.status_code == 200
    assert "text/html" in warning.headers["content-type"]
    assert "This is not a blind client dossier" in warning.text
    assert "Download named internal PDF" in warning.text

    download = client.get("/dossier/{}/pdf?blind=0&confirm_named=1".format(demo_id))
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/pdf"
    assert download.content[:5] == b"%PDF-"
    assert "NAMED%20INTERNAL.pdf" in download.headers["content-disposition"]


# --- not found -------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "/dossier/missing", "/dossier/missing/pdf", "/dossier/missing/embed",
    "/batch/missing", "/batch/missing/status", "/no-such-page",
])
def test_missing_things_are_404(client, path):
    assert client.get(path, headers=JSON).status_code == 404


def test_browsers_get_the_branded_404(client):
    r = client.get("/dossier/missing", headers=HTML)
    assert r.status_code == 404
    assert "text/html" in r.headers["content-type"]
    assert "We could not find that page" in r.text


def test_api_clients_still_get_json_404(client):
    r = client.get("/dossier/missing", headers=JSON)
    assert "application/json" in r.headers["content-type"]
    assert "detail" in r.json()


def test_bad_query_value_is_a_branded_422_for_browsers(client, demo_id):
    r = client.get("/dossier/{}?blind=notanumber".format(demo_id), headers=HTML)
    assert r.status_code == 422
    assert "did not look right" in r.text


# --- method mismatches -----------------------------------------------------


@pytest.mark.parametrize("method,path", [("get", "/generate"), ("get", "/demo"), ("post", "/")])
def test_method_not_allowed(client, method, path):
    assert getattr(client, method)(path, headers=JSON).status_code == 405


# --- upload validation (no model call reached) -----------------------------


def test_upload_requires_a_file(client):
    r = client.post("/generate", data={"jd_text": "Some role"}, headers=HTML)
    assert r.status_code == 400
    assert "No CV uploaded" in r.text


def test_upload_requires_a_job_description(client):
    r = client.post("/generate", data={"jd_text": "   "},
                    files=[("cv", ("cv.txt", b"Jane Doe, engineer."))], headers=HTML)
    assert r.status_code == 400
    assert "No job description" in r.text


def test_unsupported_file_type_is_rejected(client):
    r = client.post("/generate", data={"jd_text": "Role"},
                    files=[("cv", ("virus.exe", b"MZ"))], headers=HTML)
    assert r.status_code == 400
    assert "Unsupported file" in r.text


def test_oversized_file_is_rejected(client):
    r = client.post("/generate", data={"jd_text": "Role"},
                    files=[("cv", ("big.txt", b"x" * (11 * 1024 * 1024)))], headers=HTML)
    assert r.status_code == 400
    assert "too large" in r.text.lower()


def test_a_filename_that_would_crash_the_filesystem_is_accepted(client, monkeypatch):
    """Regression: "Resume: Senior Dev.pdf" used to 500 before any model call.

    The key is not configured off here, so the request should get past
    validation and file-writing and fail only at the API-key check.
    """
    from app import main as main_mod
    monkeypatch.setattr(main_mod.settings, "openai_api_key", "")
    r = client.post("/generate", data={"jd_text": "Role"},
                    files=[("cv", ("Resume: Senior Dev.txt", b"Jane Doe"))], headers=HTML)
    assert r.status_code == 400
    assert "No API key" in r.text  # got past the crash, stopped at the key


def test_no_api_key_gives_an_actionable_message(client, monkeypatch):
    from app import main as main_mod
    monkeypatch.setattr(main_mod.settings, "openai_api_key", "")
    r = client.post("/generate", data={"jd_text": "Role"},
                    files=[("cv", ("cv.txt", b"Jane Doe"))], headers=HTML)
    assert r.status_code == 400
    assert ".env" in r.text and "check_setup" in r.text


# --- listing pages reflect state ------------------------------------------


def test_candidates_page_lists_the_demo_report(client, demo_id):
    assert "/dossier/{}".format(demo_id) in client.get("/candidates", headers=HTML).text


def test_batch_page_is_an_actionable_triage_queue(client):
    import time
    from pathlib import Path
    from app.batch import BATCHES, Batch, BatchItem
    from tests.fixtures import sample_dossier

    row = BatchItem(filename="meera.pdf", path=Path("meera.pdf"), index=0,
                    status="done", stage="done", dossier_id="triage-dossier",
                    dossier=sample_dossier())
    batch = Batch(id="triage-test", jd_text="", model="gpt-4o", anonymise=True,
                  items=[row], finished_at=time.time())
    BATCHES[batch.id] = batch
    try:
        body = client.get("/batch/{}".format(batch.id), headers=HTML).text
        for expected in ("Search candidate", "All decisions", "Any coverage",
                         "Recommended rank", "Shortlist", "Maybe", "Reject"):
            assert expected in body

        response = client.post("/batch/{}/decision".format(batch.id), data={
            "item_index": 0, "decision": "shortlist",
        }, follow_redirects=False)
        assert response.status_code == 303
        assert row.decision == "shortlist"
        assert 'data-decision="shortlist"' in client.get("/batch/{}".format(batch.id)).text
    finally:
        BATCHES.pop(batch.id, None)


def test_batch_rejects_an_unknown_decision(client):
    import time
    from pathlib import Path
    from app.batch import BATCHES, Batch, BatchItem
    from tests.fixtures import sample_dossier

    row = BatchItem(filename="meera.pdf", path=Path("meera.pdf"), index=0,
                    status="done", stage="done", dossier=sample_dossier())
    batch = Batch(id="bad-decision-test", jd_text="", model="gpt-4o", anonymise=True,
                  items=[row], finished_at=time.time())
    BATCHES[batch.id] = batch
    try:
        response = client.post("/batch/{}/decision".format(batch.id), data={
            "item_index": 0, "decision": "hire-immediately",
        })
        assert response.status_code == 400
        assert row.decision == "unreviewed"
    finally:
        BATCHES.pop(batch.id, None)


# --- the demo's opening screen --------------------------------------------


def test_the_prefilled_sample_pair_exists_on_disk():
    """The workspace falls back to an empty brief box if these go missing.

    That degrades silently, so a rename during a re-vertical would leave the
    main screen blank with nothing failing anywhere.
    """
    from app.main import _sample_paths

    cv, jd = _sample_paths()
    assert cv.exists(), "sample CV missing at {}".format(cv)
    assert jd.exists(), "sample JD missing at {}".format(jd)


def test_the_prefilled_brief_is_in_the_vertical_being_sold():
    """A finance consultancy's main screen must not open on an ML role."""
    from app.main import _sample_paths

    _, jd = _sample_paths()
    text = jd.read_text(encoding="utf-8").lower()
    assert any(term in text for term in ("cfo", "finance", "financial")), text[:200]


def test_the_workspace_actually_renders_that_brief(client):
    body = client.get("/workspace").text
    assert "Chief Financial Officer" in body or "CFO" in body


# --- the zero-cost demo must show the free gate doing something ------------


def test_the_demo_match_exercises_the_real_constraint_gate(client):
    """A flagship stat reading zero in the demo undersells the feature.

    The gate is the cheapest thing the product does and the easiest to
    overlook, so the demo has to show it actually eliminating pairs -- with
    reasons the live code computed, not reasons written into a fixture.
    """
    from app.main import RUNS

    resp = client.post("/match/demo", follow_redirects=False)
    run = RUNS[resp.headers["location"].rsplit("/", 1)[-1]]

    assert run.blocked_pairs, "the free gate ruled nothing out"
    for pair in run.blocked_pairs:
        assert pair.check is not None and pair.check.blocked
        assert any(ch.isdigit() for ch in pair.reason), pair.reason


def test_match_matrix_exposes_rejection_reasons_and_override_actions(client, monkeypatch):
    from app.main import RUNS
    import app.main as main_mod

    response = client.post("/match/demo", follow_redirects=False)
    run_id = response.headers["location"].rsplit("/", 1)[-1]
    run = RUNS[run_id]
    body = client.get("/match/{}".format(run_id), headers=HTML).text

    assert "HARD REJECT" in body
    assert "SCREENED OUT" in body
    assert "Hard-constraint rejected" in body
    assert "Affinity-screened out" in body
    assert "Assess anyway" in body
    assert any(pair.reason in body for pair in run.blocked_pairs)

    blocked = run.blocked_pairs[0]
    monkeypatch.setattr(main_mod, "run_in_background", lambda *args: None)
    override = client.post(
        "/match/{}/pairs/{}/{}/assess".format(
            run_id, blocked.candidate_index, blocked.requisition_index),
        follow_redirects=False,
    )
    assert override.status_code == 303
    assert blocked.selected and blocked.status == "queued"
    assert "Recruiter override" in blocked.reason


def test_the_demo_match_is_in_the_finance_vertical(client):
    from app.main import RUNS

    resp = client.post("/match/demo", follow_redirects=False)
    run = RUNS[resp.headers["location"].rsplit("/", 1)[-1]]
    titles = " ".join(r.title for r in run.requisitions).lower()
    assert "financial" in titles or "finance" in titles
    assert "ml engineer" not in titles and "genai" not in titles


def test_a_free_model_run_reports_no_cost_rather_than_a_guess(client):
    """The old template applied a GPT-4o rate to whatever had run."""
    from app.main import RUNS

    resp = client.post("/match/demo", follow_redirects=False)
    run_id = resp.headers["location"].rsplit("/", 1)[-1]
    RUNS[run_id].model = "dots-studio/dots-3-note-preview:free"

    body = client.get("/match/{}".format(run_id)).text
    assert "₹0" in body


def test_the_demo_match_populates_each_employers_own_shortlist(client):
    """Tenant isolation is the demo's strongest beat and must cost nothing.

    Without source_role_id on the demo requisitions the employer dashboard is
    empty, so the claim "a client sees only their own candidates" cannot be
    shown without spending tokens on a live run.
    """
    import html as _html

    client.post("/match/demo", follow_redirects=False)
    client.post("/logout", follow_redirects=False)

    client.post("/login", data={"email": "careers@alderline.example", "password": "admin123"},
                follow_redirects=False)
    alderline = _html.unescape(client.get("/dashboard").text)

    assert "Financial Controller" in alderline
    assert "Head of Financial Planning & Analysis" in alderline
    # Meridian's opening must not appear on Alderline's dashboard.
    assert "Chief Financial Officer" not in alderline


def test_an_employer_is_not_shown_the_agencys_matching_controls(client):
    """/roles is shared between the agency and its clients.

    An employer can never see other clients' applicants, so the admin-only
    "Applicants" list and "Run the match" button must not render for them --
    otherwise they see a permanently-broken "Match 0 x N" control and a false
    "No applicants yet" message even when people have applied.
    """
    client.post("/login", data={"email": "careers@alderline.example", "password": "admin123"},
                follow_redirects=False)
    body = client.get("/roles").text

    assert "My roles" in body
    assert "Run the match" not in body
    assert "No applicants yet" not in body
    assert "Matching" in body


def _as_admin(client):
    from app.accounts import ADMIN_EMAIL, ADMIN_PASSWORD
    client.post("/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
                follow_redirects=False)


def test_candidates_page_shows_the_applicant_pool_unscored(client):
    """Applicants who used the seeker form never appeared on /candidates --
    only CVs the admin had personally run through /generate did."""
    _as_admin(client)
    body = client.get("/candidates").text
    assert "Meera Ramanathan" in body
    assert "Applied, awaiting match" in body


def test_filtering_candidates_by_role_previews_the_free_constraint_gate(client):
    """Picking a role should show who clears it before anything is spent."""
    from app.main import ROLE_LIBRARY
    _as_admin(client)
    cfo_id = next(rid for rid, rc in ROLE_LIBRARY.items() if rc.role_title == "Chief Financial Officer")

    body = client.get("/candidates", params={"role_id": cfo_id}).text
    assert "Chief Financial Officer" in body
    # Rohit's floor (125 LPA) is inside the CFO band (up to 130) but Daniel,
    # priced out and unwilling to relocate, should read as blocked somewhere.
    assert "gate" in body  # the badge class renders at all


def test_matching_selected_applicants_against_a_role(client, inline_threads, monkeypatch):
    """The other end of /roles/match: hand-pick CVs, hand-pick one role."""
    from app.main import APPLICANTS, ROLE_LIBRARY, RUNS, settings
    from app import matchrun
    from app.schemas import Assessment, JobBrief
    from tests.fixtures import sample_profile

    def fake_brief(jd_text, usage=None, model=None):
        return JobBrief(role_title="Chief Financial Officer", stated_min_years=5, requirements=[])

    def fake_profile(text, usage=None, model=None):
        return sample_profile()

    def fake_assess(*, profile, timeline, brief, cv_text, usage=None, model=None):
        return Assessment(executive_summary="s", fit_rationale="r")

    monkeypatch.setattr(settings, "openai_api_key", "test-key", raising=False)
    monkeypatch.setattr(matchrun.llm, "extract_job_brief", fake_brief)
    monkeypatch.setattr(matchrun.llm, "extract_profile", fake_profile)
    monkeypatch.setattr(matchrun.llm, "assess", fake_assess)

    _as_admin(client)
    cfo_id = next(rid for rid, rc in ROLE_LIBRARY.items() if rc.role_title == "Chief Financial Officer")
    meera_id = next(aid for aid, a in APPLICANTS.items() if a["prefs"].full_name == "Meera Ramanathan")

    resp = client.post("/candidates/match", data={
        "role_id": cfo_id, "applicant_ids": [meera_id], "anonymise": "",
    }, follow_redirects=False)

    assert resp.status_code == 303
    run_id = resp.headers["location"].rsplit("/", 1)[-1]
    run = RUNS[run_id]
    assert run.requisitions[0].source_role_id == cfo_id
    assert len(run.candidates) == 1
    assert run.candidates[0].prefs.full_name == "Meera Ramanathan"


def test_matching_with_no_candidates_selected_is_rejected(client):
    from app.main import ROLE_LIBRARY
    _as_admin(client)
    cfo_id = next(iter(ROLE_LIBRARY))

    resp = client.post("/candidates/match", data={"role_id": cfo_id, "applicant_ids": []})
    assert resp.status_code == 400


def test_email_action_is_disabled_without_gmail_credentials(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "gmail_address", "", raising=False)
    monkeypatch.setattr(settings, "gmail_app_password", "", raising=False)

    _as_admin(client)
    body = client.get("/candidates").text
    assert "Email notifications are off" in body


def test_emailing_selected_applicants_sends_one_message_each(client, monkeypatch):
    from app.config import settings
    from app.main import APPLICANTS, ROLE_LIBRARY
    from app import notify

    monkeypatch.setattr(settings, "gmail_address", "agency@example.com", raising=False)
    monkeypatch.setattr(settings, "gmail_app_password", "app-password", raising=False)
    sent = []
    monkeypatch.setattr(notify, "send_application_received",
                         lambda **kw: sent.append(kw))
    # main.py imported the function by name, so the patch has to land there too.
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "send_application_received", lambda **kw: sent.append(kw))

    _as_admin(client)
    cfo_id = next(rid for rid, rc in ROLE_LIBRARY.items() if rc.role_title == "Chief Financial Officer")
    meera_id = next(aid for aid, a in APPLICANTS.items() if a["prefs"].full_name == "Meera Ramanathan")

    resp = client.post("/candidates/notify", data={
        "role_id": cfo_id, "applicant_ids": [meera_id],
    }, follow_redirects=False)

    assert resp.status_code == 303
    assert "notified=1" in resp.headers["location"]
    assert len(sent) == 1
    assert sent[0]["to_email"] == "meera.ramanathan.fin@example.com"
    assert sent[0]["role_title"] == "Chief Financial Officer"


def test_emailing_with_no_candidates_selected_is_rejected(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "gmail_address", "agency@example.com", raising=False)
    monkeypatch.setattr(settings, "gmail_app_password", "app-password", raising=False)

    _as_admin(client)
    resp = client.post("/candidates/notify", data={"role_id": "", "applicant_ids": []})
    assert resp.status_code == 400
