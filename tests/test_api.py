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


def test_embed_renders_the_document(client, demo_id):
    r = client.get("/dossier/{}/embed".format(demo_id))
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text or "<html" in r.text


def test_pdf_downloads(client, demo_id):
    r = client.get("/dossier/{}/pdf?blind=1".format(demo_id))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


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
