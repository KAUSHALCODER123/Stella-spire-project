"""End-to-end journeys, and the boundaries between them.

One complete story: an employer registers, posts a role, a candidate applies,
the agency runs matching, and the employer sees exactly their own shortlist
and nothing else. Then the edges around it.

The three model calls are stubbed, so this runs offline and in seconds.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.accounts import ADMIN_EMAIL, ADMIN_PASSWORD, COMPANIES, seed
from app.analysis import build_timeline
from app.main import APPLICANTS, ROLE_LIBRARY, STORE, app
from app.matchrun import RUNS
from app.schemas import Assessment, JobBrief, Requirement
from tests.fixtures import sample_profile

HTML = {"accept": "text/html"}
CV_BYTES = (
    b"Priya Raghavan\nPrincipal Machine Learning Engineer\n"
    b"Built RAG systems on Kubernetes with PyTorch and Kubeflow.\n"
    b"Jan 2018 - Present, Cognizant, Hyderabad.\n"
)


@pytest.fixture(autouse=True)
def clean_state():
    """Each journey starts from a known board."""
    ROLE_LIBRARY.clear()
    APPLICANTS.clear()
    RUNS.clear()
    STORE.clear()
    COMPANIES.clear()
    seed(force=True)
    yield


@pytest.fixture
def stub_llm(monkeypatch, inline_threads):
    from app import matchrun

    def fake_brief(jd_text, usage=None, model=None):
        return JobBrief(role_title="Stub", stated_min_years=3, requirements=[
            Requirement(text="Kubernetes", kind="must_have", category="technical"),
            Requirement(text="PyTorch", kind="must_have", category="technical"),
        ])

    def fake_profile(text, usage=None, model=None):
        return sample_profile()

    def fake_assess(**kw):
        return Assessment(executive_summary="Summary.", fit_rationale="Rationale.")

    monkeypatch.setattr(matchrun.llm, "extract_job_brief", fake_brief)
    monkeypatch.setattr(matchrun.llm, "extract_profile", fake_profile)
    monkeypatch.setattr(matchrun.llm, "assess", fake_assess)


def signed_in(email, password=ADMIN_PASSWORD):
    c = TestClient(app)
    r = c.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, "sign-in failed for {}".format(email)
    return c


def post_role(client, title, **fields):
    data = {"role_title": title}
    data.update({k: str(v) for k, v in fields.items()})
    r = client.post("/post-role", data=data, follow_redirects=False)
    assert r.status_code == 303
    return next(rid for rid, rc in ROLE_LIBRARY.items() if rc.role_title == title)


def apply_as(name, **fields):
    c = TestClient(app)
    data = {"full_name": name}
    data.update({k: str(v) for k, v in fields.items()})
    r = c.post("/apply", data=data, files={"cv": (name + " CV.txt", CV_BYTES)}, headers=HTML)
    assert r.status_code == 200 and "Application received" in r.text
    return c


# ==========================================================================
# The full journey
# ==========================================================================


def test_register_post_apply_match_and_see_your_own_shortlist(stub_llm):
    # 1. An employer registers from the landing page.
    employer = TestClient(app)
    r = employer.post("/register", data={
        "name": "Kestrel Labs", "email": "talent@kestrel.example",
        "password": "password1", "confirm": "password1",
        "industry": "FinTech", "location": "Bengaluru",
    }, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/dashboard"

    # 2. They post a role.
    role_id = post_role(employer, "ML Platform Lead", location="Bengaluru",
                        min_years=4, ctc_min_lpa=40, ctc_max_lpa=90,
                        max_notice_days=90, must_have_skills="Kubernetes, PyTorch")
    assert ROLE_LIBRARY[role_id].company_id is not None

    # 3. A candidate applies through the public form.
    apply_as("Priya Raghavan", years_experience=9, current_location="Bengaluru",
             notice_period_days=60, expected_ctc_lpa=70, min_acceptable_ctc_lpa=60)
    assert len(APPLICANTS) == 1

    # 4. The agency runs matching.
    admin = signed_in(ADMIN_EMAIL)
    r = admin.post("/roles/match", data={"anonymise": "on"}, follow_redirects=False)
    assert r.status_code == 303
    run = list(RUNS.values())[-1]
    assert run.assessed_pairs, "the match produced no assessments"

    # 5. The employer sees the candidate on their own dashboard.
    body = employer.get("/dashboard", headers=HTML).text
    assert "Candidates matched to your roles" in body
    assert "ML Platform Lead" in body

    # 6. And can open that dossier.
    dossier_id = run.assessed_pairs[0].dossier_id
    assert employer.get("/dossier/{}".format(dossier_id), headers=HTML).status_code == 200


def test_an_employer_cannot_open_another_companys_candidate(stub_llm):
    a = signed_in("hiring@fintrail.example")
    b = signed_in("careers@alderline.example")
    post_role(a, "Fintrail Role", min_years=2, must_have_skills="Kubernetes")
    apply_as("Some Candidate", years_experience=9)

    signed_in(ADMIN_EMAIL).post("/roles/match", data={"anonymise": "on"}, follow_redirects=False)
    run = list(RUNS.values())[-1]
    dossier_id = run.assessed_pairs[0].dossier_id

    assert a.get("/dossier/{}".format(dossier_id), headers=HTML).status_code == 200
    r = b.get("/dossier/{}".format(dossier_id), headers=HTML)
    assert r.status_code == 400
    assert "not on one of your shortlists" in r.text


def test_an_employer_cannot_lift_the_blind_with_a_query_string(stub_llm):
    """Identity is released by the agency, not by editing a URL."""
    employer = signed_in("hiring@fintrail.example")
    post_role(employer, "Blind Test Role", min_years=2, must_have_skills="Kubernetes")
    apply_as("Arjun Menon", years_experience=9)
    signed_in(ADMIN_EMAIL).post("/roles/match", data={"anonymise": "on"}, follow_redirects=False)

    dossier_id = list(RUNS.values())[-1].assessed_pairs[0].dossier_id
    body = employer.get("/dossier/{}?blind=0".format(dossier_id), headers=HTML).text.lower()
    assert "arjun" not in body and "menon" not in body

    # The agency can.
    admin_body = signed_in(ADMIN_EMAIL).get("/dossier/{}?blind=0".format(dossier_id), headers=HTML).text
    assert "Arjun" in admin_body


# ==========================================================================
# Unauthenticated access
# ==========================================================================


@pytest.mark.parametrize("path", [
    "/dashboard", "/roles", "/post-role", "/candidates", "/shortlists", "/workspace",
])
def test_private_pages_redirect_anonymous_visitors(path):
    r = TestClient(app).get(path, headers=HTML, follow_redirects=False)
    assert r.status_code == 303 and "/login" in r.headers["location"]


def test_a_dossier_url_is_useless_without_a_session(stub_llm):
    """Regression: these routes had no guard, so a URL alone exposed a
    candidate's name, email, phone, salary and full CV text."""
    employer = signed_in("hiring@fintrail.example")
    post_role(employer, "Leak Test", min_years=2, must_have_skills="Kubernetes")
    apply_as("Arjun Menon", years_experience=9)
    signed_in(ADMIN_EMAIL).post("/roles/match", data={"anonymise": "on"}, follow_redirects=False)
    dossier_id = list(RUNS.values())[-1].assessed_pairs[0].dossier_id

    anon = TestClient(app)
    for suffix in ["", "?blind=0", "/embed", "/embed?blind=0", "/pdf", "/pdf?blind=0"]:
        r = anon.get("/dossier/{}{}".format(dossier_id, suffix), headers=HTML, follow_redirects=False)
        assert r.status_code == 303, suffix
        assert "arjun" not in r.text.lower(), suffix


def test_status_endpoints_do_not_leak_candidate_names(stub_llm):
    post_role(signed_in("hiring@fintrail.example"), "Status Test", min_years=2,
              must_have_skills="Kubernetes")
    apply_as("Priya Raghavan", years_experience=9)
    signed_in(ADMIN_EMAIL).post("/roles/match", data={"anonymise": "on"}, follow_redirects=False)
    run_id = list(RUNS.keys())[-1]

    r = TestClient(app).get("/match/{}/status".format(run_id), follow_redirects=False)
    assert r.status_code == 303
    assert "priya" not in r.text.lower()


def test_state_changing_demo_routes_need_an_account():
    anon = TestClient(app)
    before = len(STORE)
    for path in ["/demo", "/match/demo"]:
        assert anon.post(path, follow_redirects=False).status_code == 303
    assert len(STORE) == before, "an anonymous visitor grew the server store"


def test_api_docs_are_not_public():
    anon = TestClient(app)
    for path in ["/docs", "/redoc", "/openapi.json"]:
        assert anon.get(path).status_code == 404, path


# ==========================================================================
# Edge cases
# ==========================================================================


def test_matching_with_no_applicants_says_so():
    post_role(signed_in("hiring@fintrail.example"), "Lonely Role")
    r = signed_in(ADMIN_EMAIL).post("/roles/match", data={}, headers=HTML)
    assert r.status_code == 400 and "No applicants" in r.text


def test_matching_with_no_roles_says_so():
    apply_as("Nobody Wants Me")
    r = signed_in(ADMIN_EMAIL).post("/roles/match", data={}, headers=HTML)
    assert r.status_code == 400 and "No open roles" in r.text


def test_a_brand_new_employer_sees_an_empty_state_not_an_error():
    c = TestClient(app)
    c.post("/register", data={"name": "Empty Co", "email": "e@empty.example",
                              "password": "password1", "confirm": "password1"},
           follow_redirects=False)
    body = c.get("/dashboard", headers=HTML).text
    assert body.count("No roles posted") or "No matches yet" in body
    assert "Traceback" not in body


def test_unicode_survives_the_whole_journey(stub_llm):
    employer = signed_in("jobs@qadira.example")
    post_role(employer, "Ingénieur ML — Sénior", location="Dubaï",
              min_years=2, must_have_skills="Kubernetes")
    apply_as("Zoë Müller-Nakamura", years_experience=9)
    signed_in(ADMIN_EMAIL).post("/roles/match", data={"anonymise": "on"}, follow_redirects=False)

    body = employer.get("/dashboard", headers=HTML).text
    assert "Ingénieur" in body


def test_a_very_long_role_title_does_not_break_a_page():
    employer = signed_in("hiring@fintrail.example")
    post_role(employer, "X" * 400)
    r = employer.get("/roles", headers=HTML)
    assert r.status_code == 200 and "Traceback" not in r.text


def test_html_in_a_role_title_is_escaped():
    employer = signed_in("hiring@fintrail.example")
    post_role(employer, "<script>alert(1)</script>Lead")
    body = employer.get("/roles", headers=HTML).text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_a_role_with_no_title_is_refused():
    r = signed_in("hiring@fintrail.example").post(
        "/post-role", data={"role_title": "   "}, headers=HTML)
    assert r.status_code == 400 and "needs a title" in r.text


def test_deleting_a_role_that_does_not_exist_is_harmless():
    c = signed_in("hiring@fintrail.example")
    r = c.post("/roles/nonexistent/delete", follow_redirects=False)
    assert r.status_code == 303


def test_the_landing_page_counts_reflect_reality():
    post_role(signed_in("hiring@fintrail.example"), "Counted Role")
    body = TestClient(app).get("/", headers=HTML).text
    assert "Counted Role" in body


def test_signing_in_again_does_not_stack_sessions():
    c = signed_in(ADMIN_EMAIL)
    r = c.get("/login", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/dashboard"


def test_login_next_parameter_cannot_redirect_off_site():
    """An open redirect turns a login link into a phishing primitive."""
    c = TestClient(app)
    r = c.post("/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD,
                               "next": "https://evil.example/steal"},
               follow_redirects=False)
    assert r.status_code == 303
    assert not r.headers["location"].startswith("http"), r.headers["location"]
