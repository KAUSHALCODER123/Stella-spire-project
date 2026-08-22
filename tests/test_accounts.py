"""Accounts, the single sign-in, and tenant isolation.

The rule worth protecting: one login path, and the account decides what is
visible. An employer must never see another employer's openings, and must
never reach the agency's parsing workspace.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.accounts import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    COMPANIES,
    authenticate,
    create_company,
    find_by_email,
    hash_password,
    seed,
    verify_password,
)
from app.main import ROLE_LIBRARY, app

HTML = {"accept": "text/html"}


# --- password handling -----------------------------------------------------


def test_hash_is_salted_so_two_identical_passwords_differ():
    assert hash_password("hunter2") != hash_password("hunter2")


def test_a_correct_password_verifies():
    assert verify_password("hunter2", hash_password("hunter2"))


@pytest.mark.parametrize("wrong", ["Hunter2", "hunter", "hunter22", "", " hunter2"])
def test_a_wrong_password_does_not(wrong):
    assert not verify_password(wrong, hash_password("hunter2"))


def test_a_corrupt_stored_hash_fails_closed():
    for junk in ["", "nonsense", "nosalt$", "$nohash", "zz$zz"]:
        assert not verify_password("hunter2", junk)


def test_the_plaintext_password_is_never_stored():
    company = create_company(name="T", email="pw@test.example", password="s3cretpassword")
    assert "s3cretpassword" not in company.password_hash
    COMPANIES.pop(company.id, None)


# --- accounts --------------------------------------------------------------


def test_seeding_creates_an_admin_and_some_employers():
    seed()
    admin = find_by_email(ADMIN_EMAIL)
    assert admin is not None and admin.is_admin
    assert sum(1 for c in COMPANIES.values() if not c.is_admin) >= 3


def test_email_lookup_ignores_case_and_padding():
    seed()
    assert find_by_email("  ADMIN  ") is not None


def test_authenticate_accepts_the_admin_credentials():
    seed()
    company = authenticate(ADMIN_EMAIL, ADMIN_PASSWORD)
    assert company is not None and company.is_admin


def test_authenticate_rejects_a_wrong_password():
    seed()
    assert authenticate(ADMIN_EMAIL, "nope") is None


def test_authenticate_rejects_an_unknown_account():
    assert authenticate("ghost@nowhere.example", "anything") is None


def test_initials_survive_odd_names():
    seed()
    for name in ["Acme", "Acme Corp Ltd", "  "]:
        c = create_company(name=name, email="i{}@t.example".format(len(name)), password="password1")
        assert len(c.initials) <= 2
        COMPANIES.pop(c.id, None)


# --- the web flow ----------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def signed_in(email, password=ADMIN_PASSWORD):
    c = TestClient(app)
    r = c.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, r.status_code
    return c


def test_landing_is_public(client):
    r = client.get("/", headers=HTML)
    assert r.status_code == 200
    assert "Post a job" in r.text and "Apply for a role" in r.text


def test_application_form_is_public(client):
    assert client.get("/apply", headers=HTML).status_code == 200


def test_dashboard_requires_sign_in(client):
    r = client.get("/dashboard", headers=HTML, follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in r.headers["location"]


def test_wrong_password_gives_one_generic_message(client):
    """Distinguishing 'no such account' from 'wrong password' leaks which
    addresses are registered."""
    r = client.post("/login", data={"email": ADMIN_EMAIL, "password": "wrong"}, headers=HTML)
    unknown = client.post("/login", data={"email": "ghost@x.example", "password": "wrong"}, headers=HTML)
    assert r.status_code == 401 and unknown.status_code == 401
    assert "did not match an account" in r.text
    assert "did not match an account" in unknown.text


def test_admin_sees_the_parsing_workspace():
    body = signed_in(ADMIN_EMAIL).get("/dashboard", headers=HTML).text
    assert "Analysis settings" in body or "Upload CVs" in body
    assert "/candidates" in body


def test_employer_sees_their_own_dashboard():
    body = signed_in("hiring@fintrail.example").get("/dashboard", headers=HTML).text
    assert "Candidates matched to your roles" in body
    assert "Fintrail" in body


def test_employer_cannot_reach_admin_surfaces():
    c = signed_in("hiring@fintrail.example")
    for path in ["/candidates", "/shortlists"]:
        r = c.get(path, headers=HTML)
        assert r.status_code == 400, path
        assert "recruitment team" in r.text


def test_admin_can_reach_admin_surfaces():
    c = signed_in(ADMIN_EMAIL)
    for path in ["/candidates", "/shortlists", "/roles"]:
        assert c.get(path, headers=HTML).status_code == 200, path


def test_signing_out_ends_the_session():
    c = signed_in(ADMIN_EMAIL)
    assert c.get("/dashboard", headers=HTML, follow_redirects=False).status_code == 200
    c.post("/logout", follow_redirects=False)
    assert c.get("/dashboard", headers=HTML, follow_redirects=False).status_code == 303


# --- registration ----------------------------------------------------------


@pytest.mark.parametrize("payload,expected", [
    ({"name": "", "email": "a@b.com", "password": "password1", "confirm": "password1"}, "needs a name"),
    ({"name": "X", "email": "notanemail", "password": "password1", "confirm": "password1"}, "valid email"),
    ({"name": "X", "email": "a@b.com", "password": "short", "confirm": "short"}, "at least 8"),
    ({"name": "X", "email": "a@b.com", "password": "password1", "confirm": "password2"}, "did not match"),
])
def test_registration_validation(client, payload, expected):
    r = client.post("/register", data=payload, headers=HTML)
    assert r.status_code == 400
    assert expected in r.text


def test_duplicate_email_is_refused(client):
    seed()
    r = client.post("/register", data={
        "name": "Impostor", "email": "hiring@fintrail.example",
        "password": "password1", "confirm": "password1"}, headers=HTML)
    assert r.status_code == 400 and "already exists" in r.text


def test_registering_signs_you_straight_in():
    c = TestClient(app)
    r = c.post("/register", data={
        "name": "Brand New Ltd", "email": "new@newco.example",
        "password": "password1", "confirm": "password1"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/dashboard"
    assert "Brand New Ltd" in c.get("/dashboard", headers=HTML).text


# --- tenant isolation ------------------------------------------------------


def test_an_employer_only_sees_their_own_roles():
    a = signed_in("hiring@fintrail.example")
    b = signed_in("careers@alderline.example")
    a.post("/post-role", data={"role_title": "Isolation Role A"}, follow_redirects=False)
    b.post("/post-role", data={"role_title": "Isolation Role B"}, follow_redirects=False)

    a_body = a.get("/roles", headers=HTML).text
    b_body = b.get("/roles", headers=HTML).text

    assert "Isolation Role A" in a_body and "Isolation Role B" not in a_body
    assert "Isolation Role B" in b_body and "Isolation Role A" not in b_body


def test_the_admin_sees_every_role():
    body = signed_in(ADMIN_EMAIL).get("/roles", headers=HTML).text
    assert "Isolation Role A" in body and "Isolation Role B" in body


def test_a_posted_role_records_its_owner():
    c = signed_in("people@northwind.example")
    c.post("/post-role", data={"role_title": "Owner Tagged Role"}, follow_redirects=False)
    owner = find_by_email("people@northwind.example")
    tagged = [rc for rc in ROLE_LIBRARY.values() if rc.role_title == "Owner Tagged Role"]
    assert tagged and tagged[0].company_id == owner.id


def test_an_employer_cannot_delete_another_companys_role():
    a = signed_in("hiring@fintrail.example")
    b = signed_in("careers@alderline.example")
    a.post("/post-role", data={"role_title": "Do Not Delete Me"}, follow_redirects=False)
    victim = next(rid for rid, rc in ROLE_LIBRARY.items() if rc.role_title == "Do Not Delete Me")

    b.post("/roles/{}/delete".format(victim), follow_redirects=False)
    assert victim in ROLE_LIBRARY, "another company deleted a role that was not theirs"

    a.post("/roles/{}/delete".format(victim), follow_redirects=False)
    assert victim not in ROLE_LIBRARY, "the owner should be able to delete it"


def test_posting_a_role_requires_sign_in():
    c = TestClient(app)
    r = c.post("/post-role", data={"role_title": "Anonymous Role"}, follow_redirects=False)
    assert r.status_code == 303 and "/login" in r.headers["location"]
    assert not any(rc.role_title == "Anonymous Role" for rc in ROLE_LIBRARY.values())
