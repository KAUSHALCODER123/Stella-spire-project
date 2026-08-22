"""Persistence.

The interesting property is not "it writes a row" but the round trip: a
dossier is stored as three model payloads plus the CV text, and everything
deterministic is re-derived on read. If that derivation drifts from the live
code, a restored dossier would quietly disagree with a fresh one.
"""

from __future__ import annotations

import pytest

from app import db as db_mod


class FakeDB:
    """An in-process stand-in for PostgREST."""

    def __init__(self):
        self.tables = {t: {} for t in db_mod.TABLES}
        self.available = True
        self.reason = None
        self.writes = 0

    def upsert(self, table, row):
        self.writes += 1
        self.tables[table][row["id"]] = dict(row)

    def select(self, table, query="select=*"):
        return list(self.tables[table].values())

    def delete(self, table, column, value):
        self.tables[table].pop(value, None)

    def count(self, table):
        return len(self.tables[table])


@pytest.fixture
def fake_db(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(db_mod, "get_db", lambda: fake)
    return fake


# --- companies -------------------------------------------------------------


def test_a_company_round_trips(fake_db):
    from app.accounts import COMPANIES, Company, load_from_db

    company = Company(id="c1", name="Acme", email="a@acme.example",
                      password_hash="salt$hash", is_admin=False,
                      industry="FinTech", location="Bengaluru")
    db_mod.save_company(company)
    COMPANIES.clear()

    assert load_from_db() == 1
    restored = COMPANIES["c1"]
    assert restored.name == "Acme"
    assert restored.email == "a@acme.example"
    assert restored.password_hash == "salt$hash"
    assert restored.industry == "FinTech"


def test_the_admin_flag_survives(fake_db):
    from app.accounts import COMPANIES, Company, load_from_db

    db_mod.save_company(Company(id="c2", name="Agency", email="admin",
                                password_hash="x$y", is_admin=True))
    COMPANIES.clear()
    load_from_db()
    assert COMPANIES["c2"].is_admin is True


# --- roles -----------------------------------------------------------------


def test_a_role_round_trips_with_every_field(fake_db):
    from app.intake import RoleConstraints

    role = RoleConstraints(
        role_title="GenAI Platform Lead", client_name="GCC", company_id="c1",
        location="Hyderabad", work_mode="hybrid", min_years=9, max_years=16,
        ctc_min_lpa=85, ctc_max_lpa=115, max_notice_days=60, domain="Insurance",
        must_have_skills=["RAG", "Kubernetes"], nice_to_have_skills=["FinOps"],
        notes="0-to-1 build")
    db_mod.save_role("r1", role)

    restored = dict(db_mod.load_roles())["r1"]
    assert restored.model_dump() == role.model_dump()


def test_deleting_a_role_removes_it(fake_db):
    from app.intake import RoleConstraints

    db_mod.save_role("r1", RoleConstraints(role_title="Gone", company_id="c1"))
    db_mod.delete_role("r1")
    assert db_mod.load_roles() == []


def test_one_unreadable_role_does_not_hide_the_others(fake_db):
    """A schema change must not make the whole board vanish."""
    from app.intake import RoleConstraints

    db_mod.save_role("good", RoleConstraints(role_title="Fine", company_id="c1"))
    fake_db.tables["roles"]["bad"] = {"id": "bad", "company_id": "c1",
                                      "payload": {"nonsense": True}}
    loaded = dict(db_mod.load_roles())
    assert "good" in loaded and "bad" not in loaded


# --- applicants ------------------------------------------------------------


def test_an_applicant_round_trips(fake_db, tmp_path):
    from app.intake import CandidatePreferences

    path = tmp_path / "cv.txt"
    path.write_text("cv", encoding="utf-8")
    record = {
        "prefs": CandidatePreferences(full_name="Priya", notice_period_days=60,
                                      min_acceptable_ctc_lpa=88, years_experience=11.9),
        "path": path, "filename": "Priya CV.txt", "stored": None,
    }
    db_mod.save_applicant("a1", record)

    restored = dict(db_mod.load_applicants())["a1"]
    assert restored["filename"] == "Priya CV.txt"
    assert restored["prefs"].full_name == "Priya"
    assert restored["prefs"].min_acceptable_ctc_lpa == 88
    assert str(restored["path"]) == str(path)


def test_the_storage_key_is_kept_so_a_lost_local_file_is_recoverable(fake_db, tmp_path):
    from app.intake import CandidatePreferences
    from app.storage import StoredFile

    path = tmp_path / "cv.txt"
    path.write_text("cv", encoding="utf-8")
    stored = StoredFile(local_path=path, filename="cv.txt", bucket="resumes", key="abc_cv.txt")
    db_mod.save_applicant("a1", {"prefs": CandidatePreferences(), "path": path,
                                 "filename": "cv.txt", "stored": stored})
    assert dict(db_mod.load_applicants())["a1"]["storage_key"] == "abc_cv.txt"


# --- dossiers: the round trip that matters ---------------------------------


def test_a_dossier_round_trips_and_rederives_its_computed_parts(fake_db):
    from tests.fixtures import sample_dossier

    original = sample_dossier()
    original.anonymise = True
    db_mod.save_dossier("d1", original)

    restored = dict(db_mod.load_dossiers())["d1"]

    # Stored verbatim.
    assert restored.profile.full_name == original.profile.full_name
    assert restored.brief.role_title == original.brief.role_title
    assert len(restored.assessment.requirement_matches) == len(original.assessment.requirement_matches)
    assert restored.document.text == original.document.text
    assert restored.model == original.model

    # Re-derived, and must agree with the original exactly.
    assert restored.timeline.total_experience_years == original.timeline.total_experience_years
    assert len(restored.timeline.gaps) == len(original.timeline.gaps)
    assert [f.kind for f in restored.flags] == [f.kind for f in original.flags]
    assert restored.verification.verified == original.verification.verified
    assert restored.verification.total == original.verification.total


def test_the_scores_shown_in_the_ui_survive_a_restart(fake_db):
    """A restored dossier must not quietly disagree with a fresh one."""
    from tests.fixtures import sample_dossier

    original = sample_dossier()
    db_mod.save_dossier("d1", original)
    restored = dict(db_mod.load_dossiers())["d1"]

    assert restored.suitability["percent"] == original.suitability["percent"]
    assert restored.match_counts == original.match_counts
    assert restored.experience_match == original.experience_match
    assert restored.skill_stats == original.skill_stats


def test_the_blind_setting_survives(fake_db):
    from tests.fixtures import sample_dossier

    d = sample_dossier()
    d.anonymise = False
    db_mod.save_dossier("d1", d)
    assert dict(db_mod.load_dossiers())["d1"].anonymise is False


def test_one_corrupt_dossier_does_not_lose_the_rest(fake_db):
    from tests.fixtures import sample_dossier

    db_mod.save_dossier("good", sample_dossier())
    fake_db.tables["dossiers"]["bad"] = {"id": "bad", "profile": {"positions": "not a list"},
                                         "brief": {}, "assessment": {}, "document": {}}
    loaded = dict(db_mod.load_dossiers())
    assert "good" in loaded and "bad" not in loaded


# --- degradation -----------------------------------------------------------


def test_without_a_database_every_write_is_a_silent_no_op(monkeypatch):
    """No database is a supported way to run, not an error path."""
    from app.accounts import Company
    from app.intake import CandidatePreferences, RoleConstraints
    from tests.fixtures import sample_dossier

    monkeypatch.setattr(db_mod, "get_db", lambda: None)
    db_mod.save_company(Company(id="c", name="n", email="e", password_hash="h"))
    db_mod.save_role("r", RoleConstraints(role_title="t"))
    db_mod.save_applicant("a", {"prefs": CandidatePreferences(), "path": "p",
                                "filename": "f", "stored": None})
    db_mod.save_dossier("d", sample_dossier())
    db_mod.delete_role("r")

    assert db_mod.load_companies() == []
    assert db_mod.load_roles() == []
    assert db_mod.load_applicants() == []
    assert db_mod.load_dossiers() == []


def test_a_write_failure_never_reaches_the_caller(fake_db, monkeypatch):
    """Losing durability is worth a log line; losing the user's action is not."""
    from app.accounts import Company

    def boom(*a, **k):
        raise RuntimeError("database on fire")

    monkeypatch.setattr(fake_db, "upsert", boom)
    db_mod.save_company(Company(id="c", name="n", email="e", password_hash="h"))


def test_a_read_failure_returns_empty_rather_than_raising(fake_db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("database on fire")

    monkeypatch.setattr(fake_db, "select", boom)
    # Every loader, not just the ones that happened to be written defensively:
    # a read failure at startup must degrade to memory, not crash the app.
    assert db_mod.load_companies() == []
    assert db_mod.load_roles() == []
    assert db_mod.load_applicants() == []
    assert db_mod.load_dossiers() == []


def test_status_reports_why_it_is_unavailable(monkeypatch):
    monkeypatch.setattr(db_mod, "get_db", lambda: None)
    from app.config import settings
    monkeypatch.setattr(settings, "supabase_url", "", raising=False)
    monkeypatch.setattr(settings, "supabase_key", "", raising=False)
    state = db_mod.status()
    assert state["connected"] is False
    assert "not configured" in state["detail"]
