"""Persistence over Supabase Postgres, via PostgREST.

Everything in this application lived in process memory: accounts, roles,
applicants, dossiers. A restart logged everyone out and turned every shared
link into a 404. This module makes the parts that cannot be regenerated
survive.

What is stored, and what is not:

    stored      accounts, roles, applicants, and the three model outputs of a
                dossier (profile, brief, assessment) plus the CV text
    recomputed  the timeline, the computed risk flags, and quote verification

The second list is deterministic given the first. Persisting it would create
two sources of truth that drift the moment app/analysis.py changes, so a
dossier is rehydrated by re-deriving them. Cheap, and always consistent with
the code that produces them.

Not configured, or the tables missing, means the app falls back to memory and
says so. A demo without a database should still run.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

TIMEOUT = 20
TABLES = ("companies", "roles", "applicants", "dossiers")


class Database:
    """The slice of PostgREST this application needs."""

    def __init__(self, url: str, key: str) -> None:
        self.base = "{}/rest/v1".format(url.rstrip("/"))
        self.key = key
        self.available = False
        self.reason: Optional[str] = None

    # --- plumbing ---------------------------------------------------------

    def _headers(self, extra: Optional[dict] = None) -> dict:
        headers = {
            "apikey": self.key,
            "Authorization": "Bearer {}".format(self.key),
            "Content-Type": "application/json",
        }
        headers.update(extra or {})
        return headers

    def _request(self, method: str, path: str, body: Any = None, extra: Optional[dict] = None):
        request = urllib.request.Request(
            self.base + path,
            method=method,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers=self._headers(extra),
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = response.read()
            return json.loads(payload) if payload else None

    # --- lifecycle --------------------------------------------------------

    def connect(self) -> bool:
        """Check every table before claiming the database is usable.

        A partial migration is worse than none: the app would appear to work
        and lose whichever entity was missing.
        """
        missing = []
        for table in TABLES:
            try:
                self._request("GET", "/{}?select=id&limit=1".format(table))
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 406):
                    missing.append(table)
                else:
                    self.reason = "HTTP {} on {}".format(exc.code, table)
                    return False
            except Exception as exc:  # noqa: BLE001
                self.reason = "{}: {}".format(type(exc).__name__, exc)
                return False

        if missing:
            self.reason = "tables not created: {} (run migrations/001_init.sql)".format(
                ", ".join(missing))
            return False

        self.available = True
        return True

    # --- generic operations ----------------------------------------------

    def upsert(self, table: str, row: dict) -> None:
        self._request("POST", "/{}".format(table), [row],
                      {"Prefer": "resolution=merge-duplicates,return=minimal"})

    def select(self, table: str, query: str = "select=*") -> List[dict]:
        return self._request("GET", "/{}?{}".format(table, query)) or []

    def delete(self, table: str, column: str, value: str) -> None:
        self._request("DELETE", "/{}?{}=eq.{}".format(
            table, column, urllib.parse.quote(str(value), safe="")))

    def count(self, table: str) -> int:
        return len(self.select(table, "select=id"))


_db: Optional[Database] = None
_checked = False


def get_db() -> Optional[Database]:
    """The shared connection, or None when running on memory alone."""
    global _db, _checked
    if _checked:
        return _db if (_db and _db.available) else None

    _checked = True
    from app.config import settings

    if not (settings.supabase_url and settings.supabase_key):
        log.info("no database configured; state is in memory only")
        return None

    candidate = Database(settings.supabase_url, settings.supabase_key)
    if candidate.connect():
        _db = candidate
        log.info("database connected")
        return _db

    log.warning("database unavailable (%s); state is in memory only", candidate.reason)
    _db = candidate
    return None


def status() -> dict:
    db = get_db()
    if db is not None:
        return {"connected": True, "detail": "supabase postgres"}
    from app.config import settings
    if not (settings.supabase_url and settings.supabase_key):
        return {"connected": False, "detail": "not configured"}
    return {"connected": False, "detail": _db.reason if _db else "unavailable"}


def reset_for_tests() -> None:
    global _db, _checked
    _db, _checked = None, False


# ==========================================================================
# Repositories
# ==========================================================================


def save_company(company) -> None:
    db = get_db()
    if db is None:
        return
    try:
        db.upsert("companies", {
            "id": company.id, "name": company.name, "email": company.email,
            "password_hash": company.password_hash, "is_admin": company.is_admin,
            "industry": company.industry, "location": company.location,
            "website": company.website,
        })
    except Exception as exc:  # noqa: BLE001 - never fail a signup on a write
        log.warning("could not persist company %s: %s", company.email, exc)


def load_companies() -> List[dict]:
    db = get_db()
    if db is None:
        return []
    try:
        return db.select("companies", "select=*&order=created_at.asc")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load companies: %s", exc)
        return []


def save_role(role_id: str, role) -> None:
    db = get_db()
    if db is None:
        return
    try:
        db.upsert("roles", {
            "id": role_id,
            "company_id": role.company_id,
            "payload": json.loads(role.model_dump_json()),
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("could not persist role %s: %s", role_id, exc)


def delete_role(role_id: str) -> None:
    db = get_db()
    if db is None:
        return
    try:
        db.delete("roles", "id", role_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not delete role %s: %s", role_id, exc)


def load_roles() -> List[tuple]:
    db = get_db()
    if db is None:
        return []
    from app.intake import RoleConstraints
    out = []
    try:
        rows = db.select("roles", "select=*&order=created_at.asc")
    except Exception as exc:  # noqa: BLE001 - a read failure must not kill startup
        log.warning("could not load roles: %s", exc)
        return []
    for row in rows:
        try:
            out.append((row["id"], RoleConstraints(**row["payload"])))
        except Exception as exc:  # noqa: BLE001 - one bad row must not hide the rest
            log.warning("skipping unreadable role %s: %s", row.get("id"), exc)
    return out


def save_applicant(applicant_id: str, record: dict) -> None:
    db = get_db()
    if db is None:
        return
    stored = record.get("stored")
    try:
        db.upsert("applicants", {
            "id": applicant_id,
            "filename": record["filename"],
            "prefs": json.loads(record["prefs"].model_dump_json()),
            "storage_key": getattr(stored, "key", None),
            "local_path": str(record["path"]),
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("could not persist applicant %s: %s", applicant_id, exc)


def load_applicants() -> List[tuple]:
    db = get_db()
    if db is None:
        return []
    from pathlib import Path

    from app.intake import CandidatePreferences
    out = []
    try:
        rows = db.select("applicants", "select=*&order=created_at.asc")
    except Exception as exc:  # noqa: BLE001 - a read failure must not kill startup
        log.warning("could not load applicants: %s", exc)
        return []
    for row in rows:
        try:
            out.append((row["id"], {
                "prefs": CandidatePreferences(**row["prefs"]),
                "path": Path(row["local_path"]) if row.get("local_path") else None,
                "filename": row["filename"],
                "storage_key": row.get("storage_key"),
                "stored": None,
            }))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping unreadable applicant %s: %s", row.get("id"), exc)
    return out


def save_dossier(dossier_id: str, dossier) -> None:
    """Persist only the irreducible parts of a dossier."""
    db = get_db()
    if db is None:
        return
    try:
        db.upsert("dossiers", {
            "id": dossier_id,
            "profile": json.loads(dossier.profile.model_dump_json()),
            "brief": json.loads(dossier.brief.model_dump_json()),
            "assessment": json.loads(dossier.assessment.model_dump_json()),
            "document": {
                "text": dossier.document.text,
                "page_count": dossier.document.page_count,
                "source_format": dossier.document.source_format,
                "filename": dossier.document.filename,
                "warnings": list(dossier.document.warnings),
            },
            "brief_text": dossier.brief_text or "",
            "model": dossier.model,
            "anonymise": bool(getattr(dossier, "anonymise", True)),
            "usage": {"input_tokens": dossier.usage.input_tokens,
                      "output_tokens": dossier.usage.output_tokens,
                      "calls": dossier.usage.calls},
            "elapsed": dossier.elapsed_seconds,
            "warnings": list(dossier.warnings),
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("could not persist dossier %s: %s", dossier_id, exc)


def _rehydrate(row: dict):
    """Rebuild a Dossier, re-deriving everything that is deterministic."""
    from app.analysis import build_timeline, derive_risk_flags, sort_flags
    from app.extract.documents import DocumentText
    from app.extract.llm import Usage
    from app.pipeline import Dossier
    from app.schemas import Assessment, CandidateProfile, JobBrief
    from app.verify import verify_assessment

    profile = CandidateProfile(**row["profile"])
    brief = JobBrief(**row["brief"])
    assessment = Assessment(**row["assessment"])
    doc_row = row["document"]
    document = DocumentText(
        text=doc_row.get("text", ""), page_count=doc_row.get("page_count", 0),
        source_format=doc_row.get("source_format", "txt"),
        filename=doc_row.get("filename", ""), warnings=list(doc_row.get("warnings") or []),
    )
    usage_row = row.get("usage") or {}
    timeline = build_timeline(profile)

    dossier = Dossier(
        profile=profile, timeline=timeline, brief=brief, assessment=assessment,
        flags=sort_flags(derive_risk_flags(profile, timeline, document.text) + assessment.risk_flags),
        document=document,
        usage=Usage(input_tokens=usage_row.get("input_tokens", 0),
                    output_tokens=usage_row.get("output_tokens", 0),
                    calls=usage_row.get("calls", 0)),
        brief_text=row.get("brief_text") or "",
        verification=verify_assessment(assessment, document.text, row.get("brief_text") or ""),
        model=row.get("model") or "",
        elapsed_seconds=row.get("elapsed") or 0.0,
        warnings=list(row.get("warnings") or []),
    )
    dossier.anonymise = bool(row.get("anonymise", True))
    return dossier


def load_dossiers(limit: int = 200) -> List[tuple]:
    db = get_db()
    if db is None:
        return []
    out = []
    try:
        rows = db.select("dossiers", "select=*&order=created_at.asc&limit={}".format(limit))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load dossiers: %s", exc)
        return []
    for row in rows:
        try:
            out.append((row["id"], _rehydrate(row)))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping unreadable dossier %s: %s", row.get("id"), exc)
    return out
