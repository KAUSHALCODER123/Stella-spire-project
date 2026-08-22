"""Durable storage.

The rule these tests exist to protect: archiving is best-effort. A Supabase
outage, a missing bucket or a row-level-security policy that rejects the write
must never fail a candidate's application. The local copy still processes, the
failure is recorded, and the user is not punished for our infrastructure.
"""

from __future__ import annotations

import urllib.error

import pytest

from app.storage import StoredFile, SupabaseStorage, store_document


class FakeResponse:
    def __init__(self, status=200, body=b""):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def configured(monkeypatch):
    from app import storage as mod
    from app.config import settings
    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(settings, "supabase_key", "test-key")
    monkeypatch.setattr(settings, "supabase_bucket_resumes", "resumes")
    monkeypatch.setattr(settings, "supabase_bucket_jds", "jds")
    monkeypatch.setattr(mod, "_client", None)
    return mod


# --- the happy path --------------------------------------------------------


def test_a_resume_is_written_locally_and_archived(configured, monkeypatch, tmp_path):
    seen = {}

    def fake_open(request, timeout=None):
        seen["url"] = request.full_url
        seen["method"] = request.method
        seen["data"] = request.data
        seen["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return FakeResponse(200)

    monkeypatch.setattr(configured.urllib.request, "urlopen", fake_open)
    path = tmp_path / "abc123_cv.pdf"
    result = store_document(payload=b"PDF BYTES", filename="cv.pdf", kind="resume", local_path=path)

    assert path.read_bytes() == b"PDF BYTES", "the local working copy must always be written"
    assert result.archived
    assert result.bucket == "resumes"
    assert "/storage/v1/object/resumes/abc123_cv.pdf" in seen["url"]
    assert seen["method"] == "POST"
    assert seen["data"] == b"PDF BYTES"
    assert seen["headers"]["apikey"] == "test-key"


def test_a_job_description_goes_to_the_jds_bucket(configured, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(configured.urllib.request, "urlopen",
                        lambda r, timeout=None: (seen.update(url=r.full_url), FakeResponse(200))[1])
    result = store_document(payload=b"JD", filename="role.txt", kind="jd", local_path=tmp_path / "x_role.txt")
    assert result.bucket == "jds"
    assert "/object/jds/" in seen["url"]


def test_public_url_is_reported():
    client = SupabaseStorage("https://example.supabase.co", "k")
    assert client.public_url("resumes", "a.pdf") == \
        "https://example.supabase.co/storage/v1/object/public/resumes/a.pdf"


# --- failure must never reach the user -------------------------------------


def test_row_level_security_rejection_does_not_lose_the_upload(configured, monkeypatch, tmp_path):
    """The exact failure this project hit: an anon key without a storage policy."""
    def reject(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", {},
            __import__("io").BytesIO(b'{"statusCode":"403","message":"new row violates row-level security policy"}'))

    monkeypatch.setattr(configured.urllib.request, "urlopen", reject)
    path = tmp_path / "x_cv.pdf"
    result = store_document(payload=b"BYTES", filename="cv.pdf", kind="resume", local_path=path)

    assert path.read_bytes() == b"BYTES", "the application must still work"
    assert not result.archived
    assert "row-level security" in result.error


@pytest.mark.parametrize("boom", [
    TimeoutError("timed out"),
    ConnectionResetError("reset"),
    OSError("network unreachable"),
    ValueError("nonsense"),
])
def test_any_network_failure_is_swallowed(configured, monkeypatch, tmp_path, boom):
    def raise_it(request, timeout=None):
        raise boom
    monkeypatch.setattr(configured.urllib.request, "urlopen", raise_it)
    path = tmp_path / "x_cv.pdf"
    result = store_document(payload=b"B", filename="cv.pdf", kind="resume", local_path=path)
    assert path.exists() and not result.archived and result.error


# --- unconfigured ----------------------------------------------------------


def test_without_supabase_it_is_a_plain_local_write(monkeypatch, tmp_path):
    from app import storage as mod
    from app.config import settings
    monkeypatch.setattr(settings, "supabase_url", "")
    monkeypatch.setattr(settings, "supabase_key", "")
    monkeypatch.setattr(mod, "_client", None)

    path = tmp_path / "x_cv.pdf"
    result = store_document(payload=b"B", filename="cv.pdf", kind="resume", local_path=path)
    assert path.read_bytes() == b"B"
    assert not result.archived
    assert result.error is None, "not configured is not an error"


def test_nested_local_directories_are_created(configured, monkeypatch, tmp_path):
    monkeypatch.setattr(configured.urllib.request, "urlopen", lambda r, timeout=None: FakeResponse(200))
    path = tmp_path / "deep" / "deeper" / "cv.pdf"
    store_document(payload=b"B", filename="cv.pdf", kind="resume", local_path=path)
    assert path.exists()


# --- diagnostics -----------------------------------------------------------


def test_check_reports_an_anon_key_as_usable(configured, monkeypatch):
    """Listing buckets is an admin operation; failing it does not mean uploads fail."""
    def forbidden(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {},
                                     __import__("io").BytesIO(b"{}"))
    monkeypatch.setattr(configured.urllib.request, "urlopen", forbidden)
    result = SupabaseStorage("https://example.supabase.co", "k").check()
    assert result["ok"] is True
    assert "anon key" in result["detail"]


def test_check_reports_a_real_outage_as_broken(configured, monkeypatch):
    def down(request, timeout=None):
        raise OSError("connection refused")
    monkeypatch.setattr(configured.urllib.request, "urlopen", down)
    result = SupabaseStorage("https://example.supabase.co", "k").check()
    assert result["ok"] is False


def test_check_reports_missing_configuration():
    assert SupabaseStorage("", "").check()["ok"] is False


# Characters RFC 3986 forbids in a URL path. Any of these surviving into a
# request means the key was not encoded.
ILLEGAL_IN_URL = ' "<>^`{|}' + chr(92)


# --- URL encoding ----------------------------------------------------------


def test_a_filename_with_spaces_produces_a_valid_url(configured, monkeypatch, tmp_path):
    """Regression: "Meera Ramanathan CV.txt" is a legal filename and an illegal URL.

    urllib raises InvalidURL on the raw space, so every real CV -- almost all
    of which have spaces in the name -- silently failed to archive while the
    space-free probe passed.
    """
    seen = {}

    def fake_open(request, timeout=None):
        seen["url"] = request.full_url
        return FakeResponse(200)

    monkeypatch.setattr(configured.urllib.request, "urlopen", fake_open)
    path = tmp_path / "ff9d_Meera Ramanathan CV.txt"
    result = store_document(payload=b"B", filename="Meera Ramanathan CV.txt",
                            kind="resume", local_path=path)

    assert result.archived, result.error
    assert " " not in seen["url"], seen["url"]
    assert "%20" in seen["url"]


@pytest.mark.parametrize("name", [
    "Meera Ramanathan CV.txt",
    "CV (final) v2.pdf",
    "Résumé Ann.pdf",
    "cv#1&2.docx",
    "cv+plus.txt",
])
def test_awkward_filenames_all_produce_valid_urls(configured, monkeypatch, tmp_path, name):
    seen = {}
    monkeypatch.setattr(configured.urllib.request, "urlopen",
                        lambda r, timeout=None: (seen.update(url=r.full_url), FakeResponse(200))[1])
    from app.uploads import safe_filename
    path = tmp_path / ("aaaa_" + safe_filename(name))
    result = store_document(payload=b"B", filename=name, kind="resume", local_path=path)
    assert result.archived, result.error
    url = seen["url"]
    assert not any(c in url for c in ILLEGAL_IN_URL), url


def test_public_url_is_encoded_too(configured):
    client = SupabaseStorage("https://example.supabase.co", "k")
    url = client.public_url("resumes", "Meera Ramanathan CV.txt")
    assert " " not in url and "%20" in url
    assert url.startswith("https://example.supabase.co/storage/v1/object/public/resumes/")
