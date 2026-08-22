"""Filename handling.

Regression origin: uploading "Resume: Senior Dev.pdf" crashed the server with
an unhandled OSError (500). Windows rejects several characters outright, and
`Path(name).name` alone is not a guard -- a name containing a slash is
silently truncated to its last segment rather than rejected, so
"resume</script>.txt" became "script>.txt", which still contained an illegal
character.
"""

from __future__ import annotations

import pytest

from app.uploads import display_filename, safe_filename, upload_suffix

WINDOWS_ILLEGAL = '<>:"/\\|?*'


# --- characters that crash the filesystem ---------------------------------


@pytest.mark.parametrize("name", [
    "Resume: Senior Dev.pdf",
    "CV (Java|Python).pdf",
    'quote"name.pdf',
    "who?.pdf",
    "star*.pdf",
    "less<than.pdf",
    "greater>than.pdf",
])
def test_windows_illegal_characters_are_removed(name):
    out = safe_filename(name)
    assert not any(c in out for c in WINDOWS_ILLEGAL), out


def test_control_characters_are_removed():
    assert "\x00" not in safe_filename("bad\x00name.pdf")
    assert "\n" not in safe_filename("bad\nname.pdf")


def test_the_original_crash_case():
    out = safe_filename("resume<script>alert(1)</script>.txt")
    assert not any(c in out for c in WINDOWS_ILLEGAL)
    assert out.endswith(".txt")


# --- path traversal --------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("../../etc/passwd.txt", "passwd.txt"),
    ("..\\..\\windows\\evil.docx", "evil.docx"),
    ("/absolute/path/cv.pdf", "cv.pdf"),
    ("C:\\Users\\bob\\cv.pdf", "cv.pdf"),
])
def test_directory_components_are_stripped(name, expected):
    assert safe_filename(name) == expected


def test_no_separators_survive():
    for name in ["a/b/c.pdf", "a\\b\\c.pdf"]:
        out = safe_filename(name)
        assert "/" not in out and "\\" not in out


# --- degenerate names ------------------------------------------------------


@pytest.mark.parametrize("name", ["", ".", "..", "...", "   ", ".txt", "....pdf"])
def test_degenerate_names_still_produce_something_writable(name):
    out = safe_filename(name)
    assert out
    assert out not in (".", "..")
    assert not out.startswith(".")


def test_reserved_windows_device_names_are_escaped():
    """A file literally called CON.txt cannot be created on Windows."""
    for reserved in ["CON.txt", "con.txt", "PRN.pdf", "NUL", "COM1.docx", "LPT9.txt"]:
        out = safe_filename(reserved)
        stem = out.rsplit(".", 1)[0].lower()
        assert stem not in {"con", "prn", "nul", "aux", "com1", "lpt9"}, out


def test_very_long_names_are_truncated():
    out = safe_filename("a" * 400 + ".pdf")
    assert len(out) <= 75
    assert out.endswith(".pdf")


def test_unicode_names_survive():
    out = safe_filename("Résumé — Ann Müller.pdf")
    assert out.endswith(".pdf")
    assert "Ann" in out


# --- extension detection ---------------------------------------------------


@pytest.mark.parametrize("name,suffix", [
    ("cv.PDF", ".pdf"),
    ("cv.DocX", ".docx"),
    ("cv.txt", ".txt"),
    ("noextension", ""),
    ("archive.tar.gz", ".gz"),
])
def test_upload_suffix(name, suffix):
    assert upload_suffix(name) == suffix


def test_extension_cannot_be_smuggled_through_stripped_characters():
    """The suffix is read from the sanitised name, so it matches what we store."""
    assert upload_suffix('evil.exe"') == upload_suffix("evil.exe")


# --- display name ----------------------------------------------------------


def test_display_name_keeps_the_users_own_filename():
    assert display_filename("Resume: Senior Dev.pdf") == "Resume: Senior Dev.pdf"
    assert display_filename("Résumé — Ann.pdf") == "Résumé — Ann.pdf"


def test_display_name_still_strips_directories():
    assert display_filename("../../etc/passwd.txt") == "passwd.txt"
    assert display_filename("C:\\Users\\bob\\cv.pdf") == "cv.pdf"


def test_display_name_never_empty():
    assert display_filename("") == "resume"
