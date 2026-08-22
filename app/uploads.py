"""Turning an uploaded filename into something safe to write to disk.

Regression origin: uploading a file called "Resume: Senior Dev.pdf" crashed
the server with a 500. Windows rejects `< > : " / \\ | ? *` in filenames, and
`Path(name).name` is not a sufficient guard on its own -- a name containing a
slash is silently truncated to its last segment rather than rejected, so
"resume</script>.txt" became "script>.txt", which then still contained an
illegal character.

The user always sees their original filename. This is only what we store.
"""

from __future__ import annotations

import re
from pathlib import Path

# Illegal on Windows, and a bad idea anywhere. Control characters included.
_UNSAFE = re.compile(r'[<>:"/\\|?*]|[\x00-\x1f\x7f]')

_MAX_STEM = 60
_MAX_SUFFIX = 10

# Reserved device names on Windows: a file called "CON.txt" cannot be created.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *("com{}".format(i) for i in range(1, 10)),
    *("lpt{}".format(i) for i in range(1, 10)),
}


def safe_filename(original: str) -> str:
    """A filesystem-safe basename. Never empty, never a reserved device name."""
    if not original:
        return "resume"

    # Normalise Windows separators so the basename is taken correctly.
    name = Path(original.replace("\\", "/")).name

    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""

    stem = _UNSAFE.sub("_", stem).strip(". ")
    suffix = _UNSAFE.sub("", suffix).strip(". ")

    if not stem or stem.lower() in _RESERVED:
        stem = "resume" if not stem else "file_" + stem

    stem = stem[:_MAX_STEM]
    suffix = suffix[:_MAX_SUFFIX]
    return "{}.{}".format(stem, suffix) if suffix else stem


def display_filename(original: str) -> str:
    """The name shown in the interface: the user's own, minus any path."""
    if not original:
        return "resume"
    return Path(original.replace("\\", "/")).name or "resume"


def upload_suffix(original: str) -> str:
    """Lower-cased extension, taken from the sanitised name.

    Checked against the sanitised form so an extension cannot be smuggled
    through characters that get stripped later.
    """
    return Path(safe_filename(original)).suffix.lower()
