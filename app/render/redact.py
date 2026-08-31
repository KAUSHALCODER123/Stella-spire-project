"""Strip identifying details out of model-generated prose.

Blanking the name field in the header is not anonymisation. The executive
summary, the fit rationale, the risk flags and the evidence quotes are all
free text written about a named person, and they cheerfully use that name.
A dossier headed "Candidate SD-84923C · Blind profile" whose first sentence
reads "Arjun Menon is an ML platform engineer" is worse than not anonymising
at all, because it claims a property it does not have.

The approach is deliberately blunt: over-redaction is a cosmetic problem,
under-redaction defeats the feature.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

# Name fragments too short or too common to redact on their own -- doing so
# would shred unrelated prose.
_MIN_NAME_PART = 3
_COMMON_WORDS = {"the", "and", "for", "raj", "lead", "dev", "ali", "kim", "lee", "ram", "sun", "van", "der"}

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_URL_RE = re.compile(r"\b(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com)/[\w\-/.]+", re.I)


class BlindExportError(RuntimeError):
    """A supposedly blind deliverable still contains candidate identity."""


def _name_variants(full_name: Optional[str]) -> List[str]:
    if not full_name:
        return []
    cleaned = re.sub(r"\s+", " ", full_name).strip()
    if not cleaned:
        return []

    variants = {cleaned}
    parts = [p.strip(".,") for p in cleaned.split(" ")]
    for part in parts:
        if len(part) >= _MIN_NAME_PART and part.lower() not in _COMMON_WORDS:
            variants.add(part)

    # Longest first, so "Arjun Menon" is replaced before "Arjun" can split it.
    return sorted(variants, key=len, reverse=True)


def identity_leaks(text: str, profile) -> List[str]:
    """Return identity field labels still visible in a rendered deliverable.

    Values are deliberately not returned: an error or log message about a
    privacy failure must not repeat the personal data it just caught.
    """
    text = text or ""
    folded = text.casefold()
    leaks = set()

    for variant in _name_variants(getattr(profile, "full_name", None)):
        if re.search(r"\b" + re.escape(variant.casefold()) + r"\b", folded):
            leaks.add("candidate name")

    for label, value in (
        ("email address", getattr(profile, "email", None)),
        ("profile link", getattr(profile, "linkedin", None)),
    ):
        value = (value or "").strip()
        if value and value.casefold() in folded:
            leaks.add(label)

    phone_digits = re.sub(r"\D", "", getattr(profile, "phone", None) or "")
    rendered_digits = re.sub(r"\D", "", text)
    if len(phone_digits) >= 8 and phone_digits in rendered_digits:
        leaks.add("phone number")

    return sorted(leaks)


def assert_no_identity_leaks(text: str, profile) -> None:
    """Fail closed when blind output contains any known identifier."""
    leaks = identity_leaks(text, profile)
    if leaks:
        raise BlindExportError(
            "The blind export was stopped because its final privacy check found: {}. "
            "No client file was served.".format(", ".join(leaks))
        )


def redact_text(
    text: Optional[str],
    *,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    replacement: str = "[candidate]",
) -> Optional[str]:
    """Remove name, email, phone and profile URLs from a single string."""
    if not text:
        return text

    # Contact details FIRST. An address like arjun.menon.ml@gmail.com contains
    # the candidate's name, so redacting names first shreds it into
    # "[candidate].[candidate].[email]" -- ugly, and it still leaks the shape
    # of the original.
    if email:
        text = text.replace(email, "[email]")
    if phone:
        text = text.replace(phone, "[phone]")

    # Catch contact details the model quoted straight out of the CV header,
    # even when they differ from the extracted fields.
    text = _EMAIL_RE.sub("[email]", text)
    text = _URL_RE.sub("[profile]", text)
    text = _PHONE_RE.sub("[phone]", text)

    # Names last, on whatever is left.
    for variant in _name_variants(full_name):
        text = re.sub(r"\b" + re.escape(variant) + r"\b", replacement, text, flags=re.IGNORECASE)

    return _fix_sentence_case(text, replacement)


def _fix_sentence_case(text: str, replacement: str) -> str:
    """Capitalise the placeholder when it opens a sentence.

    A name replaced by "the candidate" mid-paragraph is fine, but at the start
    of a sentence it reads as a typo -- and this prose goes to clients.
    """
    if not replacement or not replacement[:1].islower():
        return text
    capped = replacement[0].upper() + replacement[1:]
    text = re.sub(r"(^|[.!?]\s+|\n\s*)" + re.escape(replacement),
                  lambda m: m.group(1) + capped, text)
    return text


def _redact_list(values: Iterable[str], **kw) -> List[str]:
    return [redact_text(v, **kw) or "" for v in values]


def redact_dossier(dossier, replacement: str = "the candidate") -> None:
    """Redact a Dossier's generated prose in place.

    Call only on a copy destined for a blind rendering -- this mutates.
    """
    profile = dossier.profile
    kw = {
        "full_name": profile.full_name,
        "email": profile.email,
        "phone": profile.phone,
        "replacement": replacement,
    }

    a = dossier.assessment
    a.executive_summary = redact_text(a.executive_summary, **kw) or ""
    a.fit_rationale = redact_text(a.fit_rationale, **kw) or ""
    a.strengths = _redact_list(a.strengths, **kw)
    a.open_questions = _redact_list(a.open_questions, **kw)

    for match in a.requirement_matches:
        match.evidence = redact_text(match.evidence, **kw)
        match.note = redact_text(match.note, **kw)

    for flag in dossier.flags:
        flag.summary = redact_text(flag.summary, **kw) or ""
        flag.evidence = redact_text(flag.evidence, **kw)

    # The headline is shown verbatim in the blind header and routinely contains
    # the name ("Arjun Menon — Head of AI Engineering").
    profile.headline = redact_text(profile.headline, **kw)

    for position in profile.positions:
        position.summary = redact_text(position.summary, **kw)
        position.achievements = _redact_list(position.achievements, **kw)

    for skill in profile.skills:
        skill.evidence = redact_text(skill.evidence, **kw)

    profile.extraction_notes = _redact_list(profile.extraction_notes, **kw)
