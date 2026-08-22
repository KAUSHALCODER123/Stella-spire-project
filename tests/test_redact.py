"""Tests for blind-profile redaction.

Regression origin: the first rendered dossier was headed "Candidate SD-84923C
- Blind profile" and opened with "Arjun Menon is an ML platform engineer".
Blanking the name field is not anonymisation; the generated prose has to be
cleaned too.
"""

from __future__ import annotations

import pytest

from app.render.dossier import render_html
from app.render.redact import redact_text
from tests.fixtures import sample_dossier

KW = {"full_name": "Arjun Menon", "email": "arjun.menon.ml@gmail.com", "phone": "+91 98450 22417"}


def test_full_name_is_removed():
    assert "Arjun" not in redact_text("Arjun Menon is an ML platform engineer.", **KW)


def test_first_name_alone_is_removed():
    assert "Arjun" not in redact_text("On the call, Arjun explained the migration.", **KW)


def test_surname_alone_is_removed():
    assert "Menon" not in redact_text("Menon owns the AI roadmap.", **KW)


def test_redaction_is_case_insensitive():
    assert "ARJUN" not in redact_text("ARJUN MENON", **KW).upper().replace("[CANDIDATE]", "")


def test_full_name_replaced_before_parts():
    """'Arjun Menon' must become one placeholder, not two."""
    assert redact_text("Arjun Menon led it.", **KW) == "[candidate] led it."


def test_email_and_phone_are_removed():
    out = redact_text("Reach him at arjun.menon.ml@gmail.com or +91 98450 22417.", **KW)
    assert "@gmail" not in out and "98450" not in out


def test_unrelated_emails_are_also_caught():
    out = redact_text("Referred by priya.s@fintrail.io.", **KW)
    assert "priya.s@fintrail.io" not in out


def test_linkedin_urls_are_removed():
    out = redact_text("Profile: linkedin.com/in/arjunmenon-ml", **KW)
    assert "linkedin.com" not in out


def test_short_and_common_name_parts_are_not_redacted():
    """Redacting 'Lee' out of every sentence would shred the prose."""
    text = "The team lead reviewed the design."
    assert redact_text(text, full_name="Lee Ann") == text


def test_substrings_of_other_words_survive():
    """'Menon' must not be stripped out of an unrelated longer token."""
    out = redact_text("The Menonite dataset was used.", **KW)
    assert "Menonite" in out


def test_none_and_empty_are_passthrough():
    assert redact_text(None, **KW) is None
    assert redact_text("", **KW) == ""


def test_no_name_configured_leaves_names_alone():
    assert redact_text("Arjun Menon", full_name=None) == "Arjun Menon"


# --- whole-document guarantee --------------------------------------------


def test_rendered_blind_dossier_contains_no_identifiers():
    html = render_html(sample_dossier(), anonymise=True).lower()
    for identifier in ("arjun", "menon", "arjun.menon.ml@gmail.com", "98450", "linkedin.com/in/arjunmenon"):
        assert identifier not in html, "blind dossier leaked {!r}".format(identifier)


def test_identified_dossier_still_shows_the_name():
    html = render_html(sample_dossier(), anonymise=False)
    assert "Arjun Menon" in html


def test_redaction_does_not_mutate_the_caller_s_dossier():
    """build_context deep-copies; the recruiter's own view must stay intact."""
    dossier = sample_dossier()
    render_html(dossier, anonymise=True)
    assert dossier.profile.full_name == "Arjun Menon"
    assert "Arjun Menon" in dossier.assessment.executive_summary


def test_blind_dossier_keeps_the_substance():
    """Redaction must not gut the assessment."""
    html = render_html(sample_dossier(), anonymise=True)
    assert "4,200 requests per second" in html
    assert "Generative AI" in html
    assert "67%" in html


# --- blind mode must redact the source pane too ---------------------------


def test_blind_source_pane_is_redacted_and_still_traceable():
    """Regression: the trace view showed the raw CV, name and all, under a
    header that said BLIND. Redaction shifts offsets, so the spans have to be
    recomputed rather than carried over from the unredacted text."""
    from app.render.source import render_source
    from app.verify import verify_assessment
    import copy as _copy
    from app.render.redact import redact_dossier

    d = _copy.deepcopy(sample_dossier())
    name, email, phone = d.profile.full_name, d.profile.email, d.profile.phone
    d.document.text = redact_text(d.document.text, full_name=name, email=email, phone=phone)
    redact_dossier(d)
    d.verification = verify_assessment(d.assessment, d.document.text, d.brief_text)

    html = render_source(d.document.text, d.verification)
    # Case-insensitive: the CV header is upper-case, and a case-sensitive
    # check let "ARJUN MENON" through once already.
    lowered = html.lower()
    for identifier in ("arjun", "menon", "arjun.menon.ml@gmail.com", "98450"):
        assert identifier not in lowered, "blind source pane leaked {!r}".format(identifier)

    # Redaction must not destroy traceability.
    assert d.verification.verified >= 8, "redaction broke quote tracing"
    assert d.verification.spans(), "no highlight spans survived redaction"


def test_identified_source_pane_keeps_the_name():
    from app.render.source import render_source

    d = sample_dossier()
    html = render_source(d.document.text, d.verification)
    assert "ARJUN MENON" in html
