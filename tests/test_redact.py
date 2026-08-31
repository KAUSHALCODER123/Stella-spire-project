"""Tests for blind-profile redaction.

Regression origin: the first rendered dossier was headed "Candidate SD-84923C
- Blind profile" and opened with "Meera Ramanathan is an ML platform engineer".
Blanking the name field is not anonymisation; the generated prose has to be
cleaned too.
"""

from __future__ import annotations

import pytest

from app.render.dossier import render_html
from app.render.redact import BlindExportError, assert_no_identity_leaks, identity_leaks, redact_text
from tests.fixtures import sample_dossier

KW = {"full_name": "Meera Ramanathan", "email": "meera.ramanathan.fin@gmail.com", "phone": "+91 98860 41277"}


def test_full_name_is_removed():
    assert "Meera" not in redact_text("Meera Ramanathan is an ML platform engineer.", **KW)


def test_first_name_alone_is_removed():
    assert "Meera" not in redact_text("On the call, Meera explained the migration.", **KW)


def test_surname_alone_is_removed():
    assert "Ramanathan" not in redact_text("Ramanathan owns the AI roadmap.", **KW)


def test_redaction_is_case_insensitive():
    assert "ARJUN" not in redact_text("MEERA RAMANATHAN", **KW).upper().replace("[CANDIDATE]", "")


def test_full_name_replaced_before_parts():
    """'Meera Ramanathan' must become one placeholder, not two."""
    assert redact_text("Meera Ramanathan led it.", **KW) == "[candidate] led it."


def test_email_and_phone_are_removed():
    out = redact_text("Reach him at meera.ramanathan.fin@gmail.com or +91 98860 41277.", **KW)
    assert "@gmail" not in out and "98860" not in out


def test_unrelated_emails_are_also_caught():
    out = redact_text("Referred by priya.s@fintrail.io.", **KW)
    assert "priya.s@fintrail.io" not in out


def test_linkedin_urls_are_removed():
    out = redact_text("Profile: linkedin.com/in/meeraramanathan-finance", **KW)
    assert "linkedin.com" not in out


def test_short_and_common_name_parts_are_not_redacted():
    """Redacting 'Lee' out of every sentence would shred the prose."""
    text = "The team lead reviewed the design."
    assert redact_text(text, full_name="Lee Ann") == text


def test_substrings_of_other_words_survive():
    """'Ramanathan' must not be stripped out of an unrelated longer token."""
    out = redact_text("The Ramanathanite dataset was used.", **KW)
    assert "Ramanathanite" in out


def test_none_and_empty_are_passthrough():
    assert redact_text(None, **KW) is None
    assert redact_text("", **KW) == ""


def test_no_name_configured_leaves_names_alone():
    assert redact_text("Meera Ramanathan", full_name=None) == "Meera Ramanathan"


# --- whole-document guarantee --------------------------------------------


def test_rendered_blind_dossier_contains_no_identifiers():
    html = render_html(sample_dossier(), anonymise=True).lower()
    for identifier in ("meera", "ramanathan", "meera.ramanathan.fin@gmail.com", "98860", "linkedin.com/in/meeraramanathan"):
        assert identifier not in html, "blind dossier leaked {!r}".format(identifier)


def test_identified_dossier_still_shows_the_name():
    html = render_html(sample_dossier(), anonymise=False)
    assert "Meera Ramanathan" in html


def test_redaction_does_not_mutate_the_caller_s_dossier():
    """build_context deep-copies; the recruiter's own view must stay intact."""
    dossier = sample_dossier()
    render_html(dossier, anonymise=True)
    assert dossier.profile.full_name == "Meera Ramanathan"
    assert "Meera Ramanathan" in dossier.assessment.executive_summary


def test_blind_dossier_keeps_the_substance():
    """Redaction must not gut the assessment."""
    html = render_html(sample_dossier(), anonymise=True)
    assert "19 working days to 6" in html
    assert "Ind AS 109" in html
    assert "Series C" in html


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
    # check let "MEERA RAMANATHAN" through once already.
    lowered = html.lower()
    for identifier in ("meera", "ramanathan", "meera.ramanathan.fin@gmail.com", "98860"):
        assert identifier not in lowered, "blind source pane leaked {!r}".format(identifier)

    # Redaction must not destroy traceability.
    assert d.verification.verified >= 8, "redaction broke quote tracing"
    assert d.verification.spans(), "no highlight spans survived redaction"


def test_identified_source_pane_keeps_the_name():
    from app.render.source import render_source

    d = sample_dossier()
    html = render_source(d.document.text, d.verification)
    assert "MEERA RAMANATHAN" in html


# --- ordering: contacts before names --------------------------------------


def test_email_is_redacted_whole_not_shredded_by_the_name():
    """Regression: meera.ramanathan.fin@gmail.com contains the candidate's name, so
    redacting names first produced '[candidate].[candidate].[email]'."""
    out = redact_text("Reach them at meera.ramanathan.fin@gmail.com today", **KW)
    assert "[email]" in out
    assert "the candidate." not in out
    assert "[candidate]." not in out
    assert out.count("[email]") == 1


def test_contact_line_redacts_cleanly():
    line = "Bengaluru, Karnataka | meera.ramanathan.fin@gmail.com | +91 98860 41277"
    out = redact_text(line, **KW)
    assert out == "Bengaluru, Karnataka | [email] | [phone]"


def test_third_party_email_containing_no_name_still_redacted():
    out = redact_text("Referred by priya.s@fintrail.io", **KW)
    assert "priya.s@fintrail.io" not in out and "[email]" in out


# --- sentence case ---------------------------------------------------------


def test_placeholder_is_capitalised_at_the_start_of_a_sentence():
    out = redact_text("Meera Ramanathan is an ML engineer.", replacement="the candidate", **KW)
    assert out.startswith("The candidate is"), out


def test_placeholder_capitalised_after_a_full_stop():
    out = redact_text("We met. Meera Ramanathan leads the team.", replacement="the candidate", **KW)
    assert "We met. The candidate leads" in out, out


def test_placeholder_stays_lowercase_mid_sentence():
    out = redact_text("The report on Meera Ramanathan is ready.", replacement="the candidate", **KW)
    assert "on the candidate is ready" in out, out


def test_bracket_placeholder_is_left_alone():
    """'[candidate]' is a redaction marker, not prose; do not case-fix it."""
    out = redact_text("Meera Ramanathan is here.", replacement="[candidate]", **KW)
    assert out.startswith("[candidate]")


def test_final_privacy_check_reports_field_labels_not_personal_values():
    profile = sample_dossier().profile
    leaks = identity_leaks("Meera Ramanathan · +91 98860 41277", profile)
    assert leaks == ["candidate name", "phone number"]

    with pytest.raises(BlindExportError) as caught:
        assert_no_identity_leaks("Meera Ramanathan · +91 98860 41277", profile)
    message = str(caught.value)
    assert "candidate name" in message and "phone number" in message
    assert "Meera" not in message and "98860" not in message


def test_final_privacy_check_accepts_a_redacted_render():
    dossier = sample_dossier()
    html = render_html(dossier, anonymise=True)
    assert_no_identity_leaks(html, dossier.profile)
