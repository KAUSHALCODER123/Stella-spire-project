"""Applicant email notifications, sent over Gmail SMTP with an app password.

Deliberately not a general-purpose mailer: one function, one message shape,
because the only thing this app currently needs to tell a candidate is that
their CV arrived. Extend the signature when a second kind of email is needed
rather than building a template system for one caller.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from app.config import settings

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


class NotifyError(Exception):
    """Email could not be sent. The message is safe to show an admin."""


def is_configured() -> bool:
    return bool(settings.gmail_address and settings.gmail_app_password)


def send_application_received(*, to_email: str, candidate_name: str, role_title: str) -> None:
    if not is_configured():
        raise NotifyError(
            "Email is not configured. Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD to .env and restart."
        )

    first_name = (candidate_name or "there").split()[0]
    message = EmailMessage()
    message["Subject"] = "Your application — {}".format(role_title)
    message["From"] = settings.gmail_address
    message["To"] = to_email
    message.set_content(
        "Hi {},\n\n"
        "Your CV has been received for the {} role and is under review. "
        "We will be in touch if there is a fit.\n\n"
        "— {}".format(first_name, role_title, settings.agency_name)
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(settings.gmail_address, settings.gmail_app_password)
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        raise NotifyError(
            "Gmail rejected the sign-in. Check GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env."
        ) from exc
    except (smtplib.SMTPException, OSError) as exc:
        raise NotifyError("Could not send email: {}".format(exc)) from exc
