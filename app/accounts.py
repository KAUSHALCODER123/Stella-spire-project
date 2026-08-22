"""Company accounts and the single sign-in that also lets the admin in.

There is deliberately no separate admin login. Everyone uses one form; the
account decides what they see. A hidden second door is a thing to forget to
protect, and having one login path means the authorisation check lives in one
place rather than being re-derived per surface.

Passwords are PBKDF2-HMAC-SHA256 with a per-account salt. That is the right
shape (salted, stretched, constant-time compare) but 240k iterations of
stdlib PBKDF2 is a demo-grade choice, not a production one -- see the note on
`hash_password`.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ITERATIONS = 240_000
ADMIN_EMAIL = "admin"
ADMIN_PASSWORD = "admin123"


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Salted, stretched hash as 'salt$hash', both hex.

    Production would use argon2id or bcrypt via passlib. PBKDF2 from the
    standard library is used here to avoid a native dependency; it is a real
    KDF, just a less memory-hard one.
    """
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return "{}${}".format(salt.hex(), digest.hex())


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, AttributeError):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    # Constant time: a fast rejection leaks how much of the hash matched.
    return hmac.compare_digest(candidate.hex(), digest_hex)


@dataclass
class Company:
    id: str
    name: str
    email: str
    password_hash: str
    is_admin: bool = False
    industry: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def initials(self) -> str:
        parts = [p for p in self.name.split() if p]
        if not parts:
            return "??"
        if len(parts) == 1:
            return parts[0][:2].upper()
        return (parts[0][0] + parts[1][0]).upper()

    @property
    def display_role(self) -> str:
        return "Administrator" if self.is_admin else "Employer"


COMPANIES: Dict[str, Company] = {}


def _norm(email: str) -> str:
    return (email or "").strip().lower()


def find_by_email(email: str) -> Optional[Company]:
    target = _norm(email)
    for company in COMPANIES.values():
        if company.email == target:
            return company
    return None


def create_company(*, name: str, email: str, password: str, is_admin: bool = False,
                   industry: Optional[str] = None, location: Optional[str] = None,
                   website: Optional[str] = None) -> Company:
    company = Company(
        id=secrets.token_hex(6),
        name=name.strip(),
        email=_norm(email),
        password_hash=hash_password(password),
        is_admin=is_admin,
        industry=(industry or "").strip() or None,
        location=(location or "").strip() or None,
        website=(website or "").strip() or None,
    )
    COMPANIES[company.id] = company
    return company


def authenticate(email: str, password: str) -> Optional[Company]:
    company = find_by_email(email)
    if company is None:
        # Spend the same time as a real check so timing does not reveal
        # whether an account exists.
        hash_password(password)
        return None
    if not verify_password(password, company.password_hash):
        return None
    return company


def seed(force: bool = False) -> List[Company]:
    """The admin account plus a few employers, so the app is explorable.

    Seeded companies use the same password as the admin. They exist to make
    the sign-in flow demonstrable, not to be secure.
    """
    if COMPANIES and not force:
        return list(COMPANIES.values())

    created = [create_company(
        name="Stellaspire", email=ADMIN_EMAIL, password=ADMIN_PASSWORD, is_admin=True,
        industry="Executive search", location="Bengaluru",
    )]
    for name, email, industry, location in [
        ("Fintrail Technologies", "hiring@fintrail.example", "FinTech", "Bengaluru"),
        ("Meridian Global Capability Centre", "talent@meridiangcc.example", "Insurance", "Hyderabad"),
        ("Alderline NBFC", "careers@alderline.example", "Non-banking finance", "Mumbai"),
        ("Northwind Analytics", "people@northwind.example", "Data & analytics", "Gurgaon"),
        ("Qadira Digital", "jobs@qadira.example", "Enterprise technology", "Dubai"),
    ]:
        created.append(create_company(name=name, email=email, password=ADMIN_PASSWORD,
                                      industry=industry, location=location))
    return created


# --------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------


def current_company(request) -> Optional[Company]:
    company_id = request.session.get("company_id")
    return COMPANIES.get(company_id) if company_id else None


def sign_in(request, company: Company) -> None:
    request.session["company_id"] = company.id


def sign_out(request) -> None:
    request.session.pop("company_id", None)
