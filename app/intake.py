"""Structured intake, and the free filter it buys.

Two ways into the system:

  * A recruiter fills a role form. Because the answers are already
    structured, the JobBrief is built directly -- no model call to parse a
    job description at all.
  * A candidate fills a short form and attaches a CV. The CV still needs
    extraction, but everything a person can simply be asked (target roles,
    locations, notice, salary expectation) is declared rather than inferred.

What that declaration is worth: those answers are HARD constraints, and
checking them costs nothing. A candidate wanting 90 LPA against a role that
tops out at 60 is not a close call an assessment should adjudicate -- it is
arithmetic, and spending a model call to discover it is waste.

So the funnel is:

    1. constraint gate   declared facts      0 calls   <- this module
    2. affinity screen   term overlap        0 calls   app/matching.py
    3. assessment        judgement           paid

Blocks are reported with the numbers behind them, and are always overridable.
A recruiter who wants to see an over-budget candidate anyway is making a
legitimate call, and the tool should not pretend the person does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas import JobBrief, Requirement

WORK_MODES = ["onsite", "hybrid", "remote", "any"]


class CandidatePreferences(BaseModel):
    """What a job seeker tells us directly, rather than us guessing from a CV."""

    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

    target_roles: List[str] = Field(default_factory=list, description="Role titles they are looking for.")
    current_location: Optional[str] = None
    preferred_locations: List[str] = Field(default_factory=list)
    open_to_relocate: bool = False
    work_mode: str = "any"

    notice_period_days: Optional[int] = None
    current_ctc_lpa: Optional[float] = None
    expected_ctc_lpa: Optional[float] = None
    # The number that actually gates a conversation, as opposed to the ask.
    min_acceptable_ctc_lpa: Optional[float] = None

    years_experience: Optional[float] = None
    notes: Optional[str] = None


class RoleConstraints(BaseModel):
    """What a recruiter tells us directly about an opening."""

    role_title: str
    client_name: Optional[str] = None
    location: Optional[str] = None
    work_mode: str = "any"

    min_years: Optional[float] = None
    max_years: Optional[float] = None

    ctc_min_lpa: Optional[float] = None
    ctc_max_lpa: Optional[float] = None
    max_notice_days: Optional[int] = None

    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    domain: Optional[str] = None
    notes: Optional[str] = None


# --------------------------------------------------------------------------
# A brief without a model call
# --------------------------------------------------------------------------


def brief_from_constraints(rc: RoleConstraints) -> JobBrief:
    """Build a JobBrief from form answers. Costs nothing.

    A structured form has already done the work the JD-parsing call exists to
    do, so running the model over a reconstructed paragraph would be paying to
    recover information we were handed.
    """
    requirements: List[Requirement] = []
    for skill in rc.must_have_skills:
        if skill.strip():
            requirements.append(Requirement(text=skill.strip(), kind="must_have", category="technical"))
    for skill in rc.nice_to_have_skills:
        if skill.strip():
            requirements.append(Requirement(text=skill.strip(), kind="nice_to_have", category="technical"))
    if rc.domain and rc.domain.strip():
        requirements.append(Requirement(text="{} domain experience".format(rc.domain.strip()),
                                        kind="must_have", category="domain"))
    if rc.min_years:
        requirements.append(Requirement(
            text="{:g}+ years of relevant experience".format(rc.min_years),
            kind="must_have", category="experience"))

    compensation = None
    if rc.ctc_min_lpa or rc.ctc_max_lpa:
        lo = "{:g}".format(rc.ctc_min_lpa) if rc.ctc_min_lpa else "?"
        hi = "{:g}".format(rc.ctc_max_lpa) if rc.ctc_max_lpa else "?"
        compensation = "INR {}-{} LPA".format(lo, hi)

    return JobBrief(
        client_name=rc.client_name,
        role_title=rc.role_title,
        location=rc.location,
        stated_min_years=rc.min_years,
        compensation_note=compensation,
        requirements=requirements,
    )


def role_text(rc: RoleConstraints) -> str:
    """A readable rendering of the form, for the source pane and quote checks."""
    lines = ["Role: {}".format(rc.role_title)]
    if rc.client_name:
        lines.append("Client: {}".format(rc.client_name))
    if rc.location:
        lines.append("Location: {} ({})".format(rc.location, rc.work_mode))
    if rc.min_years or rc.max_years:
        lines.append("Experience: {}-{} years".format(rc.min_years or "?", rc.max_years or "?"))
    if rc.ctc_min_lpa or rc.ctc_max_lpa:
        lines.append("Compensation: {} - {} LPA".format(rc.ctc_min_lpa or "?", rc.ctc_max_lpa or "?"))
    if rc.max_notice_days:
        lines.append("Maximum notice period: {} days".format(rc.max_notice_days))
    if rc.must_have_skills:
        lines.append("Must have: " + ", ".join(rc.must_have_skills))
    if rc.nice_to_have_skills:
        lines.append("Nice to have: " + ", ".join(rc.nice_to_have_skills))
    if rc.domain:
        lines.append("Domain: {}".format(rc.domain))
    if rc.notes:
        lines.append("Notes: {}".format(rc.notes))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


@dataclass
class ConstraintCheck:
    blocks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.blocks)

    @property
    def summary(self) -> str:
        if self.blocks:
            return self.blocks[0]
        if self.warnings:
            return self.warnings[0]
        return "No declared conflicts"


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _location_overlap(prefs: CandidatePreferences, role_location: Optional[str]) -> bool:
    if not role_location:
        return True
    role = _norm(role_location)
    places = [_norm(p) for p in prefs.preferred_locations if p.strip()]
    if prefs.current_location:
        places.append(_norm(prefs.current_location))
    if not places:
        return True
    # Substring either way: "Bengaluru" should match "Bengaluru (hybrid)".
    return any(p and (p in role or role in p) for p in places)


def check_constraints(prefs: CandidatePreferences, rc: RoleConstraints) -> ConstraintCheck:
    """Compare declared facts. No model call, no inference."""
    result = ConstraintCheck()

    # --- money -----------------------------------------------------------
    floor = prefs.min_acceptable_ctc_lpa or prefs.expected_ctc_lpa
    if floor and rc.ctc_max_lpa and floor > rc.ctc_max_lpa:
        gap = floor - rc.ctc_max_lpa
        if prefs.min_acceptable_ctc_lpa:
            result.blocks.append(
                "Will not go below {:g} LPA; this role tops out at {:g} LPA ({:g} short).".format(
                    floor, rc.ctc_max_lpa, gap))
        else:
            result.warnings.append(
                "Expects {:g} LPA against a band topping out at {:g} LPA. Negotiable, but a gap.".format(
                    floor, rc.ctc_max_lpa))

    # --- notice ------------------------------------------------------------
    if prefs.notice_period_days and rc.max_notice_days and prefs.notice_period_days > rc.max_notice_days:
        over = prefs.notice_period_days - rc.max_notice_days
        if over > 30:
            result.blocks.append(
                "{}-day notice against a {}-day maximum ({} days over).".format(
                    prefs.notice_period_days, rc.max_notice_days, over))
        else:
            result.warnings.append(
                "{}-day notice against a {}-day maximum. Often negotiable by {} days.".format(
                    prefs.notice_period_days, rc.max_notice_days, over))

    # --- location ----------------------------------------------------------
    if not _location_overlap(prefs, rc.location):
        if prefs.open_to_relocate:
            result.warnings.append("Not currently in {}, but open to relocating.".format(rc.location))
        elif _norm(rc.work_mode) == "remote" or _norm(prefs.work_mode) == "remote":
            result.warnings.append("Different location, but the role or the candidate allows remote.")
        else:
            result.blocks.append(
                "Role is in {} and the candidate is not open to relocating.".format(rc.location))

    # --- work mode ---------------------------------------------------------
    cand_mode, role_mode = _norm(prefs.work_mode), _norm(rc.work_mode)
    if cand_mode == "remote" and role_mode == "onsite":
        result.blocks.append("Wants remote; this role is fully onsite.")
    elif cand_mode == "onsite" and role_mode == "remote":
        result.warnings.append("Prefers onsite; this role is remote.")

    # --- experience --------------------------------------------------------
    years = prefs.years_experience
    if years is not None and rc.min_years and years < rc.min_years * 0.6:
        result.blocks.append(
            "{:g} years against a {:g}-year minimum -- too far short to assess.".format(years, rc.min_years))
    elif years is not None and rc.min_years and years < rc.min_years:
        result.warnings.append(
            "{:g} years against a {:g}-year minimum.".format(years, rc.min_years))
    if years is not None and rc.max_years and years > rc.max_years * 1.6:
        result.warnings.append(
            "{:g} years against a band ending at {:g}. Over-qualification is a retention risk.".format(
                years, rc.max_years))

    return result


def preferences_from_profile(profile, timeline=None) -> CandidatePreferences:
    """Fall back to whatever the CV declared, for candidates who filled no form.

    This keeps the gate usable on the ad-hoc upload path: a CV that states its
    own notice period and expected CTC can still be filtered for free.
    """
    return CandidatePreferences(
        full_name=profile.full_name,
        email=profile.email,
        phone=profile.phone,
        current_location=profile.location,
        notice_period_days=profile.notice_period_days,
        current_ctc_lpa=profile.current_ctc_lpa,
        expected_ctc_lpa=profile.expected_ctc_lpa,
        years_experience=timeline.total_experience_years if timeline else None,
    )
