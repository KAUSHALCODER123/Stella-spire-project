"""Pydantic schemas for SpireDossier.

Two distinct LLM passes, deliberately kept separate:

  1. EXTRACTION  (CV text -> CandidateProfile)   -- facts only, no judgement.
     Evaluable against hand-labelled ground truth. See eval/run_eval.py.

  2. ASSESSMENT  (CandidateProfile + JD -> Assessment) -- judgement, always
     carrying a verbatim quote from the CV as evidence.

Anything that can be computed deterministically from the extracted facts
(employment gaps, tenure, total experience) is NOT asked of the model.
See app/analysis.py. A number we can derive is a number we can defend.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Pass 1: extraction
# --------------------------------------------------------------------------

# Dates are "YYYY-MM" strings, or the literal "present". We avoid date objects
# because CVs routinely omit the month, and a wrong-but-confident day value is
# worse than a coarse one.
MONTH_PATTERN = r"^(\d{4}-(0[1-9]|1[0-2])|present)$"


class Position(BaseModel):
    company: str
    title: str
    start: str = Field(description="Start date as YYYY-MM. Infer the month as 01 if only a year is given.")
    end: str = Field(description="End date as YYYY-MM, or the literal string 'present' for the current role.")
    location: Optional[str] = None
    employment_type: Optional[Literal["full_time", "contract", "internship", "consulting", "founder"]] = None
    summary: Optional[str] = Field(default=None, description="One neutral sentence on scope of the role.")
    achievements: List[str] = Field(default_factory=list, description="Verbatim or lightly cleaned bullet points.")
    team_size: Optional[int] = Field(default=None, description="Only if explicitly stated in the CV. Never guess.")


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    end_year: Optional[int] = None


class Skill(BaseModel):
    name: str
    category: Literal[
        "language", "framework", "cloud", "data", "ml", "infra", "domain", "leadership", "tool", "other"
    ]
    evidence: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim snippet from the CV showing this skill applied in real work. "
            "Null if the skill appears only in a skills list with no supporting context. "
            "Do NOT paraphrase - copy the text."
        ),
    )


class CandidateProfile(BaseModel):
    """Facts extracted from the CV. No opinions, no scoring."""

    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None

    headline: Optional[str] = Field(default=None, description="Current title and company, e.g. 'Staff ML Engineer at Swiggy'.")
    positions: List[Position] = Field(default_factory=list, description="Reverse-chronological, most recent first.")
    education: List[Education] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)

    # Common on Indian-market CVs; null when absent rather than invented.
    notice_period_days: Optional[int] = None
    current_ctc_lpa: Optional[float] = Field(default=None, description="Current CTC in INR lakhs per annum, if stated.")
    expected_ctc_lpa: Optional[float] = Field(default=None, description="Expected CTC in INR lakhs per annum, if stated.")

    extraction_notes: List[str] = Field(
        default_factory=list,
        description="Anything ambiguous or unreadable in the source document that a recruiter should verify by hand.",
    )


# --------------------------------------------------------------------------
# Job description
# --------------------------------------------------------------------------


class Requirement(BaseModel):
    text: str = Field(description="The requirement, as a single self-contained phrase.")
    kind: Literal["must_have", "nice_to_have"]
    category: Literal["technical", "domain", "experience", "education", "leadership", "logistics"]


class JobBrief(BaseModel):
    client_name: Optional[str] = None
    role_title: str
    location: Optional[str] = None
    seniority: Optional[str] = None
    requirements: List[Requirement] = Field(default_factory=list)
    stated_min_years: Optional[float] = None
    compensation_note: Optional[str] = None


# --------------------------------------------------------------------------
# Pass 2: assessment
# --------------------------------------------------------------------------


class RequirementMatch(BaseModel):
    requirement: str
    verdict: Literal["strong", "partial", "absent", "unclear"]
    evidence: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim quote from the CV that justifies the verdict. "
            "REQUIRED for a 'strong' verdict. Null only for 'absent'."
        ),
    )
    note: Optional[str] = Field(default=None, description="One short sentence. Only when it adds something the quote does not.")


class RiskFlag(BaseModel):
    kind: Literal[
        "employment_gap",
        "career_break",
        "short_tenure",
        "title_inflation",
        "claim_without_evidence",
        "jd_market_impossibility",
        "seniority_mismatch",
        "logistics",
    ]
    severity: Literal["low", "medium", "high"]
    summary: str = Field(description="One sentence a recruiter can verify against the CV in under ten seconds.")
    evidence: Optional[str] = Field(default=None, description="Verbatim CV quote, or the computed dates, supporting the flag.")


class Assessment(BaseModel):
    executive_summary: str = Field(
        description=(
            "3-4 sentences for a client hiring manager: what this person does, the scale they "
            "operate at, and their single most relevant qualification for THIS role. "
            "No adjectives that the CV does not support."
        )
    )
    fit_rationale: str = Field(description="2-3 sentences on why this candidate is or is not a fit for this specific brief.")
    requirement_matches: List[RequirementMatch] = Field(default_factory=list)
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list, description="Max 4, each tied to something concrete in the CV.")
    open_questions: List[str] = Field(
        default_factory=list,
        description="Max 4 questions the recruiter should ask on the screening call to close gaps in this dossier.",
    )
