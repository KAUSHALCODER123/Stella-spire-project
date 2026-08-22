"""A fully-populated Dossier built by hand, with no API call.

This exists so the renderer, the templates and the PDF pipeline can be tested
and iterated on without spending a token or needing a key.

The candidate is chosen deliberately: a finance leader in Stellaspire's actual
vertical -- CFO and analytics leadership for GCCs and growth companies -- who
has taken a career break. That combination is the firm's business in one
screen, and it exercises every branch the templates can take: strong, partial,
absent and unclear verdicts, a computed career break, a market-impossibility
flag against the client's own brief, evidenced and bare skills, and a quote
that deliberately fails verification.
"""

from __future__ import annotations

from pathlib import Path

from app.analysis import build_timeline, derive_risk_flags, sort_flags
from app.extract.documents import DocumentText
from app.extract.llm import Usage
from app.pipeline import Dossier
from app.schemas import (
    Assessment,
    CandidateProfile,
    Education,
    JobBrief,
    Position,
    Requirement,
    RequirementMatch,
    RiskFlag,
    Skill,
)
from app.verify import verify_assessment


def sample_profile() -> CandidateProfile:
    return CandidateProfile(
        full_name="Meera Ramanathan",
        email="meera.ramanathan.fin@gmail.com",
        phone="+91 98860 41277",
        location="Bengaluru, Karnataka",
        linkedin="linkedin.com/in/meeraramanathan-finance",
        headline="Head of Finance at Aventra Capital (NBFC)",
        notice_period_days=60,
        current_ctc_lpa=82.0,
        expected_ctc_lpa=105.0,
        positions=[
            Position(
                company="Aventra Capital", title="Head of Finance",
                start="2022-09", end="present", location="Bengaluru",
                employment_type="full_time", team_size=17,
                summary="Owns FP&A, controllership, treasury and investor reporting.",
                achievements=[
                    "Own FP&A, controllership, treasury and investor reporting for a Rs 1,850 crore "
                    "AUM lending book across 14 states.",
                    "Rebuilt the monthly close from 19 working days to 6 by moving reconciliation off "
                    "spreadsheets onto a controls-based process in NetSuite.",
                    "Led the Series C raise of USD 34 million: built the model, ran diligence across "
                    "three funds, and closed in 11 weeks.",
                    "Took the entity through two RBI inspections with no adverse observations.",
                    "Built the finance team from 4 to 17, including the first FP&A hire and a "
                    "dedicated regulatory reporting analyst.",
                ],
            ),
            Position(
                company="Rupeek Fintech", title="Senior Manager, Financial Planning & Analysis",
                start="2018-06", end="2021-02", location="Bengaluru", employment_type="full_time",
                achievements=[
                    "Owned the annual operating plan and rolling 13-week cash forecast through a "
                    "period when disbursals grew 4x.",
                    "Built unit economics reporting by product and channel; the contribution margin "
                    "view it produced led to two channels being shut.",
                    "Partnered with the credit team on provisioning under Ind AS 109, including the "
                    "ECL model refresh at the onset of COVID.",
                ],
            ),
            Position(
                company="Deloitte India", title="Manager, Audit and Assurance",
                start="2014-08", end="2018-05", location="Bengaluru", employment_type="full_time",
                achievements=[
                    "Statutory audit lead for three NBFC clients and one small finance bank.",
                    "Ran the Ind AS transition for a listed housing finance company.",
                    "Managed audit teams of 6 to 9 across engagements.",
                ],
            ),
            Position(
                company="S. R. Batliboi & Co.", title="Article Assistant",
                start="2011-07", end="2014-07", location="Chennai", employment_type="full_time",
                achievements=["Statutory and internal audit across banking and manufacturing clients."],
            ),
        ],
        education=[
            Education(institution="ICAI", degree="Chartered Accountant", end_year=2014),
            Education(institution="Christ University, Bengaluru", degree="B.Com (Honours)", end_year=2011),
        ],
        skills=[
            Skill(name="Ind AS 109 / ECL", category="domain",
                  evidence="Partnered with the credit team on provisioning under Ind AS 109, "
                           "including the ECL model refresh at the onset of COVID."),
            Skill(name="Controllership", category="domain",
                  evidence="Rebuilt the monthly close from 19 working days to 6"),
            Skill(name="FP&A", category="domain",
                  evidence="Owned the annual operating plan and rolling 13-week cash forecast"),
            Skill(name="Statutory audit", category="domain",
                  evidence="Statutory audit lead for three NBFC clients and one small finance bank."),
            Skill(name="RBI regulatory reporting", category="domain",
                  evidence="Took the entity through two RBI inspections with no adverse observations."),
            Skill(name="Fundraising and diligence", category="domain",
                  evidence="Led the Series C raise of USD 34 million"),
            Skill(name="Three-statement modelling", category="domain", evidence="built the model"),
            Skill(name="Treasury", category="domain", evidence=None),
            Skill(name="Investor relations", category="domain", evidence=None),
            Skill(name="NetSuite", category="tool", evidence="a controls-based process in NetSuite"),
            Skill(name="Anaplan", category="tool", evidence=None),
            Skill(name="Tally", category="tool", evidence=None),
            Skill(name="Power BI", category="data", evidence=None),
            Skill(name="SQL", category="data", evidence=None),
            Skill(name="Team leadership", category="leadership",
                  evidence="Built the finance team from 4 to 17"),
            Skill(name="Board reporting", category="leadership", evidence=None),
        ],
        certifications=["Chartered Accountant, ICAI (All India Rank 41)", "CFA Level II"],
        extraction_notes=[
            "The CV marks Mar 2021 to Aug 2022 as a career break, stated as full-time caregiving "
            "following a family illness, with CFA Level II completed and part-time IFRS 16 "
            "consulting during the period.",
        ],
    )


def sample_brief() -> JobBrief:
    reqs = [
        ("Qualified Chartered Accountant", "must_have", "education"),
        ("15+ years post-qualification experience in financial services", "must_have", "experience"),
        ("10+ years as a CFO of a venture-backed FinTech", "must_have", "experience"),
        ("Deep expertise in Ind AS, including Ind AS 109 expected credit loss", "must_have", "domain"),
        ("Proven controllership: owning month-end close and audit readiness", "must_have", "domain"),
        ("FP&A leadership: annual operating plan, rolling forecast, unit economics", "must_have", "domain"),
        ("Experience raising institutional capital and running investor diligence", "must_have", "domain"),
        ("RBI regulatory reporting for an NBFC or payments entity", "must_have", "domain"),
        ("Led a finance team of at least 15", "must_have", "leadership"),
        ("Board and audit committee exposure", "must_have", "leadership"),
        ("CFA or MBA in finance", "nice_to_have", "education"),
        ("Experience with a Global Capability Centre finance function", "nice_to_have", "domain"),
        ("NetSuite or Anaplan implementation", "nice_to_have", "technical"),
    ]
    return JobBrief(
        client_name="Confidential (Series C FinTech)",
        role_title="Chief Financial Officer",
        location="Bengaluru (hybrid)",
        seniority="CFO",
        stated_min_years=15,
        compensation_note="INR 95-130 LPA plus ESOPs",
        requirements=[Requirement(text=t, kind=k, category=c) for t, k, c in reqs],
    )


def sample_assessment() -> Assessment:
    return Assessment(
        executive_summary=(
            "Meera Ramanathan is a Chartered Accountant running finance for a Rs 1,850 crore AUM "
            "NBFC, where she owns FP&A, controllership, treasury and investor reporting and has "
            "grown the function from four people to seventeen. The most decision-relevant fact for "
            "this brief is that she has already done the job it describes: she took a month-end "
            "close from nineteen working days to six, and closed a USD 34 million Series C in "
            "eleven weeks. She has not held the CFO title, which is what the brief nominally asks "
            "for, but she has held its scope."
        ),
        fit_rationale=(
            "Strong on substance, short on title. Every operational requirement here — close "
            "discipline, Ind AS 109, RBI reporting, fundraise, team build — she has evidence for, "
            "in the same regulatory environment the client operates in. The gap is formal CFO "
            "tenure, and the brief's ten-year version of that requirement is not satisfiable by "
            "anyone. Treat the title as the negotiable item."
        ),
        requirement_matches=[
            RequirementMatch(
                requirement="Qualified Chartered Accountant", verdict="strong",
                evidence="Chartered Accountant, ICAI - 2014 (first attempt, All India Rank 41)"),
            RequirementMatch(
                requirement="15+ years post-qualification experience in financial services",
                verdict="partial",
                evidence="Statutory and internal audit across banking and manufacturing clients.",
                note="Career spans 14 years since 2011, all of it in financial services. Qualified "
                     "in 2014, so around 11 years post-qualification against a stated 15."),
            RequirementMatch(
                requirement="10+ years as a CFO of a venture-backed FinTech", verdict="absent",
                note="Nobody satisfies this. See the market-impossibility flag — the requirement "
                     "should be renegotiated before the search continues."),
            RequirementMatch(
                requirement="Deep expertise in Ind AS, including Ind AS 109 expected credit loss",
                verdict="strong",
                evidence="Partnered with the credit team on provisioning under Ind AS 109, including "
                         "the ECL model refresh at the onset of COVID.",
                note="Also ran the Ind AS transition for a listed housing finance company at Deloitte."),
            RequirementMatch(
                requirement="Proven controllership: owning month-end close and audit readiness",
                verdict="strong",
                evidence="Rebuilt the monthly close from 19 working days to 6 by moving reconciliation "
                         "off spreadsheets onto a controls-based process in NetSuite.",
                note="Directly on point: the client's stated problem is a three-week close."),
            RequirementMatch(
                requirement="FP&A leadership: annual operating plan, rolling forecast, unit economics",
                verdict="strong",
                evidence="Owned the annual operating plan and rolling 13-week cash forecast through a "
                         "period when disbursals grew 4x."),
            RequirementMatch(
                requirement="Experience raising institutional capital and running investor diligence",
                verdict="strong",
                evidence="Led the Series C raise of USD 34 million: built the model, ran diligence "
                         "across three funds, and closed in 11 weeks."),
            RequirementMatch(
                requirement="RBI regulatory reporting for an NBFC or payments entity", verdict="strong",
                evidence="Took the entity through two RBI inspections with no adverse observations."),
            RequirementMatch(
                requirement="Led a finance team of at least 15", verdict="strong",
                evidence="Built the finance team from 4 to 17, including the first FP&A hire and a "
                         "dedicated regulatory reporting analyst."),
            RequirementMatch(
                requirement="Board and audit committee exposure", verdict="unclear",
                evidence="Head of Finance, Aventra Capital, Bengaluru, Sep 2022 - Present",
                note="Investor reporting is evidenced; board and audit committee attendance is not "
                     "stated either way. Ask on the call."),
            RequirementMatch(
                requirement="CFA or MBA in finance", verdict="partial",
                evidence="CFA Level II candidate - cleared 2022",
                note="Level II cleared, not a charterholder."),
            RequirementMatch(
                requirement="Experience with a Global Capability Centre finance function",
                verdict="absent"),
            RequirementMatch(
                requirement="NetSuite or Anaplan implementation", verdict="partial",
                evidence="moving reconciliation off spreadsheets onto a controls-based process in NetSuite",
                note="Used NetSuite to deliver a process change. Whether she ran the implementation "
                     "itself is not stated."),
        ],
        risk_flags=[
            RiskFlag(
                kind="jd_market_impossibility", severity="high",
                summary=(
                    "The brief asks for 10+ years as CFO of a venture-backed FinTech. India's "
                    "venture-backed fintech sector barely had CFO-level roles a decade ago, so the "
                    "pool that satisfies this literally is in the low single digits nationally. "
                    "Recommend rewriting to 5+ years at Head of Finance level or above in a "
                    "regulated lender, which is what the job actually needs."),
                evidence="10+ years as a CFO of a venture-backed FinTech"),
            RiskFlag(
                kind="seniority_mismatch", severity="medium",
                summary=(
                    "She has held CFO scope without the CFO title. Tell the client explicitly rather "
                    "than letting them screen on the title, and expect a title conversation at offer."),
                evidence="Head of Finance at Aventra Capital (NBFC)"),
            RiskFlag(
                kind="claim_without_evidence", severity="low",
                summary=(
                    "Treasury, Anaplan, Power BI and SQL appear in the skills list with nothing behind "
                    "them in the described work. Not necessarily overstated, but unverified."),
                evidence="Anaplan, SQL, Power BI, board reporting"),
        ],
        strengths=[
            "Has already solved the client's stated problem: close cut from 19 days to 6.",
            "Closed a USD 34 million Series C in eleven weeks, including the model and diligence.",
            "Two RBI inspections with no adverse observations, in the same regulatory regime.",
            "Built a finance team from 4 to 17, including the first FP&A hire.",
        ],
        open_questions=[
            "Do you attend board and audit committee meetings directly, or report into someone who does?",
            "Was the NetSuite move an implementation you led, or a process change on an existing system?",
            "Given the close is currently three weeks, what would your first ninety days look like?",
            "The brief is written for a CFO title — is that something you would need on day one?",
        ],
    )


def sample_dossier() -> Dossier:
    profile = sample_profile()
    timeline = build_timeline(profile)
    assessment = sample_assessment()

    root = Path(__file__).resolve().parent.parent
    cv_text = (root / "data" / "samples" / "cv_meera_ramanathan.txt").read_text(encoding="utf-8")
    jd_text = (root / "data" / "samples" / "jd_cfo_fintech_gcc.txt").read_text(encoding="utf-8")

    document = DocumentText(text=cv_text, page_count=2, source_format="pdf",
                            filename="Meera Ramanathan CV.pdf")

    return Dossier(
        profile=profile,
        timeline=timeline,
        brief=sample_brief(),
        assessment=assessment,
        flags=sort_flags(derive_risk_flags(profile, timeline, cv_text) + assessment.risk_flags),
        document=document,
        usage=Usage(input_tokens=16240, output_tokens=4180, calls=3,
                    call_log=["extract_profile", "extract_job_brief", "assess"]),
        brief_text=jd_text,
        verification=verify_assessment(assessment, cv_text, jd_text),
        model="gpt-4o",
        elapsed_seconds=28.4,
        warnings=[],
    )
