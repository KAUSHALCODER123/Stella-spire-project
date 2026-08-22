"""A fully-populated Dossier built by hand, with no API call.

This exists so the renderer, the template and the PDF pipeline can be tested
and iterated on without spending a token or needing a key. It mirrors
data/samples/cv_arjun_menon.txt assessed against
data/samples/jd_genai_platform_lead.txt, and includes every branch the
template can take: strong/partial/absent/unclear verdicts, computed and
model-generated flags, evidenced and bare skills, and an employment gap.
"""

from __future__ import annotations

from pathlib import Path

from app.analysis import build_timeline, derive_risk_flags, sort_flags
from app.extract.documents import DocumentText
from app.extract.llm import Usage
from app.pipeline import Dossier
from app.verify import verify_assessment
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


def sample_profile() -> CandidateProfile:
    return CandidateProfile(
        full_name="Arjun Menon",
        email="arjun.menon.ml@gmail.com",
        phone="+91 98450 22417",
        location="Bengaluru, Karnataka",
        linkedin="linkedin.com/in/arjunmenon-ml",
        headline="Head of AI Engineering at Fintrail Technologies",
        notice_period_days=90,
        current_ctc_lpa=68.0,
        expected_ctc_lpa=90.0,
        positions=[
            Position(
                company="Fintrail Technologies",
                title="Head of AI Engineering",
                start="2024-01",
                end="present",
                location="Bengaluru",
                employment_type="full_time",
                summary="Owns the AI roadmap for lending risk products.",
                achievements=[
                    "Rebuilt the credit-decision feature pipeline on Feast and Spark Structured Streaming, "
                    "cutting feature freshness from 6 hours to under 90 seconds.",
                    "Introduced shadow deployment for all model releases; rollback time went from a "
                    "two-hour manual process to four minutes.",
                    "Shipped an internal RAG assistant over policy documents using LangChain and pgvector, "
                    "now used daily by the underwriting team.",
                ],
            ),
            Position(
                company="Razorpay",
                title="Senior Machine Learning Engineer",
                start="2021-03",
                end="2023-12",
                location="Bengaluru",
                employment_type="full_time",
                achievements=[
                    "Built and owned the real-time fraud scoring service handling ~4,200 requests per "
                    "second at p99 latency of 40ms, written in Go with a PyTorch model served via TorchServe.",
                    "Designed the offline-to-online feature parity test suite that caught a training-serving "
                    "skew bug responsible for a 12% drop in recall.",
                    "Migrated model training from ad-hoc notebooks to Kubeflow Pipelines on GKE.",
                ],
            ),
            Position(
                company="Mu Sigma",
                title="Machine Learning Engineer",
                start="2019-07",
                end="2020-08",
                location="Bengaluru",
                employment_type="full_time",
                achievements=[
                    "Built demand-forecasting models for a US retail client using gradient boosted trees; "
                    "reduced forecast error (WMAPE) by 8 percentage points.",
                ],
            ),
            Position(
                company="Tata Consultancy Services",
                title="Data Analyst",
                start="2017-06",
                end="2019-06",
                location="Chennai",
                employment_type="full_time",
                achievements=["Built the branch-level delinquency dashboard used by 40+ regional managers."],
            ),
        ],
        education=[
            Education(institution="National Institute of Technology, Trichy", degree="M.Tech",
                      field_of_study="Computer Science", end_year=2017),
            Education(institution="Anna University", degree="B.Tech",
                      field_of_study="Information Technology", end_year=2015),
        ],
        skills=[
            Skill(name="Python", category="language",
                  evidence="written in Go with a PyTorch model served via TorchServe"),
            Skill(name="Go", category="language",
                  evidence="real-time fraud scoring service ... written in Go"),
            Skill(name="PyTorch", category="ml",
                  evidence="a PyTorch model served via TorchServe"),
            Skill(name="TensorFlow", category="ml", evidence=None),
            Skill(name="CUDA", category="ml", evidence=None),
            Skill(name="LLM fine-tuning", category="ml", evidence=None),
            Skill(name="Kubeflow", category="ml",
                  evidence="Migrated model training from ad-hoc notebooks to Kubeflow Pipelines on GKE"),
            Skill(name="LangChain", category="ml",
                  evidence="RAG assistant over policy documents using LangChain and pgvector"),
            Skill(name="pgvector", category="data",
                  evidence="using LangChain and pgvector"),
            Skill(name="Spark", category="data",
                  evidence="feature pipeline on Feast and Spark Structured Streaming"),
            Skill(name="Feast", category="data", evidence="on Feast and Spark Structured Streaming"),
            Skill(name="Kafka", category="data", evidence=None),
            Skill(name="Kubernetes", category="infra", evidence="Kubeflow Pipelines on GKE"),
            Skill(name="Terraform", category="infra", evidence=None),
            Skill(name="Docker", category="infra", evidence=None),
            Skill(name="AWS", category="cloud", evidence=None),
            Skill(name="GCP", category="cloud", evidence="Kubeflow Pipelines on GKE"),
            Skill(name="Fraud & risk", category="domain",
                  evidence="Own the AI roadmap for Fintrail's lending risk products"),
            Skill(name="Mentoring", category="leadership",
                  evidence="Mentored three junior engineers through the ML engineering ladder"),
        ],
        certifications=["Google Cloud Professional Machine Learning Engineer (2022)"],
        extraction_notes=[],
    )


def sample_brief() -> JobBrief:
    reqs = [
        ("10+ years of hands-on Generative AI engineering experience", "must_have", "experience"),
        ("12+ years total software engineering experience", "must_have", "experience"),
        ("Architecting and operating RAG systems at enterprise scale", "must_have", "technical"),
        ("LLM fine-tuning (LoRA / QLoRA) on production workloads", "must_have", "technical"),
        ("MLOps: CI/CD for models, monitoring, drift detection", "must_have", "technical"),
        ("Hands-on Kubernetes and infrastructure-as-code", "must_have", "technical"),
        ("Real-time, low-latency inference serving", "must_have", "technical"),
        ("Led a team of at least 8 engineers", "must_have", "leadership"),
        ("Insurance or financial services domain experience", "must_have", "domain"),
        ("Setting up an AI function inside a Global Capability Centre", "nice_to_have", "domain"),
        ("Vector databases at scale", "nice_to_have", "technical"),
    ]
    return JobBrief(
        client_name="Confidential GCC (US insurer)",
        role_title="GenAI Platform Lead",
        location="Bengaluru (hybrid)",
        seniority="Lead / Head of",
        stated_min_years=12,
        compensation_note="INR 85–110 LPA + 15% bonus",
        requirements=[Requirement(text=t, kind=k, category=c) for t, k, c in reqs],
    )


def sample_assessment() -> Assessment:
    return Assessment(
        executive_summary=(
            "Arjun Menon is an ML platform engineer with nine years of experience, currently running AI "
            "engineering at a Bengaluru lending-risk fintech. His strongest and most relevant work is "
            "production ML infrastructure at scale — a fraud scoring service at 4,200 rps and 40ms p99 at "
            "Razorpay — rather than GenAI specifically, where his experience is one shipped internal RAG "
            "assistant. For a 0-to-1 GenAI platform build in insurance, he brings the platform and "
            "financial-services depth but not the GenAI depth or the team-leadership scale the brief asks for."
        ),
        fit_rationale=(
            "Strong on the infrastructure half of this role and weak on the GenAI half. He has done the hard "
            "parts of production ML — feature stores, training-serving skew, shadow deploys, low-latency "
            "serving — in a regulated financial context, which maps well onto an insurance GCC. The gaps are "
            "real though: one RAG project against a brief wanting deep GenAI, no evidence of fine-tuning, and "
            "no stated team larger than three mentees against a requirement to have led eight. Worth "
            "shortlisting only if the client will trade GenAI tenure for platform rigour."
        ),
        requirement_matches=[
            RequirementMatch(
                requirement="10+ years of hands-on Generative AI engineering experience",
                verdict="absent",
                note="Nobody has this. See the market-impossibility flag below — this requirement should be renegotiated with the client.",
            ),
            RequirementMatch(
                requirement="12+ years total software engineering experience",
                verdict="partial",
                evidence="Data Analyst, Tata Consultancy Services, Jun 2017 - Jun 2019",
                note="Computed total is 9.1 years from Jun 2017. Three years short of the stated minimum.",
            ),
            RequirementMatch(
                requirement="Architecting and operating RAG systems at enterprise scale",
                verdict="partial",
                evidence="Shipped an internal RAG assistant over policy documents using LangChain and pgvector, now used daily by the underwriting team.",
                note="One internal RAG system in daily use by a single team. Real, but not enterprise scale, and not architected as a platform for others.",
            ),
            RequirementMatch(
                requirement="LLM fine-tuning (LoRA / QLoRA) on production workloads",
                verdict="unclear",
                evidence="LLM fine-tuning",
                note="Appears in the skills list only. No project, model or outcome anywhere in the CV supports it. Ask directly on the call.",
            ),
            RequirementMatch(
                requirement="MLOps: CI/CD for models, monitoring, drift detection",
                verdict="strong",
                evidence="Introduced shadow deployment for all model releases; rollback time went from a two-hour manual process to four minutes.",
                note="Also evidenced by the offline-to-online feature parity suite that caught a training-serving skew bug.",
            ),
            RequirementMatch(
                requirement="Hands-on Kubernetes and infrastructure-as-code",
                verdict="partial",
                evidence="Migrated model training from ad-hoc notebooks to Kubeflow Pipelines on GKE.",
                note="Kubernetes is evidenced through Kubeflow on GKE. Terraform is listed but never shown in use.",
            ),
            RequirementMatch(
                requirement="Real-time, low-latency inference serving",
                verdict="strong",
                evidence="real-time fraud scoring service handling ~4,200 requests per second at p99 latency of 40ms, written in Go with a PyTorch model served via TorchServe",
                note="The strongest single line on this CV and directly on point for the role.",
            ),
            RequirementMatch(
                requirement="Led a team of at least 8 engineers",
                verdict="absent",
                note="No team size stated at any role. The largest leadership evidence is mentoring three junior engineers.",
            ),
            RequirementMatch(
                requirement="Insurance or financial services domain experience",
                verdict="strong",
                evidence="Own the AI roadmap for Fintrail's lending risk products.",
                note="Financial services yes, across Razorpay, Fintrail and a TCS banking client. Insurance specifically, no.",
            ),
            RequirementMatch(
                requirement="Setting up an AI function inside a Global Capability Centre",
                verdict="absent",
            ),
            RequirementMatch(
                requirement="Vector databases at scale",
                verdict="partial",
                evidence="using LangChain and pgvector",
                note="pgvector on an internal tool. No indication of scale.",
            ),
        ],
        risk_flags=[
            RiskFlag(
                kind="jd_market_impossibility",
                severity="high",
                summary=(
                    "The brief asks for 10+ years of hands-on Generative AI engineering. Production GenAI "
                    "tooling is roughly four years old, so no candidate on the market can satisfy this. "
                    "Recommend renegotiating to 3+ years GenAI on top of 10+ years ML/platform engineering "
                    "before the search continues."
                ),
                evidence="10+ years of hands-on Generative AI engineering experience",
            ),
            RiskFlag(
                kind="claim_without_evidence",
                severity="medium",
                summary=(
                    "The summary claims expertise the body of the CV does not demonstrate. CUDA, TensorFlow "
                    "and LLM fine-tuning appear only in the skills list, with no project or outcome behind them."
                ),
                evidence="Expert in large-scale distributed systems and production machine learning. Deep specialisation in MLOps, real-time inference and GenAI platform architecture.",
            ),
            RiskFlag(
                kind="seniority_mismatch",
                severity="medium",
                summary=(
                    "The role requires hiring and running a team of six to eight. The CV evidences individual "
                    "contribution and mentoring of three, with no headcount, budget or hiring ownership stated."
                ),
                evidence="Mentored three junior engineers through the ML engineering ladder.",
            ),
        ],
        strengths=[
            "Production low-latency ML serving at genuine scale — 4,200 rps at 40ms p99.",
            "Deployment safety practices most candidates only talk about: shadow deploys, four-minute rollback.",
            "Caught and fixed a training-serving skew bug worth 12 points of recall.",
            "Nine years continuously in Indian financial services — Razorpay, Fintrail, a TCS banking client.",
        ],
        open_questions=[
            "What exactly was your role in the LLM fine-tuning work implied by your skills list? Which models, which technique, what shipped?",
            "How many engineers have you hired and managed directly, with headcount and reporting lines?",
            "The RAG assistant is used by the underwriting team — how many documents, how many queries a day, who maintains it now?",
            "What happened between Aug 2020 and Mar 2021?",
        ],
    )


def sample_dossier() -> Dossier:
    profile = sample_profile()
    timeline = build_timeline(profile)
    assessment = sample_assessment()
    flags = sort_flags(derive_risk_flags(profile, timeline) + assessment.risk_flags)

    usage = Usage(input_tokens=18432, output_tokens=3910, calls=3,
                  call_log=["extract_profile", "extract_job_brief", "assess"])

    # The real sample CV, so the source pane and quote verification have
    # something genuine to work against.
    root = Path(__file__).resolve().parent.parent
    cv_text = (root / "data" / "samples" / "cv_arjun_menon.txt").read_text(encoding="utf-8")
    jd_text = (root / "data" / "samples" / "jd_genai_platform_lead.txt").read_text(encoding="utf-8")

    document = DocumentText(
        text=cv_text,
        page_count=2,
        source_format="pdf",
        filename="cv_arjun_menon.pdf",
    )
    verification = verify_assessment(assessment, cv_text, jd_text)

    return Dossier(
        profile=profile,
        timeline=timeline,
        brief=sample_brief(),
        assessment=assessment,
        flags=flags,
        document=document,
        usage=usage,
        brief_text=jd_text,
        verification=verification,
        model="gpt-4o",
        elapsed_seconds=24.6,
        warnings=[],
    )
