"""End-to-end: raw CV file + job description -> a complete dossier.

The order matters and is worth stating, because it is the argument for the
whole design:

    1. read the document          (no interpretation)
    2. extract facts              (model, evaluable against ground truth)
    3. compute the timeline       (arithmetic, always correct)
    4. derive arithmetic flags    (arithmetic, always correct)
    5. parse the client brief     (model)
    6. assess against the brief   (model, every claim carrying a quote)
    7. verify every quote          (string search against the CV)
    8. merge flags                 (computed first, then judgement)

Steps 3, 4 and 7 are the reason a recruiter can trust the output. The model
never gets a chance to be wrong about a date.

Steps 2 and 5 read different documents and neither needs the other's
output, so on a single-CV run they fire concurrently -- assessment (6) is
the first step that actually needs both, and it still waits for them in
order below.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from app.analysis import TimelineAnalysis, build_timeline, derive_risk_flags, sort_flags
from app.extract import llm
from app.extract.documents import DocumentText, extract_text
from app.schemas import Assessment, CandidateProfile, JobBrief, RiskFlag
from app.verify import VerificationReport, verify_assessment

log = logging.getLogger(__name__)


@dataclass
class Dossier:
    """Everything needed to render a candidate dossier."""

    profile: CandidateProfile
    timeline: TimelineAnalysis
    brief: JobBrief
    assessment: Assessment
    flags: List[RiskFlag]

    document: DocumentText
    usage: llm.Usage
    brief_text: str = ""
    verification: Optional[VerificationReport] = None
    model: str = ""
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    # --- summary numbers used by the templates ---------------------------

    @property
    def match_counts(self) -> dict:
        counts = {"strong": 0, "partial": 0, "absent": 0, "unclear": 0}
        for m in self.assessment.requirement_matches:
            counts[m.verdict] = counts.get(m.verdict, 0) + 1
        return counts

    @property
    def must_have_coverage(self) -> Optional[float]:
        """Share of must-have requirements rated strong or partial, 0-1.

        Reported rather than compressed into a single "match score", because a
        percentage that blends must-haves with nice-to-haves is a number nobody
        can act on.
        """
        must = {r.text for r in self.brief.requirements if r.kind == "must_have"}
        if not must:
            return None
        matched = [
            m for m in self.assessment.requirement_matches
            if m.requirement in must and m.verdict in ("strong", "partial")
        ]
        return len(matched) / len(must)

    @property
    def high_severity_flags(self) -> List[RiskFlag]:
        return [f for f in self.flags if f.severity == "high"]

    # --- scores for the match report -------------------------------------

    @property
    def matched_requirements(self) -> List:
        return [m for m in self.assessment.requirement_matches if m.verdict in ("strong", "partial")]

    @property
    def missing_requirements(self) -> List:
        return [m for m in self.assessment.requirement_matches if m.verdict in ("absent", "unclear")]

    @property
    def experience_match(self) -> dict:
        """Candidate years against the years the brief asks for."""
        required = self.brief.stated_min_years
        actual = self.timeline.total_experience_years
        span = self.timeline.career_span_years
        out = self.timeline.months_out_of_work
        # `effective` is the figure every verdict is actually reached on, so it
        # belongs in the dict on every path. Leaving it out when no minimum is
        # stated makes a template that reads it render blank rather than fail.
        base = {"required": None, "actual": actual, "span": span, "months_out": out,
                "ratio": 1.0, "verdict": "not specified", "shortfall": 0.0,
                "effective": max(actual, span) if out else actual}
        if not required:
            return base

        # Measured against the calendar, not against billed months. Someone who
        # took eighteen months out is not eighteen months less senior, and a
        # brief asking for "15+ years" is reaching for seniority. The worked
        # experience figure is still shown next to it, so nothing is hidden.
        effective = max(actual, span) if out else actual

        ratio = min(effective / required, 1.0)
        shortfall = round(max(required - effective, 0.0), 1)
        if effective >= required:
            verdict = "meets"
        elif ratio >= 0.8:
            verdict = "close"
        else:
            verdict = "short"
        base.update({"required": required, "ratio": ratio, "verdict": verdict,
                     "shortfall": shortfall, "effective": effective})
        return base

    @property
    def skill_stats(self) -> dict:
        total = len(self.profile.skills)
        evidenced = sum(1 for s in self.profile.skills if s.evidence)
        return {
            "total": total,
            "evidenced": evidenced,
            "listed_only": total - evidenced,
            "ratio": evidenced / total if total else 0.0,
        }

    @property
    def suitability(self) -> dict:
        """A single headline number, shown alongside the parts it is made of.

        Weighted so must-have coverage dominates, with a penalty for
        high-severity flags. The components are always displayed next to it --
        a score a recruiter cannot decompose is a score they cannot argue with,
        which makes it useless.
        """
        coverage = self.must_have_coverage
        coverage = coverage if coverage is not None else 0.0

        matches = self.assessment.requirement_matches
        strength = (sum(1 for m in matches if m.verdict == "strong") / len(matches)) if matches else 0.0

        exp = self.experience_match["ratio"]
        penalty = min(0.15, 0.05 * len(self.high_severity_flags))

        score = max(0.0, min(1.0, 0.55 * coverage + 0.25 * strength + 0.20 * exp - penalty))
        if score >= 0.72:
            band, tone = "Strong fit", "ok"
        elif score >= 0.45:
            band, tone = "Possible fit", "warn"
        else:
            band, tone = "Weak fit", "bad"

        return {
            "score": score,
            "percent": round(score * 100),
            "band": band,
            "tone": tone,
            "coverage": coverage,
            "strength": strength,
            "experience": exp,
            "penalty": penalty,
        }


def build_dossier(
    *,
    cv_path: str | Path,
    jd_text: str,
    usage: Optional[llm.Usage] = None,
    model: Optional[str] = None,
    brief: Optional[JobBrief] = None,
    on_stage: Optional[Callable[[str], None]] = None,
    display_name: Optional[str] = None,
) -> Dossier:
    """Build one dossier.

    `brief` lets a batch parse the client job description once and reuse it
    across every candidate, saving N-1 model calls.
    `on_stage` reports progress for the batch UI.
    `display_name` is the filename the user recognises; without it the report
    would cite the uuid-prefixed name we store on disk.
    """
    from app.config import settings

    started = time.perf_counter()
    usage = usage or llm.Usage()
    model = model or settings.model
    warnings: List[str] = []

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name)

    # 1. read
    stage("reading")
    document = extract_text(cv_path)
    if display_name:
        document.filename = display_name
    warnings.extend(document.warnings)
    if not document.text.strip():
        raise ValueError("No readable text in {}. If this is a scanned PDF, it needs OCR first.".format(Path(cv_path).name))

    # 2. extract facts, and 5. the client brief -- run together when both are
    # needed. They read different documents and neither depends on the
    # other's output, so waiting for one before starting the other was pure
    # latency: on a single-CV run this halves the time spent waiting on a
    # model before assessment can even begin.
    stage("extracting")
    log.info("extracting profile (%d chars)", document.char_count)
    extraction_model = settings.extraction_model or model
    if brief is None:
        stage("brief")
        log.info("parsing client brief")
        with ThreadPoolExecutor(max_workers=2) as pool:
            profile_future = pool.submit(
                llm.extract_profile, document.text, usage=usage, model=extraction_model)
            brief_future = pool.submit(
                llm.extract_job_brief, jd_text, usage=usage, model=model)
            profile = profile_future.result()
            brief = brief_future.result()
    else:
        profile = llm.extract_profile(document.text, usage=usage, model=extraction_model)

    if not profile.positions:
        warnings.append("No work history could be extracted. The dossier below will be thin.")

    # 3 + 4. arithmetic
    timeline = build_timeline(profile)
    computed_flags = derive_risk_flags(profile, timeline, document.text)

    # 6. assessment
    stage("assessing")
    log.info("assessing against %d requirements", len(brief.requirements))
    assessment = llm.assess(
        profile=profile,
        timeline=timeline,
        brief=brief,
        cv_text=document.text,
        usage=usage,
        model=model,
    )

    # 7. verify every model-written quote actually appears in the CV. The
    #    prompt asks for verbatim quotes; this is what checks.
    stage("verifying")
    verification = verify_assessment(assessment, document.text, jd_text)
    if verification.unverified:
        warnings.append(
            "{} of {} quotes could not be located in the source CV and are marked UNVERIFIED.".format(
                len(verification.unverified), verification.total
            )
        )

    # 8. merge: computed flags first, they are the ones that are always right
    flags = sort_flags(computed_flags + assessment.risk_flags)

    # A 'strong' verdict with no quote violates the prompt contract. Rather
    # than trust it, demote it and say so -- silent acceptance here would
    # undermine the one guarantee the product makes.
    for match in assessment.requirement_matches:
        if match.verdict == "strong" and not match.evidence:
            match.verdict = "unclear"
            match.note = (match.note + " " if match.note else "") + "[Demoted: no supporting quote was produced.]"
            warnings.append("A requirement match was demoted because the model gave no quote for it.")

    return Dossier(
        profile=profile,
        timeline=timeline,
        brief=brief,
        assessment=assessment,
        flags=flags,
        document=document,
        usage=usage,
        brief_text=jd_text,
        verification=verification,
        model=model,
        elapsed_seconds=round(time.perf_counter() - started, 1),
        warnings=warnings,
    )
