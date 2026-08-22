"""End-to-end: raw CV file + job description -> a complete dossier.

The order matters and is worth stating, because it is the argument for the
whole design:

    1. read the document          (no interpretation)
    2. extract facts              (model, evaluable against ground truth)
    3. compute the timeline       (arithmetic, always correct)
    4. derive arithmetic flags    (arithmetic, always correct)
    5. parse the client brief     (model)
    6. assess against the brief   (model, every claim carrying a quote)
    7. merge flags                (computed first, then judgement)

Steps 3, 4 and 7 are the reason a recruiter can trust the output. The model
never gets a chance to be wrong about a date.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.analysis import TimelineAnalysis, build_timeline, derive_risk_flags, sort_flags
from app.extract import llm
from app.extract.documents import DocumentText, extract_text
from app.schemas import Assessment, CandidateProfile, JobBrief, RiskFlag

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


def build_dossier(
    *,
    cv_path: str | Path,
    jd_text: str,
    usage: Optional[llm.Usage] = None,
) -> Dossier:
    started = time.perf_counter()
    usage = usage or llm.Usage()
    warnings: List[str] = []

    # 1. read
    document = extract_text(cv_path)
    warnings.extend(document.warnings)
    if not document.text.strip():
        raise ValueError("No readable text in {}. If this is a scanned PDF, it needs OCR first.".format(Path(cv_path).name))

    # 2. extract facts
    log.info("extracting profile (%d chars)", document.char_count)
    profile = llm.extract_profile(document.text, usage=usage)
    if not profile.positions:
        warnings.append("No work history could be extracted. The dossier below will be thin.")

    # 3 + 4. arithmetic
    timeline = build_timeline(profile)
    computed_flags = derive_risk_flags(profile, timeline)

    # 5. client brief
    log.info("parsing client brief")
    brief = llm.extract_job_brief(jd_text, usage=usage)

    # 6. assessment
    log.info("assessing against %d requirements", len(brief.requirements))
    assessment = llm.assess(
        profile=profile,
        timeline=timeline,
        brief=brief,
        cv_text=document.text,
        usage=usage,
    )

    # 7. merge: computed flags first, they are the ones that are always right
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
        elapsed_seconds=round(time.perf_counter() - started, 1),
        warnings=warnings,
    )
