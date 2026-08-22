"""Run many CVs against many roles.

Phases, in order, because each one depends on the last:

    1. briefs      parse every job description        M model calls
    2. extract     read every CV into a profile       N model calls
    3. screen      affinity for all M*N pairs         0 model calls
    4. assess      full assessment on selected pairs  K model calls

Steps 1 and 2 are independent of each other and of everything downstream, so
each runs concurrently within its phase. Step 3 is instant. Step 4 is the only
expensive phase, and the screen in step 3 is what keeps K well below M*N.

A failure is always local: a CV that cannot be read fails its own row, a brief
that cannot be parsed fails its own column, and everything else still runs.
The only global failure is an account-level one (no credit, bad key), which
would fail every call anyway.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.analysis import TimelineAnalysis, build_timeline, derive_risk_flags, sort_flags
from app.extract import llm
from app.extract.documents import DocumentText, extract_text
from app.matching import (
    DEFAULT_MIN_AFFINITY,
    DEFAULT_MIN_TERM_RATIO,
    DEFAULT_TOP_ROLES,
    Affinity,
    estimate_calls,
    plan_pairs,
    score_affinity,
)
from app.pipeline import Dossier
from app.schemas import CandidateProfile, JobBrief, RiskFlag
from app.verify import verify_assessment

log = logging.getLogger(__name__)

MAX_WORKERS = 4
PHASES = ["queued", "briefs", "extracting", "screening", "assessing", "done"]


@dataclass
class Requisition:
    """One role in a run."""

    index: int
    filename: str
    jd_text: str
    brief: Optional[JobBrief] = None
    error: Optional[str] = None

    @property
    def title(self) -> str:
        if self.brief and self.brief.role_title:
            return self.brief.role_title
        return Path(self.filename).stem.replace("_", " ").replace("-", " ").title()

    @property
    def client(self) -> str:
        return (self.brief.client_name if self.brief else None) or "Unspecified client"

    @property
    def ok(self) -> bool:
        return self.brief is not None and self.error is None


@dataclass
class Candidate:
    """One CV in a run, extracted exactly once and reused for every role."""

    index: int
    filename: str
    path: Path
    profile: Optional[CandidateProfile] = None
    timeline: Optional[TimelineAnalysis] = None
    document: Optional[DocumentText] = None
    computed_flags: List[RiskFlag] = field(default_factory=list)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.profile is not None and self.error is None

    @property
    def name(self) -> str:
        if self.profile and self.profile.full_name:
            return self.profile.full_name
        return Path(self.filename).stem.replace("_", " ").title()

    @property
    def years(self) -> float:
        return self.timeline.total_experience_years if self.timeline else 0.0


@dataclass
class Pair:
    candidate_index: int
    requisition_index: int
    affinity: Affinity
    selected: bool
    reason: str
    status: str = "screened"          # screened | queued | running | done | failed | skipped
    dossier_id: Optional[str] = None
    dossier: Optional[Dossier] = None
    error: Optional[str] = None

    @property
    def suitability(self) -> Optional[int]:
        return self.dossier.suitability["percent"] if self.dossier else None

    @property
    def sort_key(self) -> Tuple:
        """Assessed pairs always outrank screened-only ones.

        A real assessment and an affinity guess are different kinds of number
        and must never be sorted into the same ordering as if comparable.
        """
        return (0 if self.dossier else 1, -(self.suitability or 0), -self.affinity.score)


@dataclass
class MatchRun:
    id: str
    model: str
    anonymise: bool
    requisitions: List[Requisition]
    candidates: List[Candidate]
    top_roles: int = DEFAULT_TOP_ROLES
    assess_all: bool = False
    extraction_model: str = ""

    pairs: List[Pair] = field(default_factory=list)
    phase: str = "queued"
    usage: llm.Usage = field(default_factory=llm.Usage)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    error_kind: str = "error"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --- progress ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.finished_at is None

    @property
    def elapsed(self) -> float:
        return round((self.finished_at or time.time()) - self.started_at, 1)

    @property
    def selected_pairs(self) -> List[Pair]:
        return [p for p in self.pairs if p.selected]

    @property
    def assessed_pairs(self) -> List[Pair]:
        return [p for p in self.pairs if p.dossier]

    @property
    def completed(self) -> int:
        done = sum(1 for r in self.requisitions if r.brief or r.error)
        done += sum(1 for c in self.candidates if c.profile or c.error)
        done += sum(1 for p in self.selected_pairs if p.status in ("done", "failed"))
        return done

    @property
    def total_steps(self) -> int:
        return len(self.requisitions) + len(self.candidates) + max(len(self.selected_pairs), 0)

    @property
    def percent(self) -> int:
        total = self.total_steps
        return int(self.completed / total * 100) if total else 100

    # --- views ------------------------------------------------------------

    def pair(self, ci: int, ri: int) -> Optional[Pair]:
        for p in self.pairs:
            if p.candidate_index == ci and p.requisition_index == ri:
                return p
        return None

    def shortlist_for(self, ri: int) -> List[Pair]:
        """Candidates for one role, best first."""
        rows = [p for p in self.pairs if p.requisition_index == ri and (p.dossier or p.selected)]
        return sorted(rows, key=lambda p: p.sort_key)

    def roles_for(self, ci: int) -> List[Pair]:
        """Roles one candidate was matched to, best first. These are the tags."""
        rows = [p for p in self.pairs if p.candidate_index == ci and (p.dossier or p.selected)]
        return sorted(rows, key=lambda p: p.sort_key)

    def cost(self) -> Dict[str, int]:
        return estimate_calls(
            n_candidates=len(self.candidates),
            n_requisitions=len(self.requisitions),
            n_selected=len(self.selected_pairs),
        )


RUNS: Dict[str, MatchRun] = {}


def create_run(
    *,
    jds: List[Tuple[str, str]],
    cvs: List[Tuple[str, Path]],
    model: str,
    anonymise: bool,
    top_roles: int = DEFAULT_TOP_ROLES,
    assess_all: bool = False,
    extraction_model: str = "",
) -> MatchRun:
    """`jds` is (filename, text); `cvs` is (filename, saved path)."""
    run = MatchRun(
        id=uuid.uuid4().hex[:12],
        model=model,
        anonymise=anonymise,
        top_roles=top_roles,
        assess_all=assess_all,
        extraction_model=extraction_model,
        requisitions=[Requisition(index=i, filename=n, jd_text=t) for i, (n, t) in enumerate(jds)],
        candidates=[Candidate(index=i, filename=n, path=p) for i, (n, p) in enumerate(cvs)],
    )
    RUNS[run.id] = run
    return run


def execute(run_id: str, store: Dict[str, Dossier]) -> None:
    """Run a match. Intended for a background thread."""
    run = RUNS.get(run_id)
    if run is None:
        return
    try:
        _execute(run, store)
    except Exception as exc:  # noqa: BLE001 - the run must always terminate
        log.error("run %s crashed: %s", run.id, exc, exc_info=True)
        run.error = str(exc)
        run.error_kind = getattr(exc, "kind", "error")
    finally:
        run.phase = "done"
        run.finished_at = time.time()


def _execute(run: MatchRun, store: Dict[str, Dossier]) -> None:
    # --- 1. briefs --------------------------------------------------------
    run.phase = "briefs"

    def parse_brief(req: Requisition) -> None:
        try:
            req.brief = llm.extract_job_brief(req.jd_text, usage=run.usage, model=run.model)
        except Exception as exc:  # noqa: BLE001
            log.error("run %s: brief %s failed: %s", run.id, req.filename, exc)
            req.error = str(exc)
            if getattr(exc, "kind", "") in ("quota", "auth", "model"):
                run.error, run.error_kind = str(exc), exc.kind  # type: ignore[attr-defined]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(parse_brief, run.requisitions))

    if run.error_kind in ("quota", "auth", "model"):
        return  # account-level: every remaining call would fail identically

    # --- 2. extract -------------------------------------------------------
    run.phase = "extracting"

    def extract(cand: Candidate) -> None:
        try:
            doc = extract_text(cand.path)
            doc.filename = cand.filename
            if not doc.text.strip():
                raise ValueError("No readable text. If this is a scanned PDF it needs OCR first.")
            cand.document = doc
            cand.warnings = list(doc.warnings)
            cand.profile = llm.extract_profile(
                doc.text, usage=run.usage, model=run.extraction_model or run.model)
            cand.timeline = build_timeline(cand.profile)
            cand.computed_flags = derive_risk_flags(cand.profile, cand.timeline)
        except Exception as exc:  # noqa: BLE001
            log.error("run %s: cv %s failed: %s", run.id, cand.filename, exc)
            cand.error = str(exc)
            if getattr(exc, "kind", "") in ("quota", "auth", "model"):
                run.error, run.error_kind = str(exc), exc.kind  # type: ignore[attr-defined]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(extract, run.candidates))

    if run.error_kind in ("quota", "auth", "model"):
        return

    # --- 3. screen (free) -------------------------------------------------
    run.phase = "screening"
    affinities: Dict[Tuple[int, int], Affinity] = {}
    for cand in run.candidates:
        if not cand.ok:
            continue
        for req in run.requisitions:
            if not req.ok:
                continue
            affinities[(cand.index, req.index)] = score_affinity(cand.profile, cand.timeline, req.brief)

    plans = plan_pairs(
        affinities,
        n_candidates=len(run.candidates),
        n_requisitions=len(run.requisitions),
        top_roles=run.top_roles,
        assess_all=run.assess_all,
    )
    run.pairs = [
        Pair(candidate_index=p.candidate_index, requisition_index=p.requisition_index,
             affinity=p.affinity, selected=p.selected, reason=p.reason,
             status="queued" if p.selected else "skipped")
        for p in plans
    ]

    # --- 4. assess the selected pairs -------------------------------------
    run.phase = "assessing"

    def assess(pair: Pair) -> None:
        cand = run.candidates[pair.candidate_index]
        req = run.requisitions[pair.requisition_index]
        pair.status = "running"
        try:
            assessment = llm.assess(
                profile=cand.profile, timeline=cand.timeline, brief=req.brief,
                cv_text=cand.document.text, usage=run.usage, model=run.model,
            )
            verification = verify_assessment(assessment, cand.document.text, req.jd_text)

            for match in assessment.requirement_matches:
                if match.verdict == "strong" and not match.evidence:
                    match.verdict = "unclear"
                    match.note = (match.note + " " if match.note else "") + \
                        "[Demoted: no supporting quote was produced.]"

            dossier = Dossier(
                profile=cand.profile, timeline=cand.timeline, brief=req.brief,
                assessment=assessment,
                flags=sort_flags(cand.computed_flags + assessment.risk_flags),
                document=cand.document, usage=llm.Usage(), brief_text=req.jd_text,
                verification=verification, model=run.model, warnings=list(cand.warnings),
            )
            dossier.anonymise = run.anonymise  # type: ignore[attr-defined]
            dossier_id = uuid.uuid4().hex[:12]
            store[dossier_id] = dossier
            pair.dossier, pair.dossier_id, pair.status = dossier, dossier_id, "done"
        except Exception as exc:  # noqa: BLE001 - one pair must not stop the run
            log.error("run %s: pair c%d/r%d failed: %s", run.id, pair.candidate_index,
                      pair.requisition_index, exc)
            pair.error, pair.status = str(exc), "failed"
            if getattr(exc, "kind", "") in ("quota", "auth"):
                run.error, run.error_kind = str(exc), exc.kind  # type: ignore[attr-defined]

    # Grouped by role: consecutive calls then share the brief prefix, which is
    # what makes prompt caching pay. Ungrouped, every call is a cache miss.
    selected = sorted(run.selected_pairs, key=lambda p: p.requisition_index)
    if selected:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(assess, selected))

    log.info("run %s finished in %.1fs: %d roles, %d candidates, %d assessed",
             run.id, run.elapsed, len(run.requisitions), len(run.candidates), len(run.assessed_pairs))


def status_payload(run: MatchRun) -> dict:
    return {
        "running": run.running,
        "phase": run.phase,
        "percent": run.percent,
        "completed": run.completed,
        "total": run.total_steps,
        "elapsed": run.elapsed,
        "error": run.error,
        "error_kind": run.error_kind,
        "roles": [
            {"index": r.index, "title": r.title, "ok": r.ok, "error": r.error}
            for r in run.requisitions
        ],
        "candidates": [
            {"index": c.index, "filename": c.filename, "name": c.name if c.ok else c.filename,
             "ok": c.ok, "error": c.error}
            for c in run.candidates
        ],
        "assessed": len(run.assessed_pairs),
        "selected": len(run.selected_pairs),
    }
