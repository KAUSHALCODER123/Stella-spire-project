"""Run many CVs against one client brief, then rank them.

Two things make this more than a loop over `build_dossier`:

1. The brief is parsed ONCE and shared. Ten candidates against one job is
   1 + 10x2 model calls, not 10x3 -- a third of the assessment cost saved
   before any other optimisation.

2. Candidates run concurrently. The work is almost entirely waiting on the
   API, so a small thread pool turns ten minutes into about two. The pool is
   deliberately small: rate limits are the binding constraint, not CPU.

Progress is polled rather than pushed. A batch is a handful of items over a
couple of minutes; websockets would be more machinery than the problem needs.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.extract.llm import LLMError, Usage, extract_job_brief
from app.pipeline import Dossier, build_dossier
from app.schemas import JobBrief

log = logging.getLogger(__name__)

MAX_WORKERS = 4

# Ordered, so the UI can render a progress track rather than a spinner.
STAGES = ["queued", "reading", "extracting", "assessing", "verifying", "done"]


@dataclass
class BatchItem:
    filename: str
    path: Path
    index: int
    status: str = "queued"          # queued | running | done | failed
    stage: str = "queued"
    dossier_id: Optional[str] = None
    dossier: Optional[Dossier] = None
    error: Optional[str] = None
    elapsed: float = 0.0
    # Recruiter disposition belongs to the shortlist, not to the generated
    # dossier. It starts neutral and survives filtering/reordering in this run.
    decision: str = "unreviewed"     # unreviewed | shortlist | maybe | reject

    # --- ranking inputs, read straight off the finished dossier ----------

    @property
    def coverage(self) -> float:
        if not self.dossier:
            return -1.0
        c = self.dossier.must_have_coverage
        return c if c is not None else -1.0

    @property
    def strong(self) -> int:
        return self.dossier.match_counts["strong"] if self.dossier else 0

    @property
    def high_flags(self) -> int:
        return len(self.dossier.high_severity_flags) if self.dossier else 0

    @property
    def stage_index(self) -> int:
        try:
            return STAGES.index(self.stage)
        except ValueError:
            return 0


@dataclass
class Batch:
    id: str
    jd_text: str
    model: str
    anonymise: bool
    items: List[BatchItem]
    brief: Optional[JobBrief] = None
    usage: Usage = field(default_factory=Usage)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    error_kind: str = "error"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --- progress ---------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def completed(self) -> int:
        return sum(1 for i in self.items if i.status in ("done", "failed"))

    @property
    def succeeded(self) -> List[BatchItem]:
        return [i for i in self.items if i.status == "done" and i.dossier]

    @property
    def failed(self) -> List[BatchItem]:
        return [i for i in self.items if i.status == "failed"]

    @property
    def running(self) -> bool:
        return self.finished_at is None

    @property
    def percent(self) -> int:
        return int(self.completed / self.total * 100) if self.total else 100

    @property
    def elapsed(self) -> float:
        return round((self.finished_at or time.time()) - self.started_at, 1)

    def ranked(self) -> List[BatchItem]:
        """Best fit first.

        Sorted on must-have coverage, then strong matches, then fewest
        high-severity flags. Deliberately NOT a single blended score -- a
        recruiter needs to see which axis a candidate won on.
        """
        return sorted(
            self.succeeded,
            key=lambda i: (-i.coverage, -i.strong, i.high_flags, i.filename.lower()),
        )


# Process-local, like the dossier store. Swapping in SQLite is contained.
BATCHES: Dict[str, Batch] = {}


def create_batch(*, files: List[tuple], jd_text: str, model: str, anonymise: bool) -> Batch:
    """`files` is a list of (filename, saved_path)."""
    batch = Batch(
        id=uuid.uuid4().hex[:12],
        jd_text=jd_text,
        model=model,
        anonymise=anonymise,
        items=[BatchItem(filename=name, path=path, index=n) for n, (name, path) in enumerate(files)],
    )
    BATCHES[batch.id] = batch
    return batch


def run_batch(batch_id: str, store: Dict[str, Dossier]) -> None:
    """Execute a batch. Intended to run in a background thread."""
    batch = BATCHES.get(batch_id)
    if batch is None:
        return

    # Parse the brief once for everyone.
    try:
        batch.brief = extract_job_brief(batch.jd_text, usage=batch.usage, model=batch.model)
        log.info("batch %s: brief parsed, %d requirements", batch.id, len(batch.brief.requirements))
    except Exception as exc:  # noqa: BLE001
        log.error("batch %s: brief failed: %s", batch.id, exc)
        # Report the real cause. An account-level problem (no credits, bad
        # key) is not "the job description could not be parsed", and saying so
        # sends the user debugging the wrong thing.
        kind = getattr(exc, "kind", "error")
        batch.error = str(exc) if kind in ("quota", "auth", "model") else (
            "The job description could not be read: {}".format(exc))
        batch.error_kind = kind
        batch.finished_at = time.time()
        for item in batch.items:
            item.status = "failed"
            item.error = "Not analysed - the run stopped before this resume was reached."
        return

    def work(item: BatchItem) -> None:
        # A per-resume failure must not stop the rest of the run.
        started = time.perf_counter()
        item.status = "running"
        item.stage = "reading"
        try:
            dossier = build_dossier(
                cv_path=item.path,
                jd_text=batch.jd_text,
                model=batch.model,
                brief=batch.brief,
                on_stage=lambda s: setattr(item, "stage", s),
                display_name=item.filename,
            )
            dossier_id = uuid.uuid4().hex[:12]
            dossier.anonymise = batch.anonymise  # type: ignore[attr-defined]
            store[dossier_id] = dossier
            from app import db as _db
            _db.save_dossier(dossier_id, dossier)

            with batch._lock:
                batch.usage.input_tokens += dossier.usage.input_tokens
                batch.usage.output_tokens += dossier.usage.output_tokens
                batch.usage.calls += dossier.usage.calls

            item.dossier = dossier
            item.dossier_id = dossier_id
            item.stage = "done"
            item.status = "done"
        except Exception as exc:  # noqa: BLE001 - one bad CV must not kill the run
            log.error("batch %s: %s failed: %s", batch.id, item.filename, exc)
            item.error = str(exc)
            item.status = "failed"
            item.stage = "done"
        finally:
            item.elapsed = round(time.perf_counter() - started, 1)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(work, batch.items))

    batch.finished_at = time.time()
    log.info("batch %s finished in %.1fs (%d ok, %d failed)",
             batch.id, batch.elapsed, len(batch.succeeded), len(batch.failed))


def status_payload(batch: Batch) -> dict:
    """Small JSON blob for the polling UI."""
    return {
        "running": batch.running,
        "role_title": batch.brief.role_title if batch.brief else None,
        "client_name": batch.brief.client_name if batch.brief else None,
        "percent": batch.percent,
        "completed": batch.completed,
        "total": batch.total,
        "elapsed": batch.elapsed,
        "error": batch.error,
        "items": [
            {
                "index": i.index,
                "filename": i.filename,
                "status": i.status,
                "stage": i.stage,
                "stage_index": i.stage_index,
                "elapsed": i.elapsed,
                "error": i.error,
                "dossier_id": i.dossier_id,
                "coverage": round(i.coverage * 100) if i.coverage >= 0 else None,
                "strong": i.strong,
                "high_flags": i.high_flags,
                "decision": i.decision,
            }
            for i in batch.items
        ],
    }
