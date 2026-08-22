"""FastAPI app: upload a CV and a brief, get a dossier.

One process, one language, no build step. Jinja2 templates server-side, the
rendered dossier shown in an iframe so its print stylesheet cannot collide with
the app chrome, and a PDF endpoint that streams the same HTML through Chromium.

Dossiers are held in a process-local dict. That is deliberate for a demo: no
database to provision, and nothing persists after shutdown. Swapping in SQLite
is a contained change to `STORE`.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.templating import Jinja2Templates

from app.batch import BATCHES, create_batch, run_batch, status_payload
from app.matchrun import RUNS, create_run, execute as execute_run
from app.matchrun import status_payload as run_status
from app.extract.llm import LLMError
from app.config import MODEL_CHOICES, settings
from app.uploads import display_filename, safe_filename, upload_suffix
from app.extract.documents import SUPPORTED_SUFFIXES, extract_text
from app.pipeline import Dossier, build_dossier
from app.render.dossier import COMPUTED_KINDS, _skills_by_category, candidate_ref, render_html, render_pdf
from app.render.source import render_source
from app.render.redact import redact_dossier, redact_text
from app.verify import verify_assessment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spiredossier")

app = FastAPI(title="SpireDossier")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

STORE: Dict[str, Dossier] = {}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

def _sample_paths() -> tuple[Path, Path]:
    return (
        settings.sample_dir / "cv_arjun_menon.txt",
        settings.sample_dir / "jd_genai_platform_lead.txt",
    )


def _has_key() -> bool:
    return bool(settings.openai_api_key)


def chrome(nav: str) -> dict:
    """Context every page needs for the sidebar shell."""
    return {
        "agency": settings.agency_name,
        "model": settings.model,
        "has_key": _has_key(),
        "nav": nav,
        "nav_counts": {"candidates": len(STORE), "shortlists": len(BATCHES) + len(RUNS)},
    }


def _initials(name: Optional[str], fallback: str) -> str:
    parts = [p for p in (name or "").split() if p]
    if not parts:
        return fallback[:2].upper()
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()


def _tone(pct: int) -> str:
    return "ok" if pct >= 72 else ("warn" if pct >= 45 else "bad")


_model_cache: Optional[list] = None


def available_models() -> list:
    """MODEL_CHOICES filtered to what this key can actually reach.

    Offering a model the account cannot call is a demo-day failure waiting to
    happen -- gpt-5 is not on every account. Looked up once, then cached; if
    the lookup fails we show everything rather than block the UI.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not _has_key():
        _model_cache = list(MODEL_CHOICES)
        return _model_cache
    try:
        from app.extract.llm import get_client

        ids = {m.id for m in get_client().models.list()}
        filtered = [c for c in MODEL_CHOICES if c[0] in ids]
        _model_cache = filtered or list(MODEL_CHOICES)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not list models (%s); showing all choices", exc)
        _model_cache = list(MODEL_CHOICES)
    return _model_cache


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    _, jd_sample = _sample_paths()

    recent = []
    for did, d in list(STORE.items())[-5:][::-1]:
        pct = d.suitability["percent"]
        anon = bool(getattr(d, "anonymise", settings.anonymise_by_default))
        name = candidate_ref(d) if anon else (d.profile.full_name or "Unnamed")
        recent.append({
            "id": did, "name": name, "role": d.brief.role_title,
            "initials": _initials(None if anon else d.profile.full_name, "SD"),
            "percent": pct, "tone": _tone(pct),
        })

    batches = []
    for bid, b in list(BATCHES.items())[-5:][::-1]:
        batches.append({
            "id": bid, "role": b.brief.role_title if b.brief else "Reading the brief…",
            "total": b.total, "elapsed": int(b.elapsed), "running": b.running,
        })

    best = max((d.suitability["percent"] for d in STORE.values()), default=0)

    ctx = chrome("dashboard")
    ctx.update({
        "model_choices": available_models(),
        "sample_jd": jd_sample.read_text(encoding="utf-8") if jd_sample.exists() else "",
        "recent": recent,
        "recent_batches": batches,
        "stats": {"dossiers": len(STORE), "shortlists": len(BATCHES), "best_match": best},
    })
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/candidates", response_class=HTMLResponse)
def candidates(request: Request):
    rows = []
    for did, d in list(STORE.items())[::-1]:
        pct = d.suitability["percent"]
        anon = bool(getattr(d, "anonymise", settings.anonymise_by_default))
        name = candidate_ref(d) if anon else (d.profile.full_name or "Unnamed candidate")
        rows.append({
            "href": "/dossier/{}".format(did),
            "badge": _initials(None if anon else d.profile.full_name, "SD"),
            "title": name,
            "subtitle": "{} · {} yrs experience".format(d.brief.role_title, d.timeline.total_experience_years),
            "meta": "{} of {} requirements".format(len(d.matched_requirements), len(d.assessment.requirement_matches)),
            "pill": "{}%".format(pct), "pill_tone": _tone(pct), "square": False,
        })
    ctx = chrome("candidates")
    ctx.update({
        "heading": "Candidates", "rows": rows,
        "empty_title": "No resumes analysed yet",
        "empty_detail": "Upload a resume and a job description to build your first match report.",
    })
    return templates.TemplateResponse(request, "list.html", ctx)


@app.get("/shortlists", response_class=HTMLResponse)
def shortlists(request: Request):
    rows = []
    for rid, r in list(RUNS.items())[::-1]:
        rows.append({
            "href": "/match/{}".format(rid),
            "badge": "{}x{}".format(len(r.requisitions), len(r.candidates)),
            "title": "{} roles x {} candidates".format(len(r.requisitions), len(r.candidates)),
            "subtitle": ", ".join(q.title for q in r.requisitions[:3]) or "Reading briefs...",
            "meta": "{}s".format(int(r.elapsed)),
            "pill": "Running" if r.running else "Done",
            "pill_tone": "info" if r.running else "ok",
            "square": True,
        })
    for bid, b in list(BATCHES.items())[::-1]:
        rows.append({
            "href": "/batch/{}".format(bid),
            "badge": str(b.total),
            "title": b.brief.role_title if b.brief else "Reading the brief…",
            "subtitle": "{} candidates · {} analysed".format(b.total, len(b.succeeded)),
            "meta": "{}s".format(int(b.elapsed)),
            "pill": "Running" if b.running else "Done",
            "pill_tone": "info" if b.running else "ok",
            "square": True,
        })
    ctx = chrome("shortlists")
    ctx.update({
        "heading": "Shortlists", "rows": rows,
        "empty_title": "No shortlists yet",
        "empty_detail": "Upload several resumes at once to rank them against one brief.",
    })
    return templates.TemplateResponse(request, "list.html", ctx)


@app.post("/generate")
async def generate(
    request: Request,
    cv: List[UploadFile] = File(default=[]),
    jd_text: str = Form(""),
    anonymise: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
):
    """One CV builds a dossier; several start a batch. Same form either way."""
    uploads = [f for f in (cv or []) if f and f.filename]

    if not uploads:
        return _error(request, "No CV uploaded.", "Choose one or more PDF, DOCX or TXT files, or open the sample.")
    if not jd_text.strip():
        return _error(request, "No job description.", "Paste the client brief so requirements can be matched against it.")
    if not _has_key():
        return _error(
            request,
            "No API key configured.",
            "Copy .env.example to .env, add your OPENAI_API_KEY, then restart the server. "
            "Run `python -m scripts.check_setup` to verify it.",
        )

    saved: List[tuple] = []
    for upload in uploads:
        suffix = upload_suffix(upload.filename)
        if suffix not in SUPPORTED_SUFFIXES:
            return _error(
                request,
                "Unsupported file: {}".format(upload.filename),
                "Supported formats are {}. Remove that file and try again.".format(", ".join(sorted(SUPPORTED_SUFFIXES))),
            )
        payload = await upload.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            return _error(
                request,
                "File too large: {}".format(upload.filename),
                "The limit is 10 MB; this one is {:.1f} MB.".format(len(payload) / 1e6),
            )
        path = settings.upload_dir / "{}_{}".format(uuid.uuid4().hex[:8], safe_filename(upload.filename))
        path.write_bytes(payload)
        # Display name stays the original; only the stored path is sanitised.
        saved.append((Path(upload.filename.replace("\\", "/")).name or "resume", path))

    chosen = model if model in {m[0] for m in available_models()} else settings.model

    # --- several CVs: run as a batch, report progress ---------------------
    if len(saved) > 1:
        batch = create_batch(files=saved, jd_text=jd_text, model=chosen, anonymise=bool(anonymise))
        threading.Thread(target=run_batch, args=(batch.id, STORE), daemon=True).start()
        return RedirectResponse("/batch/{}".format(batch.id), status_code=303)

    # --- one CV: straight through -----------------------------------------
    try:
        dossier = build_dossier(
            cv_path=saved[0][1], jd_text=jd_text, model=chosen, display_name=saved[0][0]
        )
    except LLMError as exc:
        log.error("pipeline failed (%s): %s", exc.kind, exc)
        titles = {
            "quota": "The AI account is out of credit",
            "auth": "The API key was not accepted",
            "model": "That model is not available",
            "not_a_cv": "That file does not look like a CV",
            "truncated": "The response was cut off",
        }
        return _error(request, titles.get(exc.kind, "The analysis could not be completed"), str(exc))
    except ValueError as exc:
        log.warning("unreadable document: %s", exc)
        return _error(request, "That file could not be read", str(exc))
    except Exception as exc:  # noqa: BLE001
        log.error("pipeline failed: %s", traceback.format_exc())
        return _error(request, "The analysis could not be completed", str(exc))

    dossier_id = uuid.uuid4().hex[:12]
    dossier.anonymise = bool(anonymise)  # type: ignore[attr-defined]
    STORE[dossier_id] = dossier
    return RedirectResponse("/dossier/{}".format(dossier_id), status_code=303)


@app.post("/match")
async def match(
    request: Request,
    cv: List[UploadFile] = File(default=[]),
    jd: List[UploadFile] = File(default=[]),
    jd_text: str = Form(""),
    anonymise: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    top_roles: int = Form(3),
    assess_all: Optional[str] = Form(None),
):
    """Many CVs against many roles."""
    cv_uploads = [f for f in (cv or []) if f and f.filename]
    jd_uploads = [f for f in (jd or []) if f and f.filename]

    if not cv_uploads:
        return _error(request, "No CVs uploaded.", "Add at least one CV to match against the roles.")
    if not jd_uploads and not jd_text.strip():
        return _error(request, "No roles provided.",
                      "Upload one or more job description files, or paste a single brief.")
    if not _has_key():
        return _error(request, "No API key configured.",
                      "Copy .env.example to .env, add your OPENAI_API_KEY, then restart the server.")

    # --- roles -----------------------------------------------------------
    jds: List[tuple] = []
    for upload in jd_uploads:
        if upload_suffix(upload.filename) not in SUPPORTED_SUFFIXES:
            return _error(request, "Unsupported job description: {}".format(upload.filename),
                          "Supported formats are {}.".format(", ".join(sorted(SUPPORTED_SUFFIXES))))
        payload = await upload.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            return _error(request, "Job description too large: {}".format(upload.filename),
                          "The limit is 10 MB per file.")
        path = settings.upload_dir / "{}_{}".format(uuid.uuid4().hex[:8], safe_filename(upload.filename))
        path.write_bytes(payload)
        try:
            text = extract_text(path).text
        except Exception as exc:  # noqa: BLE001
            return _error(request, "Could not read {}".format(display_filename(upload.filename)), str(exc))
        if not text.strip():
            return _error(request, "Empty job description: {}".format(display_filename(upload.filename)),
                          "That file contained no readable text.")
        jds.append((display_filename(upload.filename), text))

    if jd_text.strip():
        jds.append(("Pasted brief", jd_text))

    # --- candidates -------------------------------------------------------
    cvs: List[tuple] = []
    for upload in cv_uploads:
        if upload_suffix(upload.filename) not in SUPPORTED_SUFFIXES:
            return _error(request, "Unsupported file: {}".format(upload.filename),
                          "Supported formats are {}.".format(", ".join(sorted(SUPPORTED_SUFFIXES))))
        payload = await upload.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            return _error(request, "File too large: {}".format(upload.filename),
                          "The limit is 10 MB; this one is {:.1f} MB.".format(len(payload) / 1e6))
        path = settings.upload_dir / "{}_{}".format(uuid.uuid4().hex[:8], safe_filename(upload.filename))
        path.write_bytes(payload)
        cvs.append((display_filename(upload.filename), path))

    chosen = model if model in {m[0] for m in available_models()} else settings.model
    run = create_run(
        jds=jds, cvs=cvs, model=chosen, anonymise=bool(anonymise),
        top_roles=max(1, min(int(top_roles or 3), 10)),
        assess_all=bool(assess_all),
        extraction_model=settings.extraction_model or chosen,
    )
    threading.Thread(target=execute_run, args=(run.id, STORE), daemon=True).start()
    return RedirectResponse("/match/{}".format(run.id), status_code=303)


@app.post("/match/demo")
def match_demo(request: Request):
    """A finished many-to-many run built from stored fixtures. No model call.

    Exists for the same reason /demo does: the matrix, the shortlists and the
    role tags need to be reviewable without spending anything.
    """
    import copy as _copy

    from app.matching import Affinity
    from app.matchrun import Candidate, MatchRun, Pair, Requisition
    from tests.fixtures import sample_dossier

    base = sample_dossier()
    roles = [
        ("GenAI Platform Lead", "Confidential GCC (US insurer)", [82, 55, None]),
        ("Senior ML Engineer", "FinTech, Bengaluru", [74, 68, 31]),
        ("VP Finance", "NBFC, Mumbai", [None, None, None]),
    ]
    people = [("Arjun Menon", 8.4), ("Priya Raghavan", 11.9), ("Meera Kulkarni", 7.0)]

    run = MatchRun(
        id="demo-" + uuid.uuid4().hex[:6], model=settings.model, anonymise=False,
        requisitions=[], candidates=[],
    )
    for i, (title, client, _) in enumerate(roles):
        b = _copy.deepcopy(base.brief)
        b.role_title, b.client_name = title, client
        run.requisitions.append(Requisition(index=i, filename="{}.txt".format(title), jd_text="", brief=b))
    for i, (name, years) in enumerate(people):
        prof = _copy.deepcopy(base.profile)
        prof.full_name = name
        tl = _copy.deepcopy(base.timeline)
        tl.total_experience_months = int(years * 12)
        run.candidates.append(Candidate(index=i, filename="{}.pdf".format(name.split()[0].lower()),
                                        path=Path("."), profile=prof, timeline=tl, document=base.document))

    for ri, (_, _, scores) in enumerate(roles):
        for ci, pct in enumerate(scores):
            if pct is None:
                aff = Affinity(score=0.11, term_ratio=0.0)
                run.pairs.append(Pair(candidate_index=ci, requisition_index=ri, affinity=aff,
                                      selected=False, status="skipped",
                                      reason="nothing in this CV touches the role's requirements"))
                continue
            d = _copy.deepcopy(base)
            d.profile = run.candidates[ci].profile
            d.timeline = run.candidates[ci].timeline
            d.brief = run.requisitions[ri].brief
            keep = max(1, round(len(d.assessment.requirement_matches) * pct / 100))
            for n, m in enumerate(d.assessment.requirement_matches):
                if n >= keep and m.verdict in ("strong", "partial"):
                    m.verdict, m.evidence = "absent", None
            d.anonymise = False  # type: ignore[attr-defined]
            did = uuid.uuid4().hex[:12]
            STORE[did] = d
            run.pairs.append(Pair(candidate_index=ci, requisition_index=ri,
                                  affinity=Affinity(score=pct / 100, term_ratio=0.8),
                                  selected=True, status="done", dossier=d, dossier_id=did,
                                  reason="screened in"))

    run.phase = "done"
    run.finished_at = run.started_at + 47.0
    run.usage = llm_usage_stub()
    RUNS[run.id] = run
    return RedirectResponse("/match/{}".format(run.id), status_code=303)


def llm_usage_stub():
    from app.extract.llm import Usage
    return Usage(input_tokens=41200, output_tokens=9800, calls=12)


@app.get("/match/{run_id}", response_class=HTMLResponse)
def match_view(request: Request, run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "Match run not found. It may have been lost on server restart.")
    ctx = chrome("shortlists")
    ctx.update({"run": run, "model": run.model})
    return templates.TemplateResponse(request, "match.html", ctx)


@app.get("/match/{run_id}/status")
def match_status(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "Match run not found.")
    return JSONResponse(run_status(run))


@app.get("/batch/{batch_id}", response_class=HTMLResponse)
def batch_view(request: Request, batch_id: str):
    batch = BATCHES.get(batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found. It may have been lost on server restart.")
    ctx = chrome("shortlists")
    ctx.update({"batch": batch, "model": batch.model})
    return templates.TemplateResponse(request, "batch.html", ctx)


@app.get("/batch/{batch_id}/status")
def batch_status(batch_id: str):
    batch = BATCHES.get(batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found.")
    return JSONResponse(status_payload(batch))


@app.post("/demo")
def demo(request: Request):
    """Render the built-in fixture dossier. No API call, no key required.

    This exists so the UI, the template and the PDF path can be demonstrated
    and reviewed without spending a token.
    """
    from tests.fixtures import sample_dossier

    dossier = sample_dossier()
    dossier.anonymise = True  # type: ignore[attr-defined]
    dossier_id = "demo-" + uuid.uuid4().hex[:6]
    STORE[dossier_id] = dossier
    return RedirectResponse("/dossier/{}".format(dossier_id), status_code=303)


def _get(dossier_id: str) -> Dossier:
    dossier = STORE.get(dossier_id)
    if dossier is None:
        raise HTTPException(404, "Dossier not found. It may have been lost on server restart.")
    return dossier


@app.get("/dossier/{dossier_id}", response_class=HTMLResponse)
def review(request: Request, dossier_id: str, blind: Optional[int] = None):
    dossier = _get(dossier_id)
    anonymise = bool(getattr(dossier, "anonymise", settings.anonymise_by_default)) if blind is None else bool(blind)

    # Blind must be blind everywhere, the source pane included. Redaction
    # shifts character offsets, so the quote spans are recomputed against the
    # redacted text rather than reused from the original.
    view = dossier
    if anonymise:
        view = copy.deepcopy(dossier)
        name, email, phone = view.profile.full_name, view.profile.email, view.profile.phone
        view.document.text = redact_text(view.document.text, full_name=name, email=email, phone=phone) or ""
        # The filename is identity too: "cv_arjun_menon.pdf" defeats the whole
        # point of a blind profile.
        view.document.filename = "{}.{}".format(candidate_ref(dossier), view.document.source_format or "cv")
        redact_dossier(view)
        view.verification = verify_assessment(view.assessment, view.document.text, view.brief_text)

    # Split must-have coverage into strong vs partial for the two-tone meter.
    must = {r.text for r in view.brief.requirements if r.kind == "must_have"}
    strong_pct = partial_pct = 0.0
    if must:
        hits = [m for m in view.assessment.requirement_matches if m.requirement in must]
        strong_pct = sum(1 for m in hits if m.verdict == "strong") / len(must)
        partial_pct = sum(1 for m in hits if m.verdict == "partial") / len(must)

    anon_name = candidate_ref(dossier)
    display = anon_name if anonymise else (view.profile.full_name or "Unnamed candidate")

    ctx = chrome("candidates")
    ctx.update({
        "dossier_id": dossier_id,
        "dossier": view,
        "anonymise": anonymise,
        "ref": anon_name,
        "display_name": display,
        "computed_kinds": COMPUTED_KINDS,
        "must_have_texts": must,
        "skills_by_category": _skills_by_category(view),
        "source_html": render_source(view.document.text, view.verification),
        "strong_pct": strong_pct,
        "partial_pct": partial_pct,
        "is_demo": dossier_id.startswith("demo-"),
    })
    return templates.TemplateResponse(request, "review.html", ctx)


@app.get("/dossier/{dossier_id}/embed", response_class=HTMLResponse)
def embed(dossier_id: str, blind: int = 1):
    """The dossier itself, for the review page's iframe."""
    return HTMLResponse(render_html(_get(dossier_id), anonymise=bool(blind)))


@app.get("/dossier/{dossier_id}/pdf")
def pdf(dossier_id: str, blind: int = 1):
    dossier = _get(dossier_id)
    ref = candidate_ref(dossier)
    out = settings.output_dir / "{}_{}.pdf".format(ref, "blind" if blind else "named")
    render_pdf(dossier, out, anonymise=bool(blind))
    return FileResponse(
        out,
        media_type="application/pdf",
        filename="{} - {} - {}.pdf".format(settings.agency_name, dossier.brief.role_title, ref),
    )


def _error(request: Request, title: str, detail: str) -> HTMLResponse:
    ctx = chrome("dashboard")
    ctx.update({"title": title, "detail": detail})
    return templates.TemplateResponse(request, "error.html", ctx, status_code=400)


def _wants_html(request: Request) -> bool:
    """Browsers get the branded page; API clients keep getting JSON."""
    return "text/html" in request.headers.get("accept", "")


_FRIENDLY = {
    404: ("We could not find that page",
          "The link may be stale, or the item was lost when the server restarted. "
          "Dossiers and shortlists are held in memory for the current session only."),
    405: ("That action is not available here",
          "This page cannot be reached that way. Head back to the dashboard and try again."),
}


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if not _wants_html(request):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    title, detail = _FRIENDLY.get(exc.status_code, ("Something went wrong", str(exc.detail)))
    if exc.status_code == 404 and exc.detail and "not found" in str(exc.detail).lower():
        detail = str(exc.detail)
    ctx = chrome("dashboard")
    ctx.update({"title": title, "detail": detail})
    return templates.TemplateResponse(request, "error.html", ctx, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if not _wants_html(request):
        return JSONResponse({"detail": exc.errors()}, status_code=422)
    ctx = chrome("dashboard")
    ctx.update({
        "title": "That request did not look right",
        "detail": "One of the values in the address bar was not valid. Return to the dashboard and start again.",
    })
    return templates.TemplateResponse(request, "error.html", ctx, status_code=422)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Last resort. Never leak a traceback to the browser."""
    log.error("unhandled error on %s: %s", request.url.path, traceback.format_exc())
    if not _wants_html(request):
        return JSONResponse({"detail": "Internal server error"}, status_code=500)
    ctx = chrome("dashboard")
    ctx.update({
        "title": "Something went wrong on our side",
        "detail": "The error has been logged. Return to the dashboard and try again — "
                  "if it keeps happening, check the server console for details.",
    })
    return templates.TemplateResponse(request, "error.html", ctx, status_code=500)


@app.get("/health")
def health():
    return {
        "ok": True,
        "api_key_configured": _has_key(),
        "model": settings.model,
        "model_choices": [m[0] for m in available_models()],
        "dossiers_in_memory": len(STORE),
    }
