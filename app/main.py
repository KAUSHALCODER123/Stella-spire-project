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
import traceback
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import MODEL_CHOICES, settings
from app.extract.documents import SUPPORTED_SUFFIXES
from app.pipeline import Dossier, build_dossier
from app.render.dossier import COMPUTED_KINDS, candidate_ref, render_html, render_pdf
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
    cv_sample, jd_sample = _sample_paths()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "has_key": _has_key(),
            "model": settings.model,
            "model_choices": available_models(),
            "agency": settings.agency_name,
            "sample_jd": jd_sample.read_text(encoding="utf-8") if jd_sample.exists() else "",
            "sample_cv_name": cv_sample.name,
            "recent": [
                {"id": did, "ref": candidate_ref(d), "role": d.brief.role_title}
                for did, d in list(STORE.items())[-6:]
            ],
        },
    )


@app.post("/generate")
async def generate(
    request: Request,
    cv: Optional[UploadFile] = None,
    jd_text: str = Form(""),
    anonymise: Optional[str] = Form(None),
    use_sample: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
):
    cv_sample, jd_sample = _sample_paths()

    # --- resolve inputs ---------------------------------------------------
    if use_sample:
        cv_path = cv_sample
        jd_text = jd_sample.read_text(encoding="utf-8")
        if not cv_path.exists():
            raise HTTPException(500, "Sample CV missing from data/samples.")
    else:
        if cv is None or not cv.filename:
            return _error(request, "No CV uploaded.", "Choose a PDF, DOCX or TXT file, or run the built-in sample.")
        suffix = Path(cv.filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            return _error(
                request,
                "Unsupported file type: {}".format(suffix or "(none)"),
                "Supported formats are {}.".format(", ".join(sorted(SUPPORTED_SUFFIXES))),
            )
        if not jd_text.strip():
            return _error(request, "No job description.", "Paste the client brief so requirements can be matched against it.")

        payload = await cv.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            return _error(request, "File too large.", "The limit is 10 MB; this file is {:.1f} MB.".format(len(payload) / 1e6))

        cv_path = settings.upload_dir / "{}_{}".format(uuid.uuid4().hex[:8], Path(cv.filename).name)
        cv_path.write_bytes(payload)

    if not _has_key():
        return _error(
            request,
            "No API key configured.",
            "Copy .env.example to .env, add your OPENAI_API_KEY, then restart the server. "
            "Run `python -m scripts.check_setup` to verify it.",
        )

    # --- run --------------------------------------------------------------
    try:
        chosen = model if model in {m[0] for m in available_models()} else settings.model
        dossier = build_dossier(cv_path=cv_path, jd_text=jd_text, model=chosen)
    except Exception as exc:  # noqa: BLE001 - the UI is the error channel here
        log.error("pipeline failed: %s", traceback.format_exc())
        return _error(request, "Could not build the dossier.", str(exc))

    dossier_id = uuid.uuid4().hex[:12]
    dossier.anonymise = bool(anonymise)  # type: ignore[attr-defined]
    STORE[dossier_id] = dossier
    return RedirectResponse("/dossier/{}".format(dossier_id), status_code=303)


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

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "dossier_id": dossier_id,
            "dossier": view,
            "anonymise": anonymise,
            "ref": candidate_ref(dossier),
            "agency": settings.agency_name,
            "model": settings.model,
            "computed_kinds": COMPUTED_KINDS,
            "must_have_texts": must,
            "source_html": render_source(view.document.text, view.verification),
            "strong_pct": strong_pct,
            "partial_pct": partial_pct,
            "is_demo": dossier_id.startswith("demo-"),
        },
    )


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
    return templates.TemplateResponse(
        request,
        "error.html",
        {"title": title, "detail": detail, "agency": settings.agency_name},
        status_code=400,
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "api_key_configured": _has_key(),
        "model": settings.model,
        "model_choices": [m[0] for m in available_models()],
        "dossiers_in_memory": len(STORE),
    }
