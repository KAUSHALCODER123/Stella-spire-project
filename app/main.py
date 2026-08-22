"""FastAPI app: upload a CV and a brief, get a dossier.

One process, one language, no build step. Jinja2 templates server-side, the
rendered dossier shown in an iframe so its print stylesheet cannot collide with
the app chrome, and a PDF endpoint that streams the same HTML through Chromium.

Dossiers are held in a process-local dict. That is deliberate for a demo: no
database to provision, and nothing persists after shutdown. Swapping in SQLite
is a contained change to `STORE`.
"""

from __future__ import annotations

import logging
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.extract.documents import SUPPORTED_SUFFIXES
from app.pipeline import Dossier, build_dossier
from app.render.dossier import candidate_ref, render_html, render_pdf

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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    cv_sample, jd_sample = _sample_paths()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "has_key": _has_key(),
            "model": settings.model,
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
        dossier = build_dossier(cv_path=cv_path, jd_text=jd_text)
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

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "dossier_id": dossier_id,
            "dossier": dossier,
            "anonymise": anonymise,
            "ref": candidate_ref(dossier),
            "agency": settings.agency_name,
            "model": settings.model,
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
        "dossiers_in_memory": len(STORE),
    }
