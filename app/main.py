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
import os
import secrets
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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from starlette.middleware.sessions import SessionMiddleware

from app.accounts import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    COMPANIES,
    Company,
    authenticate,
    create_company,
    current_company,
    find_by_email,
    seed,
    sign_in,
    sign_out,
)
from app import db
from app.batch import BATCHES, create_batch, run_batch, status_payload
from app.intake import (
    CandidatePreferences,
    RoleConstraints,
    brief_from_constraints,
    check_constraints,
    role_text,
)
from app.matchrun import RUNS, create_run, execute as execute_run
from app.matchrun import status_payload as run_status
from app.extract.llm import LLMError
from app.config import MODEL_CHOICES, settings
from app.storage import get_storage, store_document
from app.uploads import display_filename, safe_filename, upload_suffix
from app.extract.documents import SUPPORTED_SUFFIXES, extract_text
from app.pipeline import Dossier, build_dossier
from app.pricing import price_usage
from app.render.dossier import COMPUTED_KINDS, _skills_by_category, candidate_ref, render_html, render_pdf
from app.render.source import render_source
from app.render.redact import redact_dossier, redact_text
from app.verify import verify_assessment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spiredossier")

# The interactive API docs enumerate every endpoint and parameter. Useful in
# development, an invitation in anything public, so they are opt-in.
app = FastAPI(
    title="SpireDossier",
    docs_url="/docs" if os.environ.get("EXPOSE_API_DOCS") else None,
    redoc_url="/redoc" if os.environ.get("EXPOSE_API_DOCS") else None,
    openapi_url="/openapi.json" if os.environ.get("EXPOSE_API_DOCS") else None,
)

# Signed-cookie sessions. The secret is regenerated on restart, which logs
# everyone out -- acceptable while accounts live in memory anyway, and it
# avoids shipping a hardcoded key that would look like a real one.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET") or secrets.token_hex(32),
    same_site="lax",
    https_only=False,
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Cost belongs in one place. Templates previously inlined a GPT-4o rate,
# which quietly invented a rupee figure for runs on the free tier.
templates.env.globals["price_usage"] = price_usage
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

STORE: Dict[str, Dossier] = {}

# Structured intake. Roles posted through the recruiter form need no JD
# parse at all, and applicants who filled the seeker form arrive with hard
# constraints already declared.
ROLE_LIBRARY: Dict[str, RoleConstraints] = {}
APPLICANTS: Dict[str, dict] = {}

# A populated board on first run. Five empty states hide the behaviour
# worth looking at; set SEED_DEMO_DATA=0 to start clean.
def _restore() -> dict:
    """Bring persisted state back before anything is seeded.

    Order matters: seeding is idempotent only if it can see what already
    exists, otherwise every restart would duplicate the demo board.
    """
    from app.accounts import load_from_db

    restored = {"companies": 0, "roles": 0, "applicants": 0, "dossiers": 0}
    if db.get_db() is None:
        return restored

    restored["companies"] = load_from_db()
    for role_id, role in db.load_roles():
        ROLE_LIBRARY[role_id] = role
        restored["roles"] += 1
    for applicant_id, record in db.load_applicants():
        APPLICANTS[applicant_id] = record
        restored["applicants"] += 1
    for dossier_id, dossier in db.load_dossiers():
        STORE[dossier_id] = dossier
        restored["dossiers"] += 1

    log.info("restored %s from the database", restored)
    return restored


RESTORED = {"companies": 0, "roles": 0, "applicants": 0, "dossiers": 0}


RESTORED = _restore()
seed()

if os.environ.get("SEED_DEMO_DATA", "1") != "0":
    from app.seed_data import seed_board

    _seeded = seed_board(ROLE_LIBRARY, APPLICANTS)
    if not _seeded.get("skipped"):
        # Persist the seed so a restart restores it rather than
        # regenerating it with fresh ids.
        for _rid, _role in ROLE_LIBRARY.items():
            db.save_role(_rid, _role)
        for _aid, _rec in APPLICANTS.items():
            db.save_applicant(_aid, _rec)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

def run_in_background(target, *args) -> None:
    """Dispatch long work off the request.

    A named seam rather than a bare threading.Thread call: tests replace this
    to run inline. Patching threading.Thread itself would be global, and would
    break the test client's own worker threads.
    """
    threading.Thread(target=target, args=args, daemon=True).start()


def _sample_paths() -> tuple[Path, Path]:
    """The pair the workspace opens with.

    These must stay in the vertical the product is being shown in: the JD here
    is prefilled into the brief box, so it is the first thing anyone reads on
    the main screen.
    """
    return (
        settings.sample_dir / "cv_meera_ramanathan.txt",
        settings.sample_dir / "jd_cfo_fintech_gcc.txt",
    )


def _has_key() -> bool:
    return bool(settings.openai_api_key)


def require_login(request: Request):
    """The signed-in company, or a redirect to the sign-in page."""
    company = current_company(request)
    if company is None:
        return None, RedirectResponse("/login?next={}".format(request.url.path), status_code=303)
    return company, None


def require_admin(request: Request):
    """Admin-only surfaces. One check, because there is one login path."""
    company, redirect = require_login(request)
    if redirect is not None:
        return None, redirect
    if not company.is_admin:
        return None, _error(
            request,
            "That area is for the recruitment team",
            "Your account can post roles and review candidates matched to them. "
            "The parsing workspace belongs to {}.".format(settings.agency_name),
        )
    return company, None


def dossier_owners(dossier_id: str) -> set:
    """Which company accounts a dossier legitimately belongs to.

    A dossier is reachable by the employer whose opening it was assessed
    against. Traced pair -> requisition -> source role -> company, so the
    answer follows the data rather than being stored twice and drifting.
    """
    owners = set()
    for run in RUNS.values():
        for pair in run.pairs:
            if pair.dossier_id != dossier_id:
                continue
            role_id = run.requisitions[pair.requisition_index].source_role_id
            role = ROLE_LIBRARY.get(role_id) if role_id else None
            if role is not None and role.company_id:
                owners.add(role.company_id)
    return owners


def authorise_dossier(request: Request, dossier_id: str):
    """Who may read one candidate's dossier.

    Before this existed the dossier routes had no guard at all: anyone holding
    a URL could read a candidate's name, email, phone, salary expectation and
    the full text of their CV through ?blind=0, with no session. That is the
    most sensitive data in the product.

    Returns (company, forced_blind, redirect).
    """
    company, redirect = require_login(request)
    if redirect is not None:
        return None, True, redirect
    if company.is_admin:
        return company, False, None
    if company.id in dossier_owners(dossier_id):
        # An employer may read a candidate matched to their own opening, but
        # cannot lift the blind: identity is released deliberately by the
        # agency, not by editing a query string.
        return company, True, None
    return None, True, _error(
        request,
        "That candidate is not on one of your shortlists",
        "You can open the candidates matched to roles your company posted. "
        "If you expected to see this one, ask {} to run matching against your "
        "openings.".format(settings.agency_name),
    )


def chrome(nav: str, request: Request = None) -> dict:
    """Context every page needs for the sidebar shell."""
    company = current_company(request) if request is not None else None
    return {
        "company": company,
        "is_admin": bool(company and company.is_admin),
        "agency": settings.agency_name,
        "model": settings.model,
        "has_key": _has_key(),
        "nav": nav,
        "nav_counts": {
            "candidates": len(STORE),
            "roles": len(ROLE_LIBRARY),
            "shortlists": len(BATCHES) + len(RUNS),
        },
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
def landing(request: Request):
    """Public front door. Two doors out: apply, or hire."""
    company = current_company(request)
    open_roles = [rc for rc in ROLE_LIBRARY.values()]
    return templates.TemplateResponse(request, "landing.html", {
        "agency": settings.agency_name,
        "company": company,
        "is_admin": bool(company and company.is_admin),
        "open_roles": open_roles[:6],
        "role_count": len(open_roles),
        "company_count": len(COMPANIES) - 1,
        "applicant_count": len(APPLICANTS),
    })


# --------------------------------------------------------------------------
# One sign-in for everyone. The account decides what is shown.
# --------------------------------------------------------------------------


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/dashboard"):
    if current_company(request):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(request, "login.html", {
        "agency": settings.agency_name, "next": next, "error": None,
        "companies": [c for c in COMPANIES.values() if not c.is_admin][:5],
    })


@app.post("/login", response_class=HTMLResponse)
def login(request: Request,
          email: str = Form(""),
          password: str = Form(""),
          next: str = Form("/dashboard")):
    company = authenticate(email, password)
    if company is None:
        # One message for both causes: saying "no such account" tells an
        # attacker which addresses are registered.
        return templates.TemplateResponse(request, "login.html", {
            "agency": settings.agency_name, "next": next,
            "error": "Those details did not match an account.",
            "companies": [c for c in COMPANIES.values() if not c.is_admin][:5],
        }, status_code=401)
    sign_in(request, company)
    target = next if next.startswith("/") else "/dashboard"
    return RedirectResponse(target, status_code=303)


@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    if current_company(request):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(request, "register.html", {
        "agency": settings.agency_name, "error": None, "values": {},
    })


@app.post("/register", response_class=HTMLResponse)
def register(request: Request,
             name: str = Form(""),
             email: str = Form(""),
             password: str = Form(""),
             confirm: str = Form(""),
             industry: str = Form(""),
             location: str = Form(""),
             website: str = Form("")):
    values = {"name": name, "email": email, "industry": industry,
              "location": location, "website": website}

    def fail(message):
        return templates.TemplateResponse(request, "register.html", {
            "agency": settings.agency_name, "error": message, "values": values,
        }, status_code=400)

    if not name.strip():
        return fail("Your company needs a name.")
    if "@" not in email or "." not in email.split("@")[-1]:
        return fail("That does not look like a valid email address.")
    if len(password) < 8:
        return fail("Choose a password of at least 8 characters.")
    if password != confirm:
        return fail("The two passwords did not match.")
    if find_by_email(email):
        return fail("An account already exists for that email address.")

    company = create_company(name=name, email=email, password=password,
                             industry=industry, location=location, website=website)
    sign_in(request, company)
    return RedirectResponse("/dashboard", status_code=303)


@app.post("/logout")
def logout(request: Request):
    sign_out(request)
    return RedirectResponse("/", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    """Role-aware home. Admin gets the parsing workspace; employers get theirs."""
    company, redirect = require_login(request)
    if redirect is not None:
        return redirect
    if company.is_admin:
        return workspace(request)
    return employer_home(request, company)


def employer_home(request: Request, company: Company):
    my_roles = [(rid, rc) for rid, rc in ROLE_LIBRARY.items() if rc.company_id == company.id]
    my_ids = {rid for rid, _ in my_roles}

    # Candidates matched to this employer's own openings, best first.
    matches = []
    for run in RUNS.values():
        for pair in run.pairs:
            if not pair.dossier:
                continue
            req = run.requisitions[pair.requisition_index]
            if req.source_role_id not in my_ids:
                continue
            cand = run.candidates[pair.candidate_index]
            matches.append({
                "dossier_id": pair.dossier_id,
                "role_title": req.title,
                "name": cand.name if not run.anonymise else pair.dossier_id[:8].upper(),
                "years": cand.years,
                "percent": pair.dossier.suitability["percent"],
                "tone": pair.dossier.suitability["tone"],
                "band": pair.dossier.suitability["band"],
            })
    matches.sort(key=lambda m: -m["percent"])

    ctx = chrome("dashboard", request)
    ctx.update({
        "my_roles": my_roles,
        "matches": matches,
        "applicant_count": len(APPLICANTS),
    })
    return templates.TemplateResponse(request, "employer.html", ctx)


@app.get("/workspace", response_class=HTMLResponse)
def workspace_route(request: Request):
    company, redirect = require_admin(request)
    if redirect is not None:
        return redirect
    return workspace(request)


def workspace(request: Request):
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

    ctx = chrome("dashboard", request)
    ctx.update({
        "model_choices": available_models(),
        "sample_jd": jd_sample.read_text(encoding="utf-8") if jd_sample.exists() else "",
        "recent": recent,
        "recent_batches": batches,
        "stats": {"dossiers": len(STORE), "shortlists": len(BATCHES), "best_match": best},
    })
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/candidates", response_class=HTMLResponse)
def candidates(request: Request, role_id: Optional[str] = None):
    company, redirect = require_admin(request)
    if redirect is not None:
        return redirect
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

    # Everyone who has applied through the seeker form, not yet scored
    # against anything. Filtering by a role previews the free constraint
    # gate so the admin only spends a model call on candidates worth it.
    selected_role = ROLE_LIBRARY.get(role_id) if role_id else None
    applicants = []
    for aid, a in list(APPLICANTS.items())[::-1]:
        prefs = a["prefs"]
        gate = check_constraints(prefs, selected_role) if selected_role else None
        applicants.append({
            "id": aid,
            "name": prefs.full_name or a["filename"],
            "initials": _initials(prefs.full_name, "CV"),
            "years": prefs.years_experience,
            "location": prefs.current_location,
            "notice": prefs.notice_period_days,
            "expected_ctc": prefs.expected_ctc_lpa,
            "gate": gate,
        })
    if selected_role is not None:
        applicants.sort(key=lambda row: bool(row["gate"] and row["gate"].blocked))

    ctx = chrome("candidates", request)
    ctx.update({
        "heading": "Candidates", "rows": rows,
        "empty_title": "No resumes analysed yet",
        "empty_detail": "Upload a resume and a job description to build your first match report.",
        "applicants": applicants,
        "roles": list(ROLE_LIBRARY.items()),
        "selected_role_id": role_id or "",
        "selected_role_title": selected_role.role_title if selected_role else "",
        "model_choices": available_models(),
    })
    return templates.TemplateResponse(request, "candidates.html", ctx)


@app.post("/candidates/match")
def match_selected_applicants(
    request: Request,
    role_id: str = Form(...),
    applicant_ids: List[str] = Form(default=[]),
    model: Optional[str] = Form(None),
    anonymise: Optional[str] = Form(None),
):
    """Match a hand-picked set of applicants against one posted role.

    The bulk /roles/match runs everyone against everything; this is the
    other end of the same pipeline, for when the admin already knows which
    CVs and which role belong together.
    """
    _, redirect = require_admin(request)
    if redirect is not None:
        return redirect
    rc = ROLE_LIBRARY.get(role_id)
    if rc is None:
        return _error(request, "That role no longer exists.",
                      "Pick a role from the list and try again.")
    chosen_ids = [aid for aid in applicant_ids if aid in APPLICANTS]
    if not chosen_ids:
        return _error(request, "No candidates selected.",
                      "Select at least one applicant to match against a role.")
    if not _has_key():
        return _error(request, "No API key configured.",
                      "Add your OPENAI_API_KEY to .env and restart the server.")

    chosen_model = model if model in {m[0] for m in available_models()} else settings.model
    run = create_run(
        jds=[(rc.role_title, role_text(rc))],
        cvs=[(APPLICANTS[aid]["filename"], APPLICANTS[aid]["path"]) for aid in chosen_ids],
        model=chosen_model,
        anonymise=bool(anonymise),
        extraction_model=settings.extraction_model or chosen_model,
    )
    run.requisitions[0].constraints = rc
    run.requisitions[0].brief = brief_from_constraints(rc)
    run.requisitions[0].source_role_id = role_id
    for cand, aid in zip(run.candidates, chosen_ids):
        cand.prefs = APPLICANTS[aid]["prefs"]

    run_in_background(execute_run, run.id, STORE)
    return RedirectResponse("/match/{}".format(run.id), status_code=303)


@app.get("/shortlists", response_class=HTMLResponse)
def shortlists(request: Request):
    company, redirect = require_admin(request)
    if redirect is not None:
        return redirect
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
    ctx = chrome("shortlists", request)
    ctx.update({
        "heading": "Shortlists", "rows": rows,
        # The zero-cost many-to-many run belongs here, where someone
        # looking for match results would actually go for it.
        "show_match_demo": True,
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
    _, redirect = require_admin(request)
    if redirect is not None:
        return redirect
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
        store_document(payload=payload, filename=upload.filename,
                        kind="resume", local_path=path)
        # Display name stays the original; only the stored path is sanitised.
        saved.append((Path(upload.filename.replace("\\", "/")).name or "resume", path))

    chosen = model if model in {m[0] for m in available_models()} else settings.model

    # --- several CVs: run as a batch, report progress ---------------------
    if len(saved) > 1:
        batch = create_batch(files=saved, jd_text=jd_text, model=chosen, anonymise=bool(anonymise))
        run_in_background(run_batch, batch.id, STORE)
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
    db.save_dossier(dossier_id, dossier)
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
    _, redirect = require_admin(request)
    if redirect is not None:
        return redirect
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
        store_document(payload=payload, filename=upload.filename,
                        kind="jd", local_path=path)
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
        store_document(payload=payload, filename=upload.filename,
                        kind="resume", local_path=path)
        cvs.append((display_filename(upload.filename), path))

    chosen = model if model in {m[0] for m in available_models()} else settings.model
    run = create_run(
        jds=jds, cvs=cvs, model=chosen, anonymise=bool(anonymise),
        top_roles=max(1, min(int(top_roles or 3), 10)),
        assess_all=bool(assess_all),
        extraction_model=settings.extraction_model or chosen,
    )
    run_in_background(execute_run, run.id, STORE)
    return RedirectResponse("/match/{}".format(run.id), status_code=303)


def _split(raw):
    """Comma or newline separated free text into a clean list."""
    out = []
    for chunk in (raw or "").replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(chunk)
    return out


def _num(raw):
    try:
        text = str(raw or "").strip()
        return float(text) if text else None
    except ValueError:
        return None


def _int(raw):
    value = _num(raw)
    return int(value) if value is not None else None


# --------------------------------------------------------------------------
# Recruiter: post a role
# --------------------------------------------------------------------------


@app.get("/post-role", response_class=HTMLResponse)
def post_role_form(request: Request):
    company, redirect = require_login(request)
    if redirect is not None:
        return redirect
    ctx = chrome("roles", request)
    ctx.update({"roles": list(ROLE_LIBRARY.items())})
    return templates.TemplateResponse(request, "post_role.html", ctx)


@app.post("/post-role")
async def post_role(
    request: Request,
    role_title: str = Form(""),
    client_name: str = Form(""),
    location: str = Form(""),
    work_mode: str = Form("any"),
    min_years: str = Form(""),
    max_years: str = Form(""),
    ctc_min_lpa: str = Form(""),
    ctc_max_lpa: str = Form(""),
    max_notice_days: str = Form(""),
    must_have_skills: str = Form(""),
    nice_to_have_skills: str = Form(""),
    domain: str = Form(""),
    notes: str = Form(""),
):
    company, redirect = require_login(request)
    if redirect is not None:
        return redirect
    if not role_title.strip():
        return _error(request, "The role needs a title.",
                      "Give the opening a title so it can be identified in the matrix.")

    rc = RoleConstraints(
        role_title=role_title.strip(),
        client_name=client_name.strip() or None,
        location=location.strip() or None,
        work_mode=work_mode,
        min_years=_num(min_years),
        max_years=_num(max_years),
        ctc_min_lpa=_num(ctc_min_lpa),
        ctc_max_lpa=_num(ctc_max_lpa),
        max_notice_days=_int(max_notice_days),
        must_have_skills=_split(must_have_skills),
        nice_to_have_skills=_split(nice_to_have_skills),
        domain=domain.strip() or None,
        notes=notes.strip() or None,
    )
    rc.company_id = company.id
    role_id = uuid.uuid4().hex[:10]
    ROLE_LIBRARY[role_id] = rc
    db.save_role(role_id, rc)
    return RedirectResponse("/roles", status_code=303)


@app.get("/roles", response_class=HTMLResponse)
def roles_page(request: Request):
    company, redirect = require_login(request)
    if redirect is not None:
        return redirect

    # An employer sees their own openings and nothing else. Only the agency
    # account sees the whole board.
    if company.is_admin:
        roles = list(ROLE_LIBRARY.items())
    else:
        roles = [(rid, rc) for rid, rc in ROLE_LIBRARY.items() if rc.company_id == company.id]

    ctx = chrome("roles", request)
    ctx.update({
        "roles": roles,
        "applicants": list(APPLICANTS.items()) if company.is_admin else [],
        "model_choices": available_models() if company.is_admin else [],
    })
    return templates.TemplateResponse(request, "roles.html", ctx)


@app.post("/roles/{role_id}/delete")
def delete_role(request: Request, role_id: str):
    company, redirect = require_login(request)
    if redirect is not None:
        return redirect
    role = ROLE_LIBRARY.get(role_id)
    if role is not None and (company.is_admin or role.company_id == company.id):
        ROLE_LIBRARY.pop(role_id, None)
        db.delete_role(role_id)
    return RedirectResponse("/roles", status_code=303)


# --------------------------------------------------------------------------
# Job seeker: apply
# --------------------------------------------------------------------------


@app.get("/apply", response_class=HTMLResponse)
def apply_form(request: Request):
    ctx = chrome("apply", request)
    ctx.update({"roles": list(ROLE_LIBRARY.values())})
    return templates.TemplateResponse(request, "apply.html", ctx)


@app.post("/apply")
async def apply(
    request: Request,
    cv: UploadFile = File(...),
    full_name: str = Form(""),
    email: str = Form(""),
    phone: str = Form(""),
    target_roles: str = Form(""),
    current_location: str = Form(""),
    preferred_locations: str = Form(""),
    open_to_relocate: Optional[str] = Form(None),
    work_mode: str = Form("any"),
    notice_period_days: str = Form(""),
    current_ctc_lpa: str = Form(""),
    expected_ctc_lpa: str = Form(""),
    min_acceptable_ctc_lpa: str = Form(""),
    years_experience: str = Form(""),
    notes: str = Form(""),
):
    if cv is None or not cv.filename:
        return _error(request, "No CV attached.",
                      "Attach your CV so recruiters can review your experience.")
    if upload_suffix(cv.filename) not in SUPPORTED_SUFFIXES:
        return _error(request, "That file type is not supported.",
                      "Please upload one of: {}.".format(", ".join(sorted(SUPPORTED_SUFFIXES))))
    payload = await cv.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        return _error(request, "That file is too large.",
                      "The limit is 10 MB; yours is {:.1f} MB.".format(len(payload) / 1e6))

    path = settings.upload_dir / "{}_{}".format(uuid.uuid4().hex[:8], safe_filename(cv.filename))
    stored = store_document(payload=payload, filename=cv.filename,
                             kind="resume", local_path=path)

    prefs = CandidatePreferences(
        full_name=full_name.strip() or None,
        email=email.strip() or None,
        phone=phone.strip() or None,
        target_roles=_split(target_roles),
        current_location=current_location.strip() or None,
        preferred_locations=_split(preferred_locations),
        open_to_relocate=bool(open_to_relocate),
        work_mode=work_mode,
        notice_period_days=_int(notice_period_days),
        current_ctc_lpa=_num(current_ctc_lpa),
        expected_ctc_lpa=_num(expected_ctc_lpa),
        min_acceptable_ctc_lpa=_num(min_acceptable_ctc_lpa),
        years_experience=_num(years_experience),
        notes=notes.strip() or None,
    )
    applicant_id = uuid.uuid4().hex[:10]
    APPLICANTS[applicant_id] = {
        "prefs": prefs,
        "path": path,
        "filename": display_filename(cv.filename),
        "stored": stored,
    }
    db.save_applicant(applicant_id, APPLICANTS[applicant_id])

    ctx = chrome("apply", request)
    ctx.update({"applicant": prefs, "roles": list(ROLE_LIBRARY.values())})
    return templates.TemplateResponse(request, "applied.html", ctx)


@app.post("/roles/match")
def match_library(request: Request,
                  model: Optional[str] = Form(None),
                  anonymise: Optional[str] = Form(None)):
    """Match every applicant against every open role.

    Roles posted through the form already carry their brief, so this run skips
    the job-description parse entirely: M calls saved before it starts, and the
    constraint gate then removes pairs before the affinity screen runs.
    """
    _, redirect = require_admin(request)
    if redirect is not None:
        return redirect
    if not ROLE_LIBRARY:
        return _error(request, "No open roles yet.",
                      "Post at least one role before running a match.")
    if not APPLICANTS:
        return _error(request, "No applicants yet.",
                      "Share the application link so candidates can submit a CV.")
    if not _has_key():
        return _error(request, "No API key configured.",
                      "Add your OPENAI_API_KEY to .env and restart the server.")

    chosen = model if model in {m[0] for m in available_models()} else settings.model
    run = create_run(
        jds=[(rc.role_title, role_text(rc)) for rc in ROLE_LIBRARY.values()],
        cvs=[(a["filename"], a["path"]) for a in APPLICANTS.values()],
        model=chosen,
        anonymise=bool(anonymise),
        extraction_model=settings.extraction_model or chosen,
    )
    for req, (role_id, rc) in zip(run.requisitions, ROLE_LIBRARY.items()):
        req.constraints = rc
        req.brief = brief_from_constraints(rc)
        req.source_role_id = role_id
    for cand, applicant in zip(run.candidates, APPLICANTS.values()):
        cand.prefs = applicant["prefs"]

    run_in_background(execute_run, run.id, STORE)
    return RedirectResponse("/match/{}".format(run.id), status_code=303)


@app.post("/match/demo")
def match_demo(request: Request):
    """A finished many-to-many run built from stored fixtures. No model call.

    Exists for the same reason /demo does: the matrix, the shortlists and the
    role tags need to be reviewable without spending anything.
    """
    _, redirect = require_admin(request)
    if redirect is not None:
        return redirect

    import copy as _copy

    from app.intake import check_constraints
    from app.matching import Affinity
    from app.matchrun import Candidate, MatchRun, Pair, Requisition
    from tests.fixtures import sample_dossier

    base = sample_dossier()

    # Finance leadership, matching the vertical the rest of the product is in.
    # The constraints are real: the gate below is the same code the live path
    # runs, so what this demo shows being ruled out for free actually was.
    roles = [
        ("Chief Financial Officer", "Meridian Global Capability Centre",
         RoleConstraints(role_title="Chief Financial Officer", client_name="Meridian GCC",
                         location="Bengaluru", work_mode="hybrid", min_years=15,
                         ctc_min_lpa=95, ctc_max_lpa=130, max_notice_days=90),
         [88, 61, None, None]),
        ("Head of Financial Planning & Analysis", "Alderline NBFC",
         RoleConstraints(role_title="Head of FP&A", client_name="Alderline NBFC",
                         location="Mumbai", work_mode="onsite", min_years=8, max_years=15,
                         ctc_min_lpa=55, ctc_max_lpa=80, max_notice_days=60),
         [70, 79, 44, None]),
        ("Financial Controller", "Alderline NBFC",
         RoleConstraints(role_title="Financial Controller", client_name="Alderline NBFC",
                         location="Mumbai", work_mode="hybrid", min_years=7,
                         ctc_min_lpa=40, ctc_max_lpa=62, max_notice_days=90),
         [None, 58, 66, None]),
    ]

    # One of these is priced out of every band and one will not leave Dubai,
    # so the free gate has something real to do rather than reporting zero.
    people = [
        ("Meera Ramanathan", 13.3, CandidatePreferences(
            full_name="Meera Ramanathan", current_location="Bengaluru", work_mode="hybrid",
            notice_period_days=60, expected_ctc_lpa=105, min_acceptable_ctc_lpa=95,
            years_experience=13.3)),
        ("Anita Krishnamurthy", 11.2, CandidatePreferences(
            full_name="Anita Krishnamurthy", current_location="Mumbai", work_mode="onsite",
            open_to_relocate=True, notice_period_days=60, expected_ctc_lpa=70,
            min_acceptable_ctc_lpa=58, years_experience=11.2)),
        ("Sneha Iyer", 8.1, CandidatePreferences(
            full_name="Sneha Iyer", current_location="Mumbai", work_mode="hybrid",
            open_to_relocate=True, notice_period_days=30, expected_ctc_lpa=52,
            min_acceptable_ctc_lpa=44, years_experience=8.1)),
        ("Daniel Okonkwo", 9.6, CandidatePreferences(
            full_name="Daniel Okonkwo", current_location="Dubai", preferred_locations=["Dubai"],
            open_to_relocate=False, work_mode="onsite", notice_period_days=30,
            expected_ctc_lpa=140, min_acceptable_ctc_lpa=135, years_experience=9.6)),
    ]

    run = MatchRun(
        id="demo-" + uuid.uuid4().hex[:6], model=settings.model, anonymise=False,
        requisitions=[], candidates=[],
    )
    # Tie each demo requisition back to the seeded role it stands for. Without
    # this the employer view shows no matched candidates, which is the one beat
    # of the demo that proves a client sees only their own shortlist.
    by_title = {rc.role_title: rid for rid, rc in ROLE_LIBRARY.items()}
    for i, (title, client, constraints, _) in enumerate(roles):
        b = _copy.deepcopy(base.brief)
        b.role_title, b.client_name = title, client
        run.requisitions.append(Requisition(
            index=i, filename="{}.txt".format(title), jd_text="", brief=b,
            constraints=constraints, source_role_id=by_title.get(title)))
    for i, (name, years, prefs) in enumerate(people):
        prof = _copy.deepcopy(base.profile)
        prof.full_name = name
        tl = _copy.deepcopy(base.timeline)
        tl.total_experience_months = int(years * 12)
        run.candidates.append(Candidate(index=i, filename="{}.pdf".format(name.split()[0].lower()),
                                        path=Path("."), profile=prof, timeline=tl,
                                        document=base.document, prefs=prefs))

    for ri, (_, _, constraints, scores) in enumerate(roles):
        for ci, pct in enumerate(scores):
            # The free gate first, exactly as the live path orders it: a pair
            # blocked on stated facts never reaches a model at all.
            check = check_constraints(run.candidates[ci].prefs, constraints)
            if check.blocked:
                run.pairs.append(Pair(candidate_index=ci, requisition_index=ri,
                                      affinity=Affinity(score=0.0, term_ratio=0.0),
                                      selected=False, status="blocked", check=check,
                                      reason=check.blocks[0]))
                continue
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
    company, redirect = require_admin(request)
    if redirect is not None:
        return redirect
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "Match run not found. It may have been lost on server restart.")
    ctx = chrome("shortlists", request)
    ctx.update({"run": run, "model": run.model})
    return templates.TemplateResponse(request, "match.html", ctx)


@app.get("/match/{run_id}/status")
def match_status(request: Request, run_id: str):
    _, redirect = require_admin(request)
    if redirect is not None:
        return redirect
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "Match run not found.")
    return JSONResponse(run_status(run))


@app.get("/batch/{batch_id}", response_class=HTMLResponse)
def batch_view(request: Request, batch_id: str):
    company, redirect = require_admin(request)
    if redirect is not None:
        return redirect
    batch = BATCHES.get(batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found. It may have been lost on server restart.")
    ctx = chrome("shortlists", request)
    ctx.update({"batch": batch, "model": batch.model})
    return templates.TemplateResponse(request, "batch.html", ctx)


@app.get("/batch/{batch_id}/status")
def batch_status(request: Request, batch_id: str):
    _, redirect = require_admin(request)
    if redirect is not None:
        return redirect
    batch = BATCHES.get(batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found.")
    return JSONResponse(status_payload(batch))


@app.post("/demo")
def demo(request: Request):
    """Render the built-in fixture dossier. No API call, no key required.

    Behind sign-in because it writes to the shared store: left open it is an
    unauthenticated way to grow server memory without limit.

    This exists so the UI, the template and the PDF path can be demonstrated
    and reviewed without spending a token.
    """
    _, redirect = require_admin(request)
    if redirect is not None:
        return redirect

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
    _, forced_blind, redirect = authorise_dossier(request, dossier_id)
    if redirect is not None:
        return redirect
    dossier = _get(dossier_id)
    if forced_blind:
        blind = 1
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

    ctx = chrome("candidates", request)
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
def embed(request: Request, dossier_id: str, blind: int = 1):
    """The dossier itself, for the review page's iframe."""
    _, forced_blind, redirect = authorise_dossier(request, dossier_id)
    if redirect is not None:
        return redirect
    if forced_blind:
        blind = 1
    return HTMLResponse(render_html(_get(dossier_id), anonymise=bool(blind)))


@app.get("/dossier/{dossier_id}/pdf")
def pdf(request: Request, dossier_id: str, blind: int = 1):
    _, forced_blind, redirect = authorise_dossier(request, dossier_id)
    if redirect is not None:
        return redirect
    if forced_blind:
        blind = 1
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
    ctx = chrome("dashboard", request)
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
    ctx = chrome("dashboard", request)
    ctx.update({"title": title, "detail": detail})
    return templates.TemplateResponse(request, "error.html", ctx, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if not _wants_html(request):
        return JSONResponse({"detail": exc.errors()}, status_code=422)
    ctx = chrome("dashboard", request)
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
    ctx = chrome("dashboard", request)
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
        "storage": "supabase" if get_storage() else "local disk only",
        "database": db.status(),
        "restored_at_startup": RESTORED,
    }
