"""Render a Dossier to HTML, and HTML to PDF.

PDF generation goes through Playwright's bundled Chromium rather than
WeasyPrint. WeasyPrint needs GTK system libraries that are a genuine ordeal to
install on Windows; Chromium ships with `playwright install chromium` and
renders the same CSS the browser preview shows, so what you see in the review
UI is what lands in the PDF.

One consequence: Chromium does not implement CSS paged-media margin boxes
(`@bottom-left`), so running headers and footers are passed to
`page.pdf(footer_template=...)` instead of being styled in the stylesheet.
"""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import settings
from app.pipeline import Dossier
from app.render.redact import redact_dossier

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

# Flag kinds produced by app/analysis.py rather than by the model. Tagged in
# the output so the reader knows which numbers are arithmetic.
COMPUTED_KINDS = {"employment_gap", "short_tenure", "title_inflation", "logistics"}

_CATEGORY_LABELS = OrderedDict(
    [
        ("ml", "Machine learning"),
        ("data", "Data"),
        ("language", "Languages"),
        ("framework", "Frameworks"),
        ("cloud", "Cloud"),
        ("infra", "Infrastructure"),
        ("domain", "Domain"),
        ("leadership", "Leadership"),
        ("tool", "Tools"),
        ("other", "Other"),
    ]
)

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def candidate_ref(dossier: Dossier) -> str:
    """A stable, non-identifying reference.

    Derived from the name so the same candidate keeps the same ref across
    regenerations, hashed so the ref itself leaks nothing.
    """
    seed = (dossier.profile.full_name or "") + (dossier.profile.email or "") or dossier.document.text[:200]
    return "SD-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:6].upper()


def _timeline_items(dossier: Dossier) -> List[Dict[str, Any]]:
    """Roles and gaps interleaved, most recent first."""
    items: List[Dict[str, Any]] = []
    intervals = sorted(dossier.timeline.intervals, key=lambda i: i.start, reverse=True)
    gaps = {g[0]: g for g in dossier.timeline.gaps}

    for iv in intervals:
        months = (iv.end.year - iv.start.year) * 12 + (iv.end.month - iv.start.month)
        is_current = iv.position.end.strip().lower() in {"present", "current", "now", "ongoing"}
        items.append(
            {
                "gap": False,
                "position": iv.position,
                "label": "{} — {}".format(iv.start.strftime("%b %Y"), "Present" if is_current else iv.end.strftime("%b %Y")),
                "months": months,
            }
        )
        # A gap starting where this role ended belongs directly beneath it.
        if iv.end in gaps:
            start, end, gap_months = gaps[iv.end]
            items.append(
                {
                    "gap": True,
                    "months": gap_months,
                    "from": start.strftime("%b %Y"),
                    "to": end.strftime("%b %Y"),
                }
            )

    # Unreadable dates still deserve a line; dropping them would hide history.
    for pos in dossier.timeline.unparseable:
        items.append({"gap": False, "position": pos, "label": "Dates unclear", "months": None})

    return items


def _skills_by_category(dossier: Dossier) -> "OrderedDict[str, list]":
    grouped: Dict[str, list] = {}
    for skill in dossier.profile.skills:
        grouped.setdefault(skill.category, []).append(skill)

    ordered: "OrderedDict[str, list]" = OrderedDict()
    for key, label in _CATEGORY_LABELS.items():
        if grouped.get(key):
            # Evidenced skills first -- they are the ones worth reading.
            ordered[label] = sorted(grouped[key], key=lambda s: (s.evidence is None, s.name.lower()))
    return ordered


def build_context(dossier: Dossier, *, anonymise: Optional[bool] = None) -> Dict[str, Any]:
    if anonymise is None:
        anonymise = settings.anonymise_by_default

    if anonymise:
        # Redaction runs on a copy: the caller keeps the identified dossier for
        # the recruiter's own view, and only the client-facing render is blind.
        ref_before = candidate_ref(dossier)
        dossier = copy.deepcopy(dossier)
        redact_dossier(dossier)
        dossier._ref_override = ref_before  # type: ignore[attr-defined]

    ref = getattr(dossier, "_ref_override", None) or candidate_ref(dossier)
    counts = dossier.match_counts
    coverage = dossier.must_have_coverage
    must_have_texts = {r.text for r in dossier.brief.requirements if r.kind == "must_have"}

    strong_pct = partial_pct = 0.0
    if must_have_texts:
        must_matches = [m for m in dossier.assessment.requirement_matches if m.requirement in must_have_texts]
        total = len(must_have_texts)
        strong_pct = sum(1 for m in must_matches if m.verdict == "strong") / total
        partial_pct = sum(1 for m in must_matches if m.verdict == "partial") / total

    return {
        "agency": {
            "name": settings.agency_name,
            "tagline": settings.agency_tagline,
            "accent": settings.agency_accent,
            "accent_soft": settings.agency_accent_soft,
        },
        "ref": ref,
        "anonymise": anonymise,
        "profile": dossier.profile,
        "brief": dossier.brief,
        "assessment": dossier.assessment,
        "timeline": dossier.timeline,
        "flags": dossier.flags,
        "counts": counts,
        "coverage": coverage,
        "strong_pct": strong_pct,
        "partial_pct": partial_pct,
        "must_have_texts": must_have_texts,
        "computed_kinds": COMPUTED_KINDS,
        "timeline_items": _timeline_items(dossier),
        "skills_by_category": _skills_by_category(dossier),
        "generated_on": date.today().strftime("%d %b %Y"),
        "elapsed": dossier.elapsed_seconds,
        "usage": dossier.usage,
        "model": settings.model,
        "warnings": dossier.warnings,
        # The filename is identity: a blind dossier whose footer reads
        # "cv_arjun_menon.pdf" is not blind.
        "source_filename": (
            "{}.{}".format(ref, dossier.document.source_format or "cv")
            if anonymise
            else Path(getattr(dossier.document, "filename", "") or "uploaded CV").name
        ),
        "source_format": dossier.document.source_format,
        "page_count": dossier.document.page_count,
    }


def render_html(dossier: Dossier, *, anonymise: Optional[bool] = None) -> str:
    template = _env.get_template("dossier.html")
    return template.render(**build_context(dossier, anonymise=anonymise))


_FOOTER_TEMPLATE = """
<div style="width:100%; font-size:7.5pt; color:#8A94A6;
            font-family:'Segoe UI',Arial,sans-serif; padding:0 14mm;
            display:flex; justify-content:space-between;">
  <span>{agency} &middot; Candidate Dossier &middot; {ref}</span>
  <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
</div>
"""


def html_to_pdf(html: str, out_path: str | Path, *, ref: str = "") -> Path:
    """Render HTML to a PDF via Playwright's bundled Chromium."""
    from playwright.sync_api import sync_playwright

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    footer = _FOOTER_TEMPLATE.format(agency=settings.agency_name, ref=ref)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            page.pdf(
                path=str(out_path),
                format="A4",
                print_background=True,
                display_header_footer=True,
                header_template="<span></span>",  # empty, but required alongside the footer
                footer_template=footer,
                margin={"top": "14mm", "bottom": "16mm", "left": "0mm", "right": "0mm"},
            )
        finally:
            browser.close()

    return out_path


def render_pdf(dossier: Dossier, out_path: str | Path, *, anonymise: Optional[bool] = None) -> Path:
    html = render_html(dossier, anonymise=anonymise)
    return html_to_pdf(html, out_path, ref=candidate_ref(dossier))
