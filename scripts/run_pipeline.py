"""Run the full pipeline from the command line.

    python -m scripts.run_pipeline --sample
    python -m scripts.run_pipeline --cv path/to/cv.pdf --jd path/to/brief.txt
    python -m scripts.run_pipeline --sample --model gpt-4.1 --named

Faster to debug than the web UI, and it prints the token cost so the
per-dossier economics are visible rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import settings

# Rough USD per 1M tokens, for an order-of-magnitude cost line only.
_PRICES = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


def _cost(model: str, tin: int, tout: int) -> str:
    for prefix, (pin, pout) in _PRICES.items():
        if model.startswith(prefix):
            usd = tin / 1e6 * pin + tout / 1e6 * pout
            return "~${:.4f} (about Rs {:.2f})".format(usd, usd * 88)
    return "unknown for this model"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build one dossier.")
    ap.add_argument("--sample", action="store_true", help="use the bundled sample CV and brief")
    ap.add_argument("--cv", help="path to a CV (pdf/docx/txt)")
    ap.add_argument("--jd", help="path to a job description text file")
    ap.add_argument("--model", default=None, help="override the model for this run")
    ap.add_argument("--named", action="store_true", help="render identified instead of blind")
    ap.add_argument("--out", default=None, help="output PDF path")
    args = ap.parse_args()

    if args.sample:
        cv_path = settings.sample_dir / "cv_arjun_menon.txt"
        jd_path = settings.sample_dir / "jd_genai_platform_lead.txt"
    elif args.cv and args.jd:
        cv_path, jd_path = Path(args.cv), Path(args.jd)
    else:
        ap.error("pass --sample, or both --cv and --jd")

    for p in (cv_path, jd_path):
        if not p.exists():
            print("Missing file: {}".format(p))
            return 1

    from app.pipeline import build_dossier
    from app.render.dossier import candidate_ref, render_pdf

    model = args.model or settings.model
    print("Building dossier with {}...".format(model))
    print("  CV:    {}".format(cv_path.name))
    print("  Brief: {}".format(jd_path.name))
    print()

    dossier = build_dossier(
        cv_path=cv_path,
        jd_text=jd_path.read_text(encoding="utf-8"),
        model=model,
    )

    ref = candidate_ref(dossier)
    t = dossier.timeline
    c = dossier.match_counts

    print("-" * 62)
    print("{}  {}".format(ref, dossier.brief.role_title))
    print("-" * 62)
    print("Name extracted     : {}".format(dossier.profile.full_name))
    print("Positions          : {} parsed, {} unreadable".format(len(t.intervals), len(t.unparseable)))
    print("Experience         : {} yrs   Avg tenure: {} yrs".format(t.total_experience_years, t.average_tenure_years))
    print("Gaps               : {}".format(
        ", ".join("{}mo {}-{}".format(m, a.strftime("%b%y"), b.strftime("%b%y")) for a, b, m in t.gaps) or "none"))
    print("Requirements       : {} total".format(len(dossier.assessment.requirement_matches)))
    print("  strong {}   partial {}   unclear {}   absent {}".format(c["strong"], c["partial"], c["unclear"], c["absent"]))
    if dossier.must_have_coverage is not None:
        print("Must-have coverage : {:.0f}%".format(dossier.must_have_coverage * 100))
    print()

    print("Flags ({}):".format(len(dossier.flags)))
    for f in dossier.flags:
        print("  [{:<6}] {:<24} {}".format(f.severity, f.kind, f.summary[:88]))
    print()

    if dossier.warnings:
        print("Warnings:")
        for w in dossier.warnings:
            print("  ! {}".format(w))
        print()

    # A quick contract check: no 'strong' verdict may exist without a quote.
    unquoted = [m for m in dossier.assessment.requirement_matches if m.verdict == "strong" and not m.evidence]
    print("Unquoted 'strong' verdicts: {} {}".format(len(unquoted), "(good)" if not unquoted else "(CONTRACT VIOLATION)"))

    u = dossier.usage
    print("Tokens             : {} in / {} out over {} calls".format(u.input_tokens, u.output_tokens, u.calls))
    print("Cost               : {}".format(_cost(model, u.input_tokens, u.output_tokens)))
    print("Elapsed            : {}s".format(dossier.elapsed_seconds))

    out = Path(args.out) if args.out else settings.output_dir / "{}.pdf".format(ref)
    render_pdf(dossier, out, anonymise=not args.named)
    print("PDF                : {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
