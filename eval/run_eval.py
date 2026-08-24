"""Measure the extraction pass against hand-labelled ground truth.

    python -m eval.run_eval
    python -m eval.run_eval --model gpt-4o
    python -m eval.run_eval --case 02_two_column

Why this exists: the product's whole claim is that a recruiter can trust what
comes out of it. "Trust it" is not a claim you can make without a number, and
a number you have not published is one you have not really checked.

The output is deliberately blunt about failures. Per-field accuracy, then the
specific cases that missed and what they missed, so a regression can be
attributed to a cause instead of just showing up as a smaller percentage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def norm_text(value: Optional[str]) -> str:
    """Case and whitespace folded; punctuation that varies by typist removed."""
    if not value:
        return ""
    out = value.lower().strip()
    out = out.replace("&", "and").replace("’", "'")
    out = re.sub(r"[.,]", "", out)
    return re.sub(r"\s+", " ", out)


def digits(value: Optional[str]) -> str:
    return re.sub(r"\D", "", value or "")


def titles_match(expected: str, got: str) -> bool:
    """Titles are worded loosely; require the distinctive words, not the string.

    "VP, Financial Reporting" and "Vice President Financial Reporting" are the
    same job, and scoring them as a miss would measure phrasing rather than
    extraction.
    """
    e, g = norm_text(expected), norm_text(got)
    if e == g:
        return True
    synonyms = {"vp": "vice president", "sr": "senior", "asst": "assistant",
                "mgr": "manager", "fp&a": "fpanda"}
    for short, long in synonyms.items():
        e, g = e.replace(short, long), g.replace(short, long)
    if e == g:
        return True
    e_words = {w for w in e.split() if len(w) > 2}
    g_words = {w for w in g.split() if len(w) > 2}
    if not e_words or not g_words:
        return False
    if not (e_words <= g_words or g_words <= e_words):
        return False
    # One title containing the other is not enough on its own: "Officer" sits
    # inside "Chief Financial Officer", and accepting that would score a
    # truncated title as a correct one. Require the shorter side to carry at
    # least half the distinctive words, and never fewer than two unless the
    # full title genuinely is one word.
    shorter, longer = sorted((len(e_words), len(g_words)))
    return shorter >= max(min(2, longer), (longer + 1) // 2)


def companies_match(expected: str, got: str) -> bool:
    e, g = norm_text(expected), norm_text(got)
    for suffix in (" ltd", " limited", " inc", " llp", " co", " india", " private"):
        e, g = e.replace(suffix, ""), g.replace(suffix, "")
    e, g = e.strip(), g.strip()
    return e == g or e in g or g in e


@dataclass
class FieldTally:
    name: str
    checked: int = 0
    correct: int = 0
    misses: List[str] = field(default_factory=list)

    def record(self, ok: bool, detail: str = "") -> None:
        self.checked += 1
        if ok:
            self.correct += 1
        elif detail:
            self.misses.append(detail)

    @property
    def pct(self) -> Optional[float]:
        return (self.correct / self.checked * 100) if self.checked else None


class Scorecard:
    def __init__(self) -> None:
        self.fields: Dict[str, FieldTally] = {}
        self.case_notes: Dict[str, List[str]] = {}
        self.failures: List[str] = []

    def tally(self, name: str) -> FieldTally:
        return self.fields.setdefault(name, FieldTally(name))

    def check(self, case_id: str, name: str, ok: bool, detail: str = "") -> None:
        self.tally(name).record(ok, "{}: {}".format(case_id, detail) if detail else "")
        if not ok and detail:
            self.case_notes.setdefault(case_id, []).append("{} -- {}".format(name, detail))


def score_case(card: Scorecard, truth: dict, profile) -> None:
    cid = truth["id"]

    card.check(cid, "name", norm_text(profile.full_name) == norm_text(truth["full_name"]),
               "expected {!r}, got {!r}".format(truth["full_name"], profile.full_name))
    card.check(cid, "email", norm_text(profile.email) == norm_text(truth["email"]),
               "expected {!r}, got {!r}".format(truth["email"], profile.email))

    if truth.get("phone_digits") is not None:
        card.check(cid, "phone", digits(profile.phone) == truth["phone_digits"],
                   "expected {}, got {!r}".format(truth["phone_digits"], profile.phone))

    for money in ("notice_period_days", "current_ctc_lpa", "expected_ctc_lpa"):
        if truth.get(money) is not None:
            got = getattr(profile, money, None)
            card.check(cid, money, got is not None and abs(float(got) - float(truth[money])) < 0.51,
                       "expected {}, got {}".format(truth[money], got))

    # --- positions --------------------------------------------------------
    expected_positions = truth["positions"]
    got_positions = list(profile.positions)

    card.check(cid, "position count", len(got_positions) == len(expected_positions),
               "expected {} roles, got {}".format(len(expected_positions), len(got_positions)))

    for exp in expected_positions:
        # Match on company first; where an employer repeats (promotions), the
        # closest start date decides which row is which.
        candidates = [p for p in got_positions if companies_match(exp["company"], p.company)]
        if not candidates:
            card.check(cid, "company", False, "missing {!r}".format(exp["company"]))
            for missing in ("title", "start date", "end date"):
                card.check(cid, missing, False, "{} not found".format(exp["company"]))
            continue

        card.check(cid, "company", True)
        match = min(candidates, key=lambda p: 0 if (p.start or "") == exp["start"] else 1)

        card.check(cid, "title", titles_match(exp["title"], match.title),
                   "{}: expected {!r}, got {!r}".format(exp["company"], exp["title"], match.title))
        card.check(cid, "start date", (match.start or "").strip() == exp["start"],
                   "{}: expected {}, got {!r}".format(exp["company"], exp["start"], match.start))
        card.check(cid, "end date", (match.end or "").strip().lower() == exp["end"].lower(),
                   "{}: expected {}, got {!r}".format(exp["company"], exp["end"], match.end))

    # --- skills: recall of the terms a recruiter would search on ----------
    blob = " ".join(norm_text(s.name) for s in profile.skills)
    for wanted in truth.get("skills_must_include", []):
        card.check(cid, "key skills", norm_text(wanted) in blob,
                   "missing skill {!r}".format(wanted))

    # --- did it notice the break, without being asked to judge it? --------
    if truth.get("expect_break_noted"):
        notes = " ".join(profile.extraction_notes).lower()
        card.check(cid, "career break noted", "break" in notes or "parenting" in notes,
                   "the break was not mentioned in extraction_notes")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Score CV extraction against ground truth.")
    ap.add_argument("--model", default=None, help="override the extraction model")
    ap.add_argument("--case", default=None, help="run a single case by id")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    args = ap.parse_args()

    from app.config import settings
    from app.extract.documents import extract_text
    from app.extract.llm import Usage, extract_profile

    truth_all = json.loads((ROOT / "ground_truth.json").read_text(encoding="utf-8"))["cases"]
    if args.case:
        truth_all = [t for t in truth_all if t["id"] == args.case]
        if not truth_all:
            print("No such case: {}".format(args.case))
            return 1

    model = args.model or settings.extraction_model or settings.model
    print("Extraction eval — {} cases, model {}".format(len(truth_all), model))
    print("=" * 78)

    card = Scorecard()
    usage = Usage()
    started = time.perf_counter()

    for truth in truth_all:
        path = CASES / "{}.txt".format(truth["id"])
        if not path.exists():
            print("  MISSING CASE FILE {}".format(path))
            continue

        case_started = time.perf_counter()
        try:
            document = extract_text(path)
            profile = extract_profile(document.text, usage=usage, model=model)
        except Exception as exc:  # noqa: BLE001 - a failure is a result
            card.failures.append("{}: {}".format(truth["id"], exc))
            print("  {:<20} FAILED  {}".format(truth["id"], str(exc)[:60]))
            # An exhausted quota will not fix itself between cases. Running the
            # other nine produces nine identical errors and a report of zeroes
            # that looks like a measurement.
            if getattr(exc, "kind", None) == "quota":
                print()
                print("Stopping: {}".format(exc))
                return 2
            continue

        before = {n: (t.checked, t.correct) for n, t in card.fields.items()}
        score_case(card, truth, profile)
        checked = sum(t.checked for t in card.fields.values()) - sum(v[0] for v in before.values())
        correct = sum(t.correct for t in card.fields.values()) - sum(v[1] for v in before.values())

        if not args.quiet:
            print("  {:<20} {:>3}/{:<3} {:>5.0f}%   {:.0f}s   {}".format(
                truth["id"], correct, checked, correct / checked * 100 if checked else 0,
                time.perf_counter() - case_started, truth["difficulty"][:34]))

    # --- report -----------------------------------------------------------
    print()
    print("PER-FIELD ACCURACY")
    print("-" * 78)
    order = ["name", "email", "phone", "position count", "company", "title",
             "start date", "end date", "key skills", "notice_period_days",
             "current_ctc_lpa", "expected_ctc_lpa", "career break noted"]
    for name in order + [n for n in card.fields if n not in order]:
        t = card.fields.get(name)
        if not t or not t.checked:
            continue
        bar = "#" * int((t.pct or 0) / 5)
        print("  {:<22} {:>3}/{:<3} {:>5.0f}%  {}".format(name, t.correct, t.checked, t.pct, bar))

    total_checked = sum(t.checked for t in card.fields.values())
    total_correct = sum(t.correct for t in card.fields.values())
    overall = total_correct / total_checked * 100 if total_checked else 0
    print("-" * 78)
    print("  {:<22} {:>3}/{:<3} {:>5.1f}%".format("OVERALL", total_correct, total_checked, overall))

    if card.case_notes:
        print()
        print("WHAT MISSED")
        print("-" * 78)
        for cid, notes in card.case_notes.items():
            diff = next((t["difficulty"] for t in truth_all if t["id"] == cid), "")
            print("  {}  ({})".format(cid, diff))
            for note in notes[:6]:
                print("      {}".format(note[:100]))
            if len(notes) > 6:
                print("      ... and {} more".format(len(notes) - 6))

    if card.failures:
        print()
        print("CASES THAT DID NOT RUN")
        for f in card.failures:
            print("  {}".format(f))

    elapsed = time.perf_counter() - started
    print()
    print("{} cases in {:.0f}s · {} in / {} out tokens".format(
        len(truth_all), elapsed, usage.input_tokens, usage.output_tokens))

    if not total_checked:
        print("Nothing was scored, so no report was written -- a file of zeroes "
              "reads like a measurement when it is actually an outage.")
        return 2

    RESULTS.mkdir(exist_ok=True)
    report = {
        "model": model,
        "cases": len(truth_all),
        "overall_pct": round(overall, 1),
        "fields": {n: {"checked": t.checked, "correct": t.correct, "pct": round(t.pct or 0, 1)}
                   for n, t in card.fields.items() if t.checked},
        "misses": card.case_notes,
        "failures": card.failures,
        "tokens": {"input": usage.input_tokens, "output": usage.output_tokens},
        "elapsed_seconds": round(elapsed, 1),
    }
    out = RESULTS / "{}.json".format(re.sub(r"[^A-Za-z0-9._-]", "_", model))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report written to {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
