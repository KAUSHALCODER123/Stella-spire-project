"""Measure what TOON encoding actually saves, on the real prompt payloads.

    python -m scripts.measure_toon

The assessment call sends a client brief and an extracted profile as prompt
input. Both are mostly uniform arrays -- requirements, positions, skills --
which TOON collapses into tables that name their fields once instead of once
per row. This script counts the tokens rather than asserting a number.

Two baselines, because the honest comparison depends on what you would
otherwise have sent:

* **Pretty JSON** (indent=2) is what most people paste into a prompt, and the
  flattering baseline.
* **Compact JSON** (no whitespace) is the hard baseline. Any saving that
  survives here is a saving in structure rather than in whitespace.

The compact figure is the one worth quoting. Beating pretty-printed JSON
mostly proves that indentation costs tokens, which nobody disputes.
"""

from __future__ import annotations

import json
import sys
from typing import List, Tuple

import tiktoken

from app.toon import encode_model

ENCODING = "o200k_base"


def count(text: str, enc) -> int:
    return len(enc.encode(text))


def payloads() -> List[Tuple[str, object]]:
    """The objects the assessment prompt actually interpolates."""
    from app.intake import RoleConstraints, brief_from_constraints
    from app.seed_data import ROLES
    from tests.fixtures import sample_dossier

    d = sample_dossier()
    out: List[Tuple[str, object]] = [
        ("brief (CFO, from a JD)", d.brief),
        ("profile (Meera Ramanathan)", d.profile),
    ]
    # Briefs built from the structured role form, which is the cheaper path
    # and the one most roles on the board take.
    for _, fields in ROLES[:4]:
        brief = brief_from_constraints(RoleConstraints(**fields))
        out.append(("brief ({})".format(brief.role_title), brief))
    return out


def main() -> int:
    enc = tiktoken.get_encoding(ENCODING)
    rows = []

    for label, obj in payloads():
        data = json.loads(obj.model_dump_json(exclude_none=True))
        toon = encode_model(obj)
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        compact = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        rows.append((label, count(toon, enc), count(compact, enc), count(pretty, enc)))

    print("TOON vs JSON on the real assessment payloads")
    print("tokenizer: {}\n".format(ENCODING))
    head = "{:<34} {:>7} {:>8} {:>8} {:>9} {:>9}"
    print(head.format("payload", "TOON", "compact", "pretty", "vs cmpct", "vs prtty"))
    print("-" * 82)

    t_toon = t_compact = t_pretty = 0
    for label, toon, compact, pretty in rows:
        t_toon += toon
        t_compact += compact
        t_pretty += pretty
        print(head.format(
            label[:34], toon, compact, pretty,
            "{:+.0f}%".format((toon - compact) / compact * 100),
            "{:+.0f}%".format((toon - pretty) / pretty * 100),
        ))

    print("-" * 82)
    print(head.format(
        "TOTAL", t_toon, t_compact, t_pretty,
        "{:+.0f}%".format((t_toon - t_compact) / t_compact * 100),
        "{:+.0f}%".format((t_toon - t_pretty) / t_pretty * 100),
    ))

    cut_compact = (t_compact - t_toon) / t_compact * 100
    cut_pretty = (t_pretty - t_toon) / t_pretty * 100
    print()
    print("{:.1f}% fewer input tokens than compact JSON.".format(cut_compact))
    print("{:.1f}% fewer than pretty-printed JSON.".format(cut_pretty))
    print()
    print("Quote the compact figure. These payloads are the structured blocks only;")
    print("the raw CV text is sent verbatim and is unaffected, so the saving on a")
    print("whole assessment call is smaller in proportion to how long the CV is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
