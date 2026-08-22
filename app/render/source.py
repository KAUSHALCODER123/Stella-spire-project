"""Render the raw CV with every traced quote marked.

The source pane is the other half of the trace: the dossier makes a claim, and
this shows the exact characters it was lifted from. Spans come from
app/verify.py as offsets into the original text, so what gets highlighted is
the real document, not a reconstruction of it.
"""

from __future__ import annotations

from html import escape
from typing import List, Optional

from app.verify import VerificationReport


def render_source(text: str, report: Optional[VerificationReport]) -> str:
    """Escaped CV text with <mark> around each located quote.

    Each mark carries data-trace so the dossier pane can scroll to and light
    up the span belonging to a given claim.
    """
    if not text:
        return '<p class="src-empty">No text could be read from this document.</p>'

    spans = report.spans() if report else []
    if not spans:
        return '<pre class="src">{}</pre>'.format(escape(text))

    out: List[str] = []
    cursor = 0
    for start, end, trace_index in spans:
        # Defensive: a stale or out-of-range span must not silently truncate
        # the document.
        if start < cursor or start >= len(text):
            continue
        end = min(end, len(text))
        out.append(escape(text[cursor:start]))
        out.append(
            '<mark class="src-mark" id="trace-{i}" data-trace="{i}" tabindex="-1">{body}</mark>'.format(
                i=trace_index, body=escape(text[start:end])
            )
        )
        cursor = end

    out.append(escape(text[cursor:]))
    return '<pre class="src">{}</pre>'.format("".join(out))
