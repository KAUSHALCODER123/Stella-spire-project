"""TOON encoding for data we put *into* a prompt.

TOON (Token-Oriented Object Notation) writes the JSON data model with
indentation instead of braces and, crucially, collapses a uniform array of
objects into a table that names its fields once:

    JSON                                TOON
    "skills": [                         skills[3]{name,category}:
      {"name":"Go","category":"lang"},    Go,lang
      {"name":"K8s","category":"infra"},  K8s,infra
      {"name":"SQL","category":"data"}    SQL,data
    ]

Every repetition of a key across rows disappears -- when the rows really are
uniform, which in this project is rarer than it looks.

What `scripts/measure_toon.py` actually measures (tiktoken o200k_base, against
the compact JSON you would otherwise send):

* A job brief comes out ~20% smaller. Its `requirements` are uniform and do
  collapse into a table.
* An extracted profile comes out ~7% LARGER. `positions` can never collapse
  because it holds a nested `achievements` list, and neither positions nor
  skills are uniform once `exclude_none` has dropped different optional keys
  from different rows.
* An assessment prompt sends exactly one brief and one profile, so the two
  effects cancel: the measured end-to-end difference is -0.2%.

So this is a readability choice, not a cost optimisation. The tabular form is
genuinely easier to scan when debugging a prompt, which is why it is still
here, but any claim about token savings should be checked against the script
before it is repeated.

INPUT ONLY. Model output stays JSON: schema-constrained generation is what
guarantees the response parses, and giving that up to save output tokens
would trade a correctness guarantee for a cost saving. Encoding here, decoding
nothing.

Spec: https://github.com/toon-format/spec (working draft 4.1)
"""

from __future__ import annotations

import math
import re
from typing import Any, List

INDENT = "  "
DELIMITER = ","

# Unquoted keys are conservative on purpose; anything else gets quoted.
_SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
# A bare string that would read back as a number must be quoted.
_NUMERIC = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$")
_RESERVED = {"true", "false", "null"}
_MUST_QUOTE = set(':"\\[]{}') | {DELIMITER}


def _escape(text: str) -> str:
    out = text.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return "".join(c if ord(c) >= 0x20 else "\\u{:04x}".format(ord(c)) for c in out)


def _needs_quotes(text: str) -> bool:
    if text == "" or text != text.strip():
        return True
    if text in _RESERVED or _NUMERIC.match(text):
        return True
    if text[0] in "-#":
        return True
    return any(c in _MUST_QUOTE or ord(c) < 0x20 for c in text)


def encode_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "null"
        # Canonical decimal, no exponent, no trailing ".0" noise.
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return repr(round(value, 10)).rstrip("0").rstrip(".")
    text = str(value)
    return '"{}"'.format(_escape(text)) if _needs_quotes(text) else text


def _key(name: str) -> str:
    return name if _SAFE_KEY.match(name) else '"{}"'.format(_escape(name))


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_uniform_table(rows: List[Any]) -> bool:
    """Can this array collapse into a table?

    Every element must be an object with the same keys in the same order, and
    every value must be a scalar. Nested structures would need the nested-brace
    form, which is not worth the complexity here -- those fall back to list
    form and still encode correctly.
    """
    if not rows or not all(isinstance(r, dict) for r in rows):
        return False
    fields = list(rows[0].keys())
    if not fields:
        return False
    return all(
        list(r.keys()) == fields and all(_is_scalar(v) for v in r.values())
        for r in rows
    )


def _encode_value(key: str, value: Any, depth: int, out: List[str]) -> None:
    pad = INDENT * depth

    if _is_scalar(value):
        out.append("{}{}: {}".format(pad, _key(key), encode_scalar(value)))
        return

    if isinstance(value, dict):
        if not value:
            # An empty object has no rows to indent; keep the key visible.
            out.append("{}{}:".format(pad, _key(key)))
            return
        out.append("{}{}:".format(pad, _key(key)))
        for k, v in value.items():
            _encode_value(str(k), v, depth + 1, out)
        return

    if isinstance(value, (list, tuple)):
        rows = list(value)
        if not rows:
            out.append("{}{}: []".format(pad, _key(key)))
            return

        if all(_is_scalar(r) for r in rows):
            out.append("{}{}[{}]: {}".format(
                pad, _key(key), len(rows), DELIMITER.join(encode_scalar(r) for r in rows)))
            return

        if _is_uniform_table(rows):
            fields = list(rows[0].keys())
            out.append("{}{}[{}]{{{}}}:".format(
                pad, _key(key), len(rows), DELIMITER.join(_key(str(f)) for f in fields)))
            for row in rows:
                out.append("{}{}{}".format(
                    pad, INDENT, DELIMITER.join(encode_scalar(row[f]) for f in fields)))
            return

        # Mixed or nested: list form, one item per line.
        out.append("{}{}[{}]:".format(pad, _key(key), len(rows)))
        for item in rows:
            if _is_scalar(item):
                out.append("{}{}- {}".format(pad, INDENT, encode_scalar(item)))
            elif isinstance(item, dict):
                first = True
                for k, v in item.items():
                    if first and _is_scalar(v):
                        out.append("{}{}- {}: {}".format(pad, INDENT, _key(str(k)), encode_scalar(v)))
                        first = False
                    else:
                        if first:
                            out.append("{}{}-".format(pad, INDENT))
                            first = False
                        _encode_value(str(k), v, depth + 2, out)
            else:
                out.append("{}{}-".format(pad, INDENT))
                _encode_value("items", item, depth + 2, out)
        return

    out.append("{}{}: {}".format(pad, _key(key), encode_scalar(value)))


def encode(data: Any) -> str:
    """Encode a JSON-compatible value as TOON."""
    out: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            _encode_value(str(k), v, 0, out)
    elif isinstance(data, (list, tuple)):
        _encode_value("items", list(data), 0, out)
    else:
        return encode_scalar(data)
    return "\n".join(out)


def encode_model(model, exclude_none: bool = True) -> str:
    """Encode a Pydantic model. Dropping nulls is most of the saving on its own."""
    return encode(model.model_dump(mode="json", exclude_none=exclude_none))
