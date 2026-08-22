"""Preflight check. Run this first, before anything else.

    python -m scripts.check_setup

Verifies that the API key works, reports which models the key can actually
reach, and makes one real structured call end to end. Catching a bad key or an
unavailable model here takes ten seconds; catching it during a demo does not.
"""

from __future__ import annotations

import sys

from pydantic import BaseModel


class _Ping(BaseModel):
    city: str
    country: str


def main() -> int:
    from app.config import settings

    print("SpireDossier setup check")
    print("-" * 52)

    if not settings.openai_api_key:
        print("FAIL  No API key found.")
        print()
        print("  1. cp .env.example .env")
        print("  2. Put your key in .env as:  OPENAI_API_KEY=sk-...")
        print("  3. Re-run this script.")
        return 1

    key = settings.openai_api_key
    print("OK    Key loaded ({}...{}, {} chars)".format(key[:7], key[-4:], len(key)))

    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=60.0)

    # --- what can this key see? -------------------------------------------
    try:
        available = sorted(m.id for m in client.models.list())
    except Exception as exc:  # noqa: BLE001 - surface anything the API says
        print("FAIL  Could not list models: {}: {}".format(type(exc).__name__, exc))
        print("      An invalid or revoked key is the usual cause.")
        return 1

    interesting = [m for m in available if m.startswith(("gpt-5", "gpt-4.1", "gpt-4o", "o3", "o4"))]
    print("OK    Key is valid. {} models visible.".format(len(available)))
    if interesting:
        print("      Suitable for this project: {}".format(", ".join(interesting[:12])))

    # --- is the configured model one of them? -----------------------------
    if settings.model not in available:
        print("FAIL  Configured model '{}' is not available to this key.".format(settings.model))
        if interesting:
            print("      Set MODEL={} in .env instead.".format(interesting[0]))
        return 1
    print("OK    Configured model '{}' is available.".format(settings.model))

    # --- does a structured call actually work? ----------------------------
    try:
        response = client.responses.parse(
            model=settings.model,
            instructions="Extract the city and country.",
            input="The office is in Bengaluru, India.",
            text_format=_Ping,
        )
    except Exception as exc:  # noqa: BLE001
        print("FAIL  Structured output call failed: {}: {}".format(type(exc).__name__, exc))
        return 1

    parsed = response.output_parsed
    if parsed is None or parsed.city.lower() != "bengaluru":
        print("FAIL  Structured output returned something unexpected: {!r}".format(parsed))
        return 1

    usage = response.usage
    print("OK    Structured output works (parsed: {}, {}).".format(parsed.city, parsed.country))
    if usage:
        print("      Test call used {} in / {} out tokens.".format(usage.input_tokens, usage.output_tokens))

    print("-" * 52)
    print("All checks passed. Next:  python -m scripts.run_pipeline --sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
