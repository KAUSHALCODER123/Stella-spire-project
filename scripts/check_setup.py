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

    # Use the app's own client so the configured base URL (OpenRouter, NVIDIA
    # NIM, anything OpenAI-compatible) is honoured. Building a fresh client
    # here silently tested openai.com instead of the provider in use.
    from app.extract.llm import get_client

    client = get_client()
    if settings.openai_base_url:
        print("OK    Provider: {}".format(settings.openai_base_url))

    # --- what can this key see? -------------------------------------------
    try:
        available = sorted(m.id for m in client.models.list())
    except Exception as exc:  # noqa: BLE001 - surface anything the API says
        print("FAIL  Could not list models: {}: {}".format(type(exc).__name__, exc))
        print("      An invalid or revoked key is the usual cause.")
        return 1

    from app.config import MODEL_CHOICES

    print("OK    Key is valid. {} models visible.".format(len(available)))
    offered = [m[0] for m in MODEL_CHOICES]
    reachable = [m for m in offered if m in available]
    missing = [m for m in offered if m not in available]
    # Print the full picker state. An earlier version truncated this list and
    # led to a wrong conclusion about which models the key could reach.
    print("      Picker options reachable ({}/{}): {}".format(len(reachable), len(offered), ", ".join(reachable) or "none"))
    if missing:
        print("      Not on this key, hidden from the picker: {}".format(", ".join(missing)))

    # --- is the configured model one of them? -----------------------------
    if settings.model not in available:
        print("FAIL  Configured model '{}' is not available to this key.".format(settings.model))
        if reachable:
            print("      Set MODEL={} in .env instead.".format(reachable[0]))
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

    # --- persistence ------------------------------------------------------
    from app import db as _db

    state = _db.status()
    if state["connected"]:
        print("OK    Database: {}".format(state["detail"]))
    else:
        print("WARN  Database: {}".format(state["detail"]))
        print("      State is in memory only, so a restart clears accounts,")
        print("      roles, applicants and dossiers.")
        if "tables not created" in (state["detail"] or ""):
            print("      Fix: open the Supabase SQL editor and run")
            print("           migrations/001_init.sql")

    from app.storage import get_storage

    client = get_storage()
    if client is None:
        print("WARN  File archiving: not configured (local disk only)")
    else:
        check = client.check()
        print("{}  File archiving: {}".format(
            "OK   " if check.get("ok") else "WARN ",
            "supabase storage" if check.get("ok") else check.get("detail")))

    print("-" * 52)
    print("All checks passed. Next:  python -m scripts.run_pipeline --sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
