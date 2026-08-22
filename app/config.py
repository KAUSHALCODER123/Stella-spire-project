"""Application settings, read from the environment (or a local .env file)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    # Any OpenAI-compatible endpoint. Blank means OpenAI itself.
    # OpenRouter: https://openrouter.ai/api/v1
    # NVIDIA NIM: https://integrate.api.nvidia.com/v1
    openai_base_url: str = ""

    # Overridable from .env, and per-run from the UI. Run
    # `python -m scripts.check_setup` to see which models your key can reach.
    #
    # gpt-4o is the default because it is the most widely available model that
    # supports json_schema structured outputs. Structured outputs need
    # gpt-4o-2024-08-06 or later -- gpt-4-turbo and gpt-4 cannot do it, so they
    # are not offered here: the whole pipeline depends on schema-constrained
    # generation.
    model: str = "gpt-4o"
    # Extraction is mechanical transcription against a strict schema, and the
    # eval harness measures it directly -- so it can run on a cheaper model
    # than the judgement pass without the difference being a guess. Set to the
    # same value as MODEL to disable the split.
    extraction_model: str = "gpt-4o-mini"
    max_tokens: int = 16000

    # Agency branding, applied to every generated dossier.
    agency_name: str = "Stellaspire"
    agency_tagline: str = "We take hiring personally."
    agency_accent: str = "#1F3A5F"
    agency_accent_soft: str = "#E8EDF4"

    # When true, candidate contact details are stripped from the rendered PDF
    # so the client evaluates the profile blind.
    anonymise_by_default: bool = True

    # Durable storage for uploaded documents. Optional: without it the app
    # works exactly as before, writing only to local disk.
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_bucket_resumes: str = "resumes"
    supabase_bucket_jds: str = "jds"

    upload_dir: Path = ROOT / "data" / "uploads"
    output_dir: Path = ROOT / "data" / "out"
    sample_dir: Path = ROOT / "data" / "samples"
    database_path: Path = ROOT / "data" / "spiredossier.db"


# Offered in the UI model picker. Each supports json_schema structured output.
MODEL_CHOICES = [
    # --- OpenAI ---------------------------------------------------------
    ("gpt-4o", "GPT-4o", "Balanced."),
    ("gpt-4o-mini", "GPT-4o mini", "Cheapest OpenAI option."),
    ("gpt-4.1", "GPT-4.1", "Stronger long-document reading."),
    ("gpt-4.1-mini", "GPT-4.1 mini", "Cheaper 4.1."),
    ("gpt-5", "GPT-5", "Better judgement."),
    ("gpt-5-mini", "GPT-5 mini", "Cheaper GPT-5."),
    ("gpt-5.4", "GPT-5.4", "Newer frontier model."),
    ("gpt-5.5", "GPT-5.5", "Newest frontier model."),
    # --- OpenRouter free tier -------------------------------------------
    # Only models that advertise structured_outputs are listed: the pipeline
    # is built on schema-constrained generation and silently degrades to
    # nothing without it. Verified working via responses.parse.
    ("nvidia/nemotron-3-super-120b-a12b:free", "Nemotron 3 Super 120B (free)", "262k context. No cost."),
    ("dots-studio/dots-3-note-preview:free", "Dots 3 Note (free)", "512k context. No cost."),
    ("z-ai/glm-5.2:free", "GLM 5.2 (free)", "256k context. No cost."),
    ("nvidia/nemotron-nano-9b-v2:free", "Nemotron Nano 9B (free)", "Fast and small. No cost."),
]


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
