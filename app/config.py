"""Application settings, read from the environment (or a local .env file)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""

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

    upload_dir: Path = ROOT / "data" / "uploads"
    output_dir: Path = ROOT / "data" / "out"
    sample_dir: Path = ROOT / "data" / "samples"
    database_path: Path = ROOT / "data" / "spiredossier.db"


# Offered in the UI model picker. Each supports json_schema structured output.
MODEL_CHOICES = [
    ("gpt-4o", "GPT-4o", "Balanced. The default."),
    ("gpt-4o-mini", "GPT-4o mini", "Cheapest. Fine for bulk extraction."),
    ("gpt-4.1", "GPT-4.1", "Stronger long-document reading."),
    ("gpt-4.1-mini", "GPT-4.1 mini", "Cheaper 4.1."),
    ("gpt-5", "GPT-5", "Better judgement on the assessment pass."),
    ("gpt-5-mini", "GPT-5 mini", "Cheaper GPT-5."),
    ("gpt-5.4", "GPT-5.4", "Newer frontier model."),
    ("gpt-5.5", "GPT-5.5", "Newest frontier model. Highest cost."),
]


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
