"""Application settings, read from the environment (or a local .env file)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""

    # Overridable from .env. Run `python -m scripts.check_setup` to see which
    # models your key can actually reach.
    model: str = "gpt-5"
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


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
