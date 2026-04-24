"""Project configuration using Pydantic settings."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Hugging Face settings
    hf_model: str = "google/gemma-4-26B-A4B-it"
    hf_api_key: str | None = None

    # Scraping settings
    scrape_interval_minutes: int = 1
    target_urls: list[str] = [
        "https://itln.in/supply-chain",
        "https://itln.in/road-transportation"
    ]

    # ChromaDB settings
    chromadb_path: Path = BASE_DIR / "data" / "cache" / "chromadb"

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"

    # App settings
    debug: bool = False
    log_level: str = "INFO"

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug_flag(cls, value: object) -> bool:
        """Treat non-boolean debug values like 'release' as False."""

        if isinstance(value, bool):
            return value

        if value is None:
            return False

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
            return False

        return bool(value)


settings = Settings()
