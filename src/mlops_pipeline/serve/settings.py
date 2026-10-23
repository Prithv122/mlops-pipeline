"""Runtime configuration, read from ``MLOPS_*`` environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLOPS_", extra="ignore")

    tracking_uri: str = "sqlite:///mlflow.db"
    model_name: str = "credit-default"
    model_alias: str = "champion"
    predlog_path: str = "predlog.sqlite"
    reference_path: str = "data/credit_default.csv.gz"
    postmortem_dir: str = "postmortems"
    slack_webhook_url: str | None = None


@lru_cache
def get_settings() -> Settings:
    """One cached instance per process; tests pass their own ``Settings`` instead."""
    return Settings()
