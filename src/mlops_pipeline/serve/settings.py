"""Runtime configuration, read from ``MLOPS_*`` environment variables or a local ``.env``.

Defaults keep everything the app writes under ``mlstore/``, the folder the DVC pipeline and the
container image use, so ``train`` followed by ``serve`` needs no configuration at all.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Blank values count as unset, so a .env copied straight from .env.example keeps the defaults.
    model_config = SettingsConfigDict(
        env_prefix="MLOPS_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    tracking_uri: str = "sqlite:///mlstore/mlflow.db"
    model_name: str = "credit-default"
    model_alias: str = "champion"
    predlog_path: str = "mlstore/predlog.sqlite"
    reference_path: str = "data/credit_default.csv.gz"
    postmortem_dir: str = "mlstore/postmortems"
    slack_webhook_url: str | None = None


@lru_cache
def get_settings() -> Settings:
    """One cached instance per process; tests pass their own ``Settings`` instead."""
    return Settings()
