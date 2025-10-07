"""Application settings management using Pydantic."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PathsConfig(BaseModel):
    """Simple container for project path configuration."""

    raw: Path
    interim: Path
    processed: Path
    models: Path


class ProvidersConfig(BaseModel):
    """Provider feature flags and configuration."""

    use_tapology: bool
    use_odds_api: bool
    odds_market: str
    disable_if_robots_disallow: bool = True
    rate_limit_seconds: float = 0.75
    rate_limit_jitter_seconds: float = 0.3
    user_agent: str = "ufc-winprob-bot/1.0"


class ModelConfig(BaseModel):
    """Model section configuration."""

    base_seed: int
    calibration: str


class EloConfig(BaseModel):
    """Component Elo configuration."""

    components: list[str]
    k_base: float
    decay_days_half_life: int
    division_half_life_overrides: Dict[str, int]


class JobsConfig(BaseModel):
    """Job scheduling configuration."""

    daily_refresh_cron: str


class StorageConfig(BaseModel):
    """Storage backend configuration."""

    backend: str


class LoggingConfig(BaseModel):
    """Logging configuration options."""

    level: str = "INFO"
    json: bool = False
    directory: Path = Path("data/logs")


class AppSettings(BaseSettings):
    """Application settings loaded from YAML and environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_name: str = "ufc-winprob"
    paths: PathsConfig
    providers: ProvidersConfig
    model: ModelConfig
    elo: EloConfig
    jobs: JobsConfig
    storage: StorageConfig
    logging: LoggingConfig = LoggingConfig()

    odds_api_key: Optional[str] = None
    http_proxy: Optional[str] = None
    tz: str = "America/Los_Angeles"
    use_live_odds: bool = False

    @classmethod
    def from_file(cls, env: str | None = None) -> "AppSettings":
        """Load configuration from YAML file merged with overrides.

        Args:
            env: Optional environment name (base, dev, prod).

        Returns:
            Resolved :class:`AppSettings` instance.
        """

        config_dir = _PROJECT_ROOT / "config"
        base_path = config_dir / "base.yaml"
        data: Dict[str, Any] = _read_yaml(base_path)
        if env and env != "base":
            env_path = config_dir / f"{env}.yaml"
            if env_path.exists():
                env_data = _read_yaml(env_path)
                if "inherit" in env_data:
                    env_data.pop("inherit")
                data = _deep_update(data, env_data)
        resolved = cls(**data)
        return resolved


def _deep_update(original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(original.get(key), dict):
            original[key] = _deep_update(original[key], value)
        else:
            original[key] = value
    return original


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data: Dict[str, Any] = yaml.safe_load(file) or {}
    inherit = data.pop("inherit", None)
    if inherit:
        base_data = _read_yaml(path.parent / f"{inherit}.yaml")
        data = _deep_update(base_data, data)
    return data


@lru_cache(maxsize=1)
def get_settings(env: str | None = None) -> AppSettings:
    """Return cached application settings."""

    return AppSettings.from_file(env)


__all__ = ["AppSettings", "get_settings"]
