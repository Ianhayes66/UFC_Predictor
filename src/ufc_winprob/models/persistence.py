"""Model persistence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_artifact(obj: Any, name: str) -> Path:
    path = MODEL_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_artifact(name: str) -> Any:
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model artifact {name} missing")
    return joblib.load(path)


__all__ = ["save_artifact", "load_artifact", "MODEL_DIR"]
