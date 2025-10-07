"""Entity resolution utilities for fighters."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pandas as pd

CROSSWALK_PATH = Path("data/external/fighter_crosswalk.parquet")
CROSSWALK_PATH.parent.mkdir(parents=True, exist_ok=True)


_DEFAULT_DATA = pd.DataFrame(
    [
        {"canonical_id": "f-001", "canonical_name": "Jon Jones", "alias": "Jonny Bones"},
        {"canonical_id": "f-001", "canonical_name": "Jon Jones", "alias": "jon jones"},
        {"canonical_id": "f-002", "canonical_name": "Alexander Volkanovski", "alias": "alex volkanovski"},
        {"canonical_id": "f-002", "canonical_name": "Alexander Volkanovski", "alias": "alexander the great"},
    ]
)
if not CROSSWALK_PATH.exists():  # pragma: no cover - executed during bootstrap
    _DEFAULT_DATA.to_parquet(CROSSWALK_PATH, index=False)


@lru_cache(maxsize=1)
def load_crosswalk() -> pd.DataFrame:
    frame = pd.read_parquet(CROSSWALK_PATH)
    frame["alias_lower"] = frame["alias"].str.lower()
    return frame


def canonicalize_name(name: str) -> str:
    crosswalk = load_crosswalk()
    match = crosswalk[crosswalk["alias_lower"] == name.lower()]
    if not match.empty:
        return str(match.iloc[0]["canonical_name"])
    return name


def merge_aliases(frame: pd.DataFrame, name_column: str = "fighter") -> pd.DataFrame:
    crosswalk = load_crosswalk()[["canonical_id", "canonical_name", "alias_lower"]]
    frame = frame.copy()
    if name_column not in frame.columns:
        raise KeyError(f"Column {name_column} missing from dataframe")
    frame["alias_lower"] = frame[name_column].str.lower()
    merged = frame.merge(crosswalk, on="alias_lower", how="left")
    merged["canonical_name"] = merged["canonical_name"].fillna(frame[name_column])
    merged["canonical_id"] = merged["canonical_id"].fillna(merged[name_column].str.lower())
    return merged.drop(columns=["alias_lower"])


__all__ = ["canonicalize_name", "merge_aliases", "load_crosswalk"]
