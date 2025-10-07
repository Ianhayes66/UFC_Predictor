"""Lightweight Great Expectations integration for pipeline gating."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from great_expectations.dataset import PandasDataset

from ..logging import logger

_EXPECTATIONS_DIR = Path("data/expectations")


class DataQualityError(RuntimeError):
    """Raised when a dataset fails a critical expectation suite."""


@dataclass(slots=True)
class ExpectationResult:
    """Represents the outcome of executing an expectation suite."""

    suite_name: str
    success: bool
    failures: list[str]


def _run_suite(df: pd.DataFrame, suite_path: Path) -> ExpectationResult:
    if not suite_path.exists():
        raise FileNotFoundError(f"Expectation suite {suite_path} missing")

    dataset = PandasDataset(df.copy())
    dataset.set_default_expectation_argument("result_format", "SUMMARY")
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for expectation in payload.get("expectations", []):
        expectation_type: str = expectation["expectation_type"]
        kwargs = expectation.get("kwargs", {})
        expectation_method = getattr(dataset, expectation_type, None)
        if expectation_method is None:
            raise AttributeError(f"Unknown expectation: {expectation_type}")
        try:
            result = expectation_method(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive guard
            failures.append(f"{expectation_type} errored: {exc}")
            continue
        if not result.success:
            details = result.result or {}
            unexpected = details.get("unexpected_list") or details.get("unexpected_count")
            failures.append(f"{expectation_type} failed: {unexpected!r}")
    success = len(failures) == 0
    return ExpectationResult(
        suite_name=payload.get("expectation_suite_name", suite_path.stem),
        success=success,
        failures=failures,
    )


def _validate(df: pd.DataFrame, suite_name: str) -> None:
    suite_path = _EXPECTATIONS_DIR / f"{suite_name}.json"
    result = _run_suite(df, suite_path)
    if not result.success:
        for failure in result.failures:
            logger.error("Data quality failure (%s): %s", suite_name, failure)
        raise DataQualityError(
            f"Expectation suite '{result.suite_name}' failed with {len(result.failures)} violation(s)."
        )
    logger.debug("Expectation suite '%s' passed", result.suite_name)


def validate_raw_to_interim(df: pd.DataFrame) -> None:
    """Validate raw ingestion output against the raw-to-interim suite."""

    _validate(df, "raw_to_interim")


def validate_interim_to_processed(df: pd.DataFrame) -> None:
    """Validate transformed data against the interim-to-processed suite."""

    _validate(df, "interim_to_processed")
    if not df.empty:
        implied = df["implied_rank"].to_numpy(dtype=float)
        normalized = df["normalized_rank"].to_numpy(dtype=float)
        shin = df["shin_rank"].to_numpy(dtype=float)
        if not (np.allclose(implied, normalized) and np.allclose(implied, shin)):
            raise DataQualityError("Rank monotonicity violated for odds conversions.")


__all__ = [
    "DataQualityError",
    "ExpectationResult",
    "validate_raw_to_interim",
    "validate_interim_to_processed",
]
