"""Reusable plotting utilities for the Streamlit UI."""

from __future__ import annotations

import io
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.metrics import bin_counts


def calibration_plot(probabilities: Iterable[float], outcomes: Iterable[int]) -> io.BytesIO:
    probs = np.asarray(list(probabilities), dtype=float)
    outcomes_arr = np.asarray(list(outcomes), dtype=float)
    bins = np.linspace(0, 1, 11)
    digitized = np.digitize(probs, bins) - 1
    bucket_prob = []
    bucket_outcome = []
    for idx in range(len(bins) - 1):
        mask = digitized == idx
        if not np.any(mask):
            continue
        bucket_prob.append(probs[mask].mean())
        bucket_outcome.append(outcomes_arr[mask].mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.scatter(bucket_prob, bucket_outcome, color="navy", label="Observed")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Win Rate")
    ax.legend()
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    return buffer


__all__ = ["calibration_plot"]
