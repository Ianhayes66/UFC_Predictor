"""Feature assembly for model training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..data.schemas import Bout, ComponentElo, Fighter, OddsSnapshot
from ..utils.time_utils import days_between
from .age_curve import age_adjustment


@dataclass
class FeatureMatrix:
    features: pd.DataFrame
    target: pd.Series


def _elo_to_series(elo: ComponentElo, prefix: str) -> pd.Series:
    data = {f"{prefix}_{component}": rating for component, rating in zip(elo.components, elo.ratings)}
    return pd.Series(data)


def build_features(
    bouts: Sequence[Bout],
    fighters: Dict[str, Fighter],
    elos: Dict[str, ComponentElo],
    odds: Dict[str, List[OddsSnapshot]] | None = None,
) -> FeatureMatrix:
    records: List[pd.Series] = []
    targets: List[int] = []
    for bout in bouts:
        fighter = fighters.get(bout.fighter_id)
        opponent = fighters.get(bout.opponent_id)
        if not fighter or not opponent:
            continue
        elo_a = elos.get(bout.fighter_id)
        elo_b = elos.get(bout.opponent_id)
        if not elo_a or not elo_b:
            continue
        series = pd.Series({
            "bout_id": bout.bout_id,
            "event_id": bout.event_id,
            "division": bout.weight_class,
            "scheduled_rounds": bout.scheduled_rounds,
            "age_a": fighter.age or 30,
            "age_b": opponent.age or 30,
            "age_diff": (fighter.age or 30) - (opponent.age or 30),
            "age_effect_a": age_adjustment(fighter.age or 30, bout.weight_class),
            "age_effect_b": age_adjustment(opponent.age or 30, bout.weight_class),
            "activity_gap": days_between(
                (fighter.dob or opponent.dob or bout.event_date.date()).isoformat(),
                bout.event_date,
            ),
            "is_main_event": int(bout.scheduled_rounds == 5),
        })
        series = pd.concat([series, _elo_to_series(elo_a, "fighter"), _elo_to_series(elo_b, "opponent")])
        if odds and bout.bout_id in odds:
            market_prob = np.median([snap.normalized_probability for snap in odds[bout.bout_id]])
            series["market_prob"] = market_prob
        else:
            series["market_prob"] = np.nan
        records.append(series)
        targets.append(1 if bout.winner == bout.fighter_id else 0)
    frame = pd.DataFrame(records)
    feature_cols = [col for col in frame.columns if col not in {"bout_id", "event_id", "division"}]
    if "market_prob" in frame.columns:
        frame["market_prob"] = frame["market_prob"].fillna(0.5)
    frame = frame.fillna(0.0)
    target_series = pd.Series(targets, index=frame.index, name="target")
    return FeatureMatrix(features=frame[feature_cols], target=target_series)


def synthetic_dataset(n: int = 200, seed: int = 42) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    components = ["striking", "grappling", "wrestling", "submissions", "cardio", "durability", "iq", "aggression"]
    fighters: Dict[str, Fighter] = {}
    elos: Dict[str, ComponentElo] = {}
    bouts: List[Bout] = []
    for idx in range(n):
        fighter_id = f"f-{idx}"
        opponent_id = f"f-{(idx+1)%n}"
        age_a = 24 + rng.integers(0, 12)
        age_b = 24 + rng.integers(0, 12)
        fighters[fighter_id] = Fighter.create(name=f"Fighter {idx}", division="LW")
        fighters[opponent_id] = fighters.get(opponent_id, Fighter.create(name=f"Fighter {(idx+1)%n}", division="LW"))
        ratings_a = (1500 + rng.normal(0, 50, size=len(components))).tolist()
        ratings_b = (1500 + rng.normal(0, 50, size=len(components))).tolist()
        elos[fighter_id] = ComponentElo(
            fighter_id=fighter_id,
            division="LW",
            components=components,
            ratings=ratings_a,
            uncertainties=[200.0] * len(components),
            last_updated=pd.Timestamp("2023-01-01", tz="UTC"),
        )
        elos[opponent_id] = ComponentElo(
            fighter_id=opponent_id,
            division="LW",
            components=components,
            ratings=ratings_b,
            uncertainties=[200.0] * len(components),
            last_updated=pd.Timestamp("2023-01-01", tz="UTC"),
        )
        bouts.append(
            Bout(
                bout_id=f"b-{idx}",
                event_id="evt-1",
                fighter_id=fighter_id,
                opponent_id=opponent_id,
                winner=fighter_id if rng.random() > 0.5 else opponent_id,
                weight_class="LW",
                scheduled_rounds=3,
                method="decision",
                end_round=3,
                end_time_seconds=900,
                result="win",
                event_date=pd.Timestamp("2023-06-01", tz="UTC"),
            )
        )
    matrix = build_features(bouts, fighters, elos, odds=None)
    return matrix


__all__ = ["FeatureMatrix", "build_features", "synthetic_dataset"]
