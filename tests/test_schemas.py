from __future__ import annotations

from datetime import date, datetime, timezone

from ufc_winprob.data.schemas import Bout, BoutStats, ComponentElo, Fighter


def test_fighter_age_computation() -> None:
    fighter = Fighter.create(name="Test Fighter", dob=date(1995, 1, 1))
    assert fighter.age is not None
    assert fighter.fighter_id


def test_bout_stats_default() -> None:
    bout = Bout(
        bout_id="1",
        event_id="evt",
        fighter_id="f1",
        opponent_id="f2",
        winner="f1",
        weight_class="LW",
        scheduled_rounds=3,
        method="decision",
        end_round=3,
        end_time_seconds=900,
        result="win",
        event_date=datetime.now(timezone.utc),
    )
    assert bout.stats.sig_strikes_landed == 0


def test_component_elo_serialization() -> None:
    elo = ComponentElo(
        fighter_id="f1",
        division="LW",
        components=["striking", "grappling"],
        ratings=[1500.0, 1500.0],
        uncertainties=[200.0, 200.0],
        last_updated=datetime.now(timezone.utc),
    )
    assert elo.as_dict()["fighter_id"] == "f1"
