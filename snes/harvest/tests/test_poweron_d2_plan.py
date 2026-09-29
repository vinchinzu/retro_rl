"""Power-on D2 must plan the seed shop and the D2 farm clear.

The power-on D2 farm is all debris, so no pocket ring has bare capacity, and
``--end-of-spring`` turns daily CLEAR_FIELD off.  Both used to drop BUY_SEEDS /
D2_FARM_CLEAR and leave the whole spring grape-only (2026-09-28 soak).
"""

from harvest.planner.day_phase_types import DayPlannerPolicy
from harvest.planner.day_plan_phases import build_day_phases


def _d2_names(**overrides):
    kwargs = dict(
        weekday=2, hour=7, season=0, day=2, has_seeds=False, has_debris=True,
        has_harvest=False, has_waterable=False, has_chickens=False,
        has_cows=False, is_rainy=False, money=300, has_plant_capacity=False,
        is_storm=False, is_festival=False,
        policy=DayPlannerPolicy(include_field_clear=False),
    )
    kwargs.update(overrides)
    return [p.phase for p in build_day_phases(**kwargs)]


def test_power_on_d2_buys_seeds_and_clears_farm():
    names = _d2_names()
    assert "BUY_SEEDS" in names
    assert "D2_FARM_CLEAR" in names


def test_later_days_still_honor_no_field_clear():
    names = _d2_names(day=5, weekday=5, has_plant_capacity=True)
    assert "D2_FARM_CLEAR" not in names
    assert "CLEAR_FIELD" not in names
