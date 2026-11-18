"""Herd-size ladder (3, 5, 8 cows): a sick cow is cured, the whole herd is fed, nobody dies overnight.

Research pins, not a route: ``LN_herd_whatif_c{N}_am*`` are cut from the twelve-cow what-if barn by
``python -m harvest.scripts.mint_herd_whatif`` (the recipe is in its docstring). Slot 1 is sick with a
seven-night death timer, the hay is 300 and the clock reads 07:00. The cow section of the day is
exactly what the planner builds: the cure trip (``cow_cure``), then the ordinary chores
(``cow_day_phases``), then the night. The barn chores go through the planner's own one-shot barn reset,
which is why a first ``COW_CHORES retry`` is expected and a stall on the first pass is not a failure.

Measured 2026-11-02, one draw each (07:00 pins), frames from the start of the morning to the end of
the barn chores; "resets" counts `COW_CHORES` passes that ended incomplete and re-entered the barn:

| cows | healthy morning | bottle on the shelf (cure at ~2.1k) | no bottle: buy, cure (cure at ~5.7k) |
| ---: | ---: | ---: | ---: |
| 3 | 3,902 (0 resets) | 8,934 (1) | 14,616 (1) |
| 5 | 23,189 (4) | 18,953 (1) | 32,150 (3) |
| 8 | 22,032 (2) | 26,026 (3) | 27,880 (2) |

``CHORES_FRAME_BUDGET`` is the alarm, not the target: the day floor ends a bad day at 48k frames and
the chores share it with watering, so five to eight cows at 20k-30k leave the fields short. Closing that
gap is the open barn-aisle work (``plan.md``), not these gates; tighten the budget as it closes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harvest.core.animal_status import (
    cow_is_sick,
    existing_cow_slots,
    read_cow_happiness,
    read_cow_timer,
    read_fed_cows_flags,
    unfed_adult_cow_slots,
)
from harvest.core.ram_catalog import read_ram_value
from harvest.planner.cow_cure import cure_day_phases
from harvest.planner.day_phase_catalog import GO_TO_SLEEP_PHASE, RETURN_HOME_PHASE, cow_day_phases
from harvest.planner.day_phase_cow import EXIT_BARN_PHASE
from harvest.planner.day_plan import DayPlanTask
from harvest.tasks.cow_task import ADDR_TOOL_BACKPACK, ADDR_TOOL_SELECTED
from harvest.tasks.nav import make_action
from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

PINS = Path(__file__).resolve().parents[1] / "custom_integrations/HarvestMoon-Snes"
LADDER = (3, 5, 8)
# Alarm line for "barn chores finished", counted from the start of the morning including the cure trip.
# The day floor ends a bad day at 48k; 36k leaves it a third for watering, shipping and the walk home.
CHORES_FRAME_BUDGET = 36_000
MEDICINE = 9
BRUSH, MILKER = 15, 14


def _run_morning(pin: str, *, night: bool = True, max_frames: int = 120_000):
    """Run the planner's cow section for one morning. Returns (env-state facts, task results)."""
    from harvest.runtime.retro_setup import make_harvest_env

    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.herd_ladder"))
    try:
        env.reset()
        ram = env.get_ram()
        cows = len(existing_cow_slots(ram))
        cure = cure_day_phases(ram)
        phases = [EXIT_BARN_PHASE, *cure, *cow_day_phases(ram)]
        if night:
            phases += [RETURN_HOME_PHASE, GO_TO_SLEEP_PHASE]
        task = DayPlanTask(phase_sequence=phases)
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        facts: dict = {"cows": cows, "plan": [p.phase for p in phases], "money_before": read_ram_value(ram, "money")}
        done: set[str] = set()
        for frame in range(max_frames):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
            for row in task.phase_results:
                name = row.get("phase")
                if name in {"COW_CURE", "COW_CHORES"} and row.get("status") == "success" and name not in done:
                    done.add(name)
                    now = env.get_ram()
                    slots = existing_cow_slots(now)
                    facts[name] = {
                        "frame": frame,
                        "fed_flags": read_fed_cows_flags(now),
                        "unfed": unfed_adult_cow_slots(now),
                        "sick": [s for s in slots if cow_is_sick(now, s)],
                        "pair": (int(now[ADDR_TOOL_SELECTED]), int(now[ADDR_TOOL_BACKPACK])),
                        "money": read_ram_value(now, "money"),
                        "reason": str(row.get("reason")),
                    }
        final = env.get_ram()
        slots = existing_cow_slots(final)
        print(
            f"[LADDER] {pin} cows={cows} cure_frame={facts.get('COW_CURE', {}).get('frame')} "
            f"chores_frame={facts.get('COW_CHORES', {}).get('frame')} total_frames={frame} "
            f"resets={sum(1 for r in task.phase_results if r.get('phase') == 'COW_CHORES' and r.get('status') != 'success')}"
        )
        facts.update(
            status=result.status,
            reason=str(result.reason),
            frame=frame,
            slots=slots,
            sick=[s for s in slots if cow_is_sick(final, s)],
            timers=[int(read_cow_timer(final, s)) for s in slots],
            happiness=[int(read_cow_happiness(final, s)) for s in slots],
            writes=env.audit().ram_writes,
            loads=env.audit().mid_run_loads,
            results=list(task.phase_results),
        )
        return facts
    finally:
        env.close()


def _need(pin: str) -> str:
    if not (PINS / f"{pin}.state").exists():
        pytest.skip(f"missing local pin {pin}; mint it with harvest.scripts.mint_herd_whatif")
    return pin


def _assert_clean_audit(facts: dict) -> None:
    assert facts["writes"] == facts["loads"] == 0, (facts["writes"], facts["loads"])
    assert facts["status"] == TaskStatus.SUCCESS, (facts["status"], facts["reason"], facts["results"])


def _assert_herd_fed_and_alive(facts: dict, cows: int) -> None:
    chores = facts["COW_CHORES"]
    assert chores["fed_flags"] == (1 << cows) - 1, chores
    assert chores["unfed"] == [], chores
    assert chores["frame"] <= CHORES_FRAME_BUDGET, chores
    assert len(facts["slots"]) == cows, f"herd {cows} -> {len(facts['slots'])} overnight"
    assert facts["sick"] == [] and facts["timers"] == [0] * cows, (facts["sick"], facts["timers"])


@pytest.mark.rom
@pytest.mark.parametrize("cows", LADDER)
def test_sick_cow_with_a_bottle_on_the_shelf_is_cured_then_the_herd_is_fed(cows):
    facts = _run_morning(_need(f"LN_herd_whatif_c{cows}_am_sick1"))
    _assert_clean_audit(facts)
    assert facts["plan"][:5] == ["EXIT_BARN", "ENSURE_MEDICINE", "NAV_TO_BARN", "ENTER_BARN", "COW_CURE"], facts["plan"]
    cure = facts["COW_CURE"]
    assert cure["sick"] == [], cure
    assert cure["pair"][0] == 0, f"the dose is consumed, the selected slot empties: {cure['pair']}"
    assert MILKER in cure["pair"], cure  # the milker stays; ENSURE_ANIMAL_TOOLS refetches the brush
    # Which slot holds which tool differs run to run; the pair is what matters.
    assert set(facts["COW_CHORES"]["pair"]) == {BRUSH, MILKER}, facts["COW_CHORES"]["pair"]
    _assert_herd_fed_and_alive(facts, cows)
    assert cure["money"] == facts["money_before"], "a bottle already on the shelf costs nothing"


@pytest.mark.rom
@pytest.mark.parametrize("cows", LADDER)
def test_sick_cow_with_no_bottle_buys_one_cures_and_feeds_the_herd(cows):
    facts = _run_morning(_need(f"LN_herd_whatif_c{cows}_am_sick1_buy"))
    _assert_clean_audit(facts)
    assert "BUY_MEDICINE" in facts["plan"] and facts["plan"].index("BUY_MEDICINE") < facts["plan"].index("COW_CURE")
    assert facts["COW_CURE"]["sick"] == []
    assert facts["money_before"] - facts["COW_CURE"]["money"] == 1000, "one 1,000G dose, nothing else bought"
    _assert_herd_fed_and_alive(facts, cows)


@pytest.mark.rom
@pytest.mark.parametrize("cows", LADDER)
def test_healthy_morning_feeds_the_whole_herd_without_a_cure_trip(cows):
    facts = _run_morning(_need(f"LN_herd_whatif_c{cows}_am"))
    _assert_clean_audit(facts)
    assert "COW_CURE" not in facts["plan"] and "BUY_MEDICINE" not in facts["plan"]
    _assert_herd_fed_and_alive(facts, cows)
    assert facts["COW_CHORES"]["money"] == facts["money_before"]


def _cure_only(pin: str, jitter: int):
    """Barn re-entry, then just the cure, after ``jitter`` idle frames. Stops at the cure receipt."""
    from harvest.planner.day_phase_cow import COW_CURE_PHASE, ENTER_BARN_PHASE, NAV_TO_BARN_PHASE
    from harvest.runtime.retro_setup import make_harvest_env

    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.cure_jitter"))
    try:
        env.reset()
        for _ in range(jitter):
            env.step(make_action())
        task = DayPlanTask(phase_sequence=[EXIT_BARN_PHASE, NAV_TO_BARN_PHASE, ENTER_BARN_PHASE, COW_CURE_PHASE])
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(40_000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING or any(
                r.get("phase") == "COW_CURE" and r.get("status") == "success" for r in task.phase_results
            ):
                break
            env.step(result.action.action if result.action else make_action())
        ram = env.get_ram()
        slots = existing_cow_slots(ram)
        return {
            "frame": frame,
            "cows": len(slots),
            "sick": [s for s in slots if cow_is_sick(ram, s)],
            "timers": [int(read_cow_timer(ram, s)) for s in slots],
            "pair": (int(ram[ADDR_TOOL_SELECTED]), int(ram[ADDR_TOOL_BACKPACK])),
            "writes": env.audit().ram_writes,
            "loads": env.audit().mid_run_loads,
            "reason": [str(r.get("reason")) for r in task.phase_results if r.get("phase") == "COW_CURE"],
        }
    finally:
        env.close()


@pytest.mark.rom
@pytest.mark.parametrize("jitter", (0, 31, 97, 257))
@pytest.mark.parametrize("cows", LADDER)
def test_cure_lands_under_perturbed_timing(cows, jitter):
    """Idle frames before the walk move the cows and the animation phase; the dose must still land.

    The 8-cow barn once let the sick bit fall after a 20-frame poll and the retry skipped a cow it had
    already cured; this is the regression net for that class of race.
    """
    facts = _cure_only(_need(f"LN_herd_whatif_c{cows}_sick1"), jitter)
    assert facts["writes"] == facts["loads"] == 0
    assert facts["cows"] == cows
    assert facts["sick"] == [] and facts["timers"] == [0] * cows, facts
    assert facts["pair"][0] == 0, f"dose not consumed: {facts['pair']}"
    assert not any("skipped" in text for text in facts["reason"]), facts["reason"]
