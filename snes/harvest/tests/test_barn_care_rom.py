"""Bounded barn replays; no campaign or overnight simulation required."""
from pathlib import Path

import pytest

from harvest.core.animal_status import cows_outside, existing_cow_slots, unfed_adult_cow_slots
from harvest.planner.day_phase_cow import COW_CHORES_PHASE, NAV_TO_BARN_PHASE, ENTER_BARN_PHASE
from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE, ENSURE_ANIMAL_TOOLS_PHASE
from harvest.planner.day_plan import DayPlanTask
from harvest.tasks.cow_task import CowChoresTask
from harvest.tasks.nav import make_action
from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

PINS = Path(__file__).resolve().parents[1] / "custom_integrations/HarvestMoon-Snes"


@pytest.mark.rom
@pytest.mark.parametrize("pin,jitter", [
    (pin, jitter)
    for pin in ("LN_care_barn_stall", "PR_e2eh6_Y2_Spring_D2")
    for jitter in (0, 31, 97, 257)
]+ [("LN_herd_whatif_c12_am", 0)])
def test_barn_care_finishes_every_requested_action(pin, jitter):
    from harvest.runtime.retro_setup import make_harvest_env

    if not (PINS / f"{pin}.state").exists():
        pytest.skip(f"missing local pin {pin}")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.barn_care"))
    try:
        env.reset()
        starting_slots = set(existing_cow_slots(env.get_ram()))
        for _ in range(jitter):
            env.step(make_action())
        phases = ([COW_CHORES_PHASE] if "Y2_Spring" not in pin else
                  [EXIT_TO_FARM_PHASE, ENSURE_ANIMAL_TOOLS_PHASE, NAV_TO_BARN_PHASE, ENTER_BARN_PHASE, COW_CHORES_PHASE])
        task = DayPlanTask(phase_sequence=phases)
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(45000):
            result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        ram = env.get_ram()
        receipt = CowChoresTask().care_receipt(ram)
        assert result.status == TaskStatus.SUCCESS, (frame, result.reason, task.phase_results, receipt)
        assert set(receipt) == starting_slots, (starting_slots, receipt, cows_outside(ram))
        assert not cows_outside(ram)
        assert receipt and all(row["complete"] for row in receipt.values()), receipt
        assert not unfed_adult_cow_slots(ram)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()
