"""Labor morning on the real ROM: pin ``PR_e2ev1n_Y2_Winter_D24`` (local only, e2e run v1n)."""

from __future__ import annotations

import pytest

from harvest.paths import GAME_DIR

pytestmark = pytest.mark.rom
PIN = "PR_e2ev1n_Y2_Winter_D24"


def test_the_wife_clears_the_doorway_and_the_farmer_walks_out():
    if not (GAME_DIR / f"{PIN}.state").is_file():
        pytest.skip(f"pin {PIN} is local-only and not present")
    from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.tasks.inventory_exit import ExitToFarmTask
    from harvest.planner.tasks.labor import labor_morning_due
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import make_action

    env = AuditedEnv(make_harvest_env(PIN, lean_step=True), capabilities=AuditCapabilities.all("harvest.labor"))
    try:
        env.reset()
        for _ in range(60):
            env.step(make_action())
        assert labor_morning_due(env.get_ram()), "the pin should be the crowded morning"
        world = lambda f: WorldState(frame=f, ram=env.get_ram(), info={}, obs=None)
        from harvest.planner.tasks import labor as labor_mod

        steps = []
        original = labor_mod.LaborMorningTask.step
        labor_mod.LaborMorningTask.step = lambda self, w: (steps.append(1), original(self, w))[1]
        exit_task = ExitToFarmTask()
        exit_task.reset(world(0))
        for used in range(12000):
            result = exit_task.step(world(used))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action is not None else make_action())
        assert result.status == TaskStatus.SUCCESS, (result.status, result.reason)
        labor_mod.LaborMorningTask.step = original
        assert len(steps) > 1000, "the labor script should have run first"
        assert int(read_ram_value(env.get_ram(), "tilemap", raw=True)) == 0
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()
