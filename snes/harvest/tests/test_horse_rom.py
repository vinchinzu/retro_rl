"""Saddle shipping on the real ROM (research pins ``LN_saddle_whatif_*``, local only).

``LN_saddle_whatif_carry_y2``: Y2 Summer D20 12:xx, adult horse at its home tile (17,20), the farmer
just picked a corn at (328,425). ``LN_saddle_whatif_harvest_y2d22``: Y2 Summer D22 12:11, 48 ripe
crops, the farmer far from both the bin and the horse. Both are minted from the sweep pins by
running the planner to that moment; nothing is written to RAM.
"""

from __future__ import annotations

import pytest

from harvest.paths import GAME_DIR

pytestmark = pytest.mark.rom

CARRY_PIN = "LN_saddle_whatif_carry_y2"
HARVEST_PIN = "LN_saddle_whatif_harvest_y2d22"


def _needs(pin: str) -> None:
    if not (GAME_DIR / f"{pin}.state").is_file():
        pytest.skip(f"research pin {pin} is local-only and not present")


def _harvest(pin: str, frames: int, monkeypatch, *, saddle: bool, far_only: bool = False):
    from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState
    from harvest.core.ram_catalog import read_ram_value
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.harvest_task import HarvestTask
    from harvest.tasks.nav import make_action

    monkeypatch.setenv("HARVEST_SADDLE", "1" if saddle else "0")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.saddle"))
    try:
        env.reset()
        ram = env.get_ram()
        before = {
            "ship": int(read_ram_value(ram, "shipping_money", raw=True)),
            "corn": int(read_ram_value(ram, "shipped_corn", raw=True)),
        }
        task = HarvestTask()
        task.reset(WorldState(frame=0, ram=ram, info={}, obs=None))
        if far_only:  # the near rings are done: what the late-park policy waits for
            from harvest.tasks import horse as saddle_mod
            task._steps = [st for st in task._steps
                           if abs(st.stand[0] - task.ship_stand[0]) + abs(st.stand[1] - task.ship_stand[1]) >= saddle_mod.PARK_FAR_FROM_BIN]
        status = TaskStatus.RUNNING
        used = 0
        for used in range(frames):
            result = task.step(WorldState(frame=used, ram=env.get_ram(), info={}, obs=None))
            status = result.status
            if status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action is not None else make_action())
        after = env.get_ram()
        out = {
            "task": task,
            "status": status,
            "frames": used,
            "ship": (int(read_ram_value(after, "shipping_money", raw=True)) - before["ship"]) * 10,
            "corn": int(read_ram_value(after, "shipped_corn", raw=True)) - before["corn"],
            "riding": bool(int(read_ram_value(after, "player_state", raw=True)) & 0x10),
            "audit": env.audit(),
        }
        return out
    finally:
        env.close()


def test_a_carried_corn_ships_at_the_horse_and_pays_the_bin_price(monkeypatch):
    _needs(CARRY_PIN)
    run = _harvest(CARRY_PIN, 4000, monkeypatch, saddle=True)
    task = run["task"]
    assert run["corn"] >= 1 and run["ship"] >= 120, (run["corn"], run["ship"], task.progress_text)
    assert task.horse_shipped >= 1, "the bin was nearer than the horse stand; pin changed?"
    assert not run["riding"]
    assert run["audit"].ram_writes == run["audit"].mid_run_loads == 0


def test_a_parked_horse_ships_far_ripe_crops_where_the_bin_walk_cannot(monkeypatch):
    _needs(HARVEST_PIN)
    saddle = _harvest(HARVEST_PIN, 4200, monkeypatch, saddle=True, far_only=True)
    bin_only = _harvest(HARVEST_PIN, 4200, monkeypatch, saddle=False, far_only=True)
    task = saddle["task"]
    assert task.park_outcome == "done", task.park_outcome
    assert not saddle["riding"]
    assert task.horse_shipped >= 10, task.progress_text
    # Same pin, same far crops: the bin walk is too far for the window; the horse ships them.
    assert saddle["corn"] >= bin_only["corn"] + 6, (saddle["corn"], bin_only["corn"])
    assert saddle["ship"] >= bin_only["ship"] + 6 * 120
    assert saddle["audit"].ram_writes == saddle["audit"].mid_run_loads == 0
