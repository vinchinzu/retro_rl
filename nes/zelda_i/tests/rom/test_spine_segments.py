"""Power-on Survival spine segments: boot the real ROM, play, assert live RAM.

Two ``run_survival_spine`` targets picked deliberately, not from the
``pre-l1``/``level1-bombs``/``level2-entry`` example list in this tier's
brief:

- ``pre-l1`` was measured red on this exact tree during this session
  (``failed_stage=bomb_walk``) -- ``nes/zelda_i/overworld/{hunt,path,topup,
  common}.py`` and ``combat.py`` are uncommitted and in flight from other
  lanes touching the same files, so any ``--through`` target that walks
  through ``overworld.gathering.pre_l1_stages`` (that includes
  ``level1-bow``/``level1-bombs``/``level1-arrows``) is unstable right now.
  A red test there would be the route, not this test, but it would burn a
  sitting establishing that -- so this file avoids the whole dedicated-hop
  chain and picks targets proven green this session instead.
- ``through="level1"`` and ``through="level2-entry"`` do NOT go through
  ``pre_l1_stages``: ``_continue_level1_spine`` takes the natural-Triforce
  branch (``level1_survival_tf_stages``) when ``through`` isn't one of the
  L1 dedicated-gathering hops, and L2 entry only adds
  ``OverworldToLevel2Controller`` (``zelda_i.level2.overworld``, not the
  volatile ``zelda_i.overworld.*`` package) on top of that. Both were run
  live this session (see report below) and passed.

This does not touch or overwrite the Clean M5 result (``run_level1_complete
.py``): it calls the Survival spine library function directly, with the
Survival health-refill assist on, exactly like
``uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video``
already does in ``nes/zelda_i/AGENTS.md``. It is a development-checkpoint
assertion, not a Clean-gate one.

    uv run pytest nes/zelda_i/tests/rom/test_spine_segments.py -m rom -q

Wall time: ~45-55s per test (a full power-on boot + Level 1, or + L2 entry).
"""

from __future__ import annotations

import pytest

from retro_harness.env import make_env, reset_obs
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR, SHARED_ROM_ZIP
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine

from .conftest import ROM

pytestmark = [
    ROM,
    pytest.mark.skipif(not SHARED_ROM_ZIP.is_file(), reason="Zelda I ROM zip missing"),
]


def _run(through: str):
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        assist = UnlimitedHealthAssist(enabled=True)
        run = run_survival_spine(
            env, obs, assist=assist, through=through, allow_pokes=True, gather=False
        )
        snap = read_snapshot(env.get_ram())
        return run, snap
    finally:
        env.close()


def test_live_power_on_through_level1_triforce() -> None:
    """Power-on -> sword -> Level 1 Triforce. Measured this session: 22335f."""
    run, snap = _run("level1")
    assert run.success, run.report()
    assert run.failed_stage is None
    assert int(snap.level) == 1
    assert int(snap.mode) == 18  # fanfare/complete mode, not PLAY_MODE
    assert int(snap.triforce) & 0x01
    assert int(snap.screen) == 0x36
    assert int(snap.bow) == 1
    # A stall ceiling, not a band: a faster route is not a failure.
    # Measured 16882f (2026-09-24, lattice walkers).
    assert run.end_frame <= 20000


def test_live_power_on_through_level2_entry() -> None:
    """Power-on -> Level 1 TF -> Moon door -> L2 entry 0x7d. Measured 21736f."""
    run, snap = _run("level2-entry")
    assert run.success, run.report()
    assert run.failed_stage is None
    assert int(snap.level) == 2
    assert int(snap.mode) == 5  # PLAY_MODE: settled in the entry room
    assert int(snap.triforce) & 0x01
    assert int(snap.screen) == 0x7D
    assert (int(snap.link_x), int(snap.link_y)) == (120, 205)
    assert int(snap.bow) == 1
    assert run.end_frame <= 26000


def test_live_power_on_gathered_level1_triforce_no_l1_assist() -> None:
    """Default gathered spine -> L1 TF, L1 health assist off.

    Measured 2026-09-24: 74227f with the hidden-rupee detours, the Blue
    Ring and a red potion. The wallet is never written: the only count
    write left before L1's Triforce is keys=1 at backtrack44.
    """
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        run = run_survival_spine(env, obs, assist=None, through="level1")
        snap = read_snapshot(env.get_ram())
    finally:
        env.close()
    assert run.success, run.report()
    assert int(snap.triforce) & 0x01
    assert int(snap.sword) == 2
    assert int(snap.heart_containers) == 7  # 6 gathered + Aquamentus
    assert run.report()["gather"]["engage_hearts"] == 1
    assert int(snap.ring) == 1
    writes = (run.report().get("inventory_assist") or {}).get("writes") or []
    assert not [w for w in writes if w.get("field") == "rupees"], writes
    assert run.end_frame <= 82000
