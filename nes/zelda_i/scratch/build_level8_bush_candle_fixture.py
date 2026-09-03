"""Build a fixture-only "ready to search" checkpoint for the L8 bush recon.

rr-6o7.1 stage 1. Starts from ``Level8BushOW.state`` (fixture-live, no
candle, no PostLevel7Handoff) and applies a fully-disclosed one-time poke:
Red Candle owned + selected, post-L7 triforce byte, and a safe/central stand
position inside the documented walkable pocket on OW 0x6D. Saves the result
as ``Level8BushWithCandleFixture`` with a provenance sidecar in the
``recon_fixture`` schema (``nes/zelda_i/level9/stair_session.py`` house
style): every poke disclosed, ``route_eligible: false``, ``fixture_only:
true``, ``natural_entry: false``.

This is the ONLY place that pokes candle/selected-item/triforce for this
bead. The search loop itself (``IsolatedBushReconController`` or the sweep
script) must never write inventory/progression.

    uv run python nes/zelda_i/scratch/build_level8_bush_candle_fixture.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    read_snapshot,
)

BEAD = "rr-6o7.1"
SOURCE_STATE = "Level8BushOW"
FIXTURE_NAME = "Level8BushWithCandleFixture"

CANDLE_RED = 2
B_ITEM_CANDLE = 4
POST_L7_TRIFORCE = 0x7F
# Safe/central stand inside the documented walkable pocket on 0x6D:
# WALKABLE_LEFT_X=(32,56) intersected with WALKABLE_SAND_Y=(88,96).
STAND_X = 48
STAND_Y = 90

SETTLE_FRAMES = 30

# Every disclosed poke, in application order. Position pokes are here (not
# progression) so the search loop starts from a known, documented-walkable
# tile rather than wherever Level8BushOW.state happened to leave Link.
FIXTURE_WRITES: list[dict[str, Any]] = [
    {"name": "red_candle", "address": ADDR_CANDLE, "address_hex": f"0x{ADDR_CANDLE:04X}", "value": CANDLE_RED},
    {"name": "selected_item_candle", "address": ADDR_SELECTED_ITEM, "address_hex": f"0x{ADDR_SELECTED_ITEM:04X}", "value": B_ITEM_CANDLE},
    {"name": "triforce_post_l7", "address": ADDR_TRIFORCE, "address_hex": f"0x{ADDR_TRIFORCE:04X}", "value": POST_L7_TRIFORCE},
    {"name": "stand_x_safe_central_0x6d", "address": ADDR_LINK_X, "address_hex": f"0x{ADDR_LINK_X:04X}", "value": STAND_X},
    {"name": "stand_y_safe_central_0x6d", "address": ADDR_LINK_Y, "address_hex": f"0x{ADDR_LINK_Y:04X}", "value": STAND_Y},
]


def _assign(env: Any, address: int, value: int) -> None:
    env.unwrapped.data.memory.assign(int(address), "|u1", int(value) & 0xFF)


def main() -> int:
    configure_headless()
    env = make_env(GAME, SOURCE_STATE, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)

    before = read_snapshot(env.get_ram())
    print(f"source ({SOURCE_STATE}) glance: {compact_snapshot(before)}")

    for write in FIXTURE_WRITES:
        _assign(env, write["address"], write["value"])

    for _ in range(SETTLE_FRAMES):
        obs, *_ = env.step(nes_idle_action())

    ram = env.get_ram()
    after = read_snapshot(ram)
    print(f"fixture glance: {compact_snapshot(after)}")

    assert int(ram[ADDR_CANDLE]) == CANDLE_RED, "candle poke did not take"
    assert int(ram[ADDR_SELECTED_ITEM]) == B_ITEM_CANDLE, "selected_item poke did not take"
    assert int(ram[ADDR_TRIFORCE]) == POST_L7_TRIFORCE, "triforce poke did not take"

    path = save_state(env, GAME_DIR, GAME, FIXTURE_NAME)
    source_path = state_path(GAME_DIR, GAME, SOURCE_STATE)
    result = {
        "ok": True,
        "candle": int(ram[ADDR_CANDLE]),
        "selected_item": int(ram[ADDR_SELECTED_ITEM]),
        "triforce": hex(int(ram[ADDR_TRIFORCE])),
        "state": compact_snapshot(after),
    }
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": BEAD,
            "phase": "level8_bush_ow_candle_ready",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "fixture_writes": FIXTURE_WRITES,
            "notes": [
                "Stage-1 fixture for rr-6o7.1: candle+selected+triforce+stand",
                "position poked directly, not walked (documented-walkable tile)",
                "search loop must never write ADDR_SELECTED_ITEM/inventory itself",
            ],
        },
        selected_trial=result,
        natural_entry=False,
    )
    print(f"saved {path}")
    print(f"saved {path.with_suffix('.provenance.json')}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
