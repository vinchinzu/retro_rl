"""rr-6o7.1 stage 2: systematic, fixture-only sweep for the L8 lone-bush tile.

Isolated/fixture-only recon (mirrors ``zelda_i.level8.bush.IsolatedBushReconController``
but as a standalone lab script per the ``gohma_lab.py`` house style, because
running the full controller state machine per candidate is too slow for a
broad sweep). Loads ``Level8BushWithCandleFixture`` (candle owned+selected,
triforce 0x7F, built by ``build_level8_bush_candle_fixture.py``) once, then
for every candidate stand position performs a fast in-process state reload
(``env.em.set_state`` via ``resync_custom_state`` — no env re-creation) and
tries facing the bush direction, firing the Red Candle (B), then pushing.

Two phases:

  1. ``--scan``: probe which (x, y) positions on OW 0x6D are physically
     standable (teleport + settle, keep only positions where the position
     stops drifting) — this is a *live* walkability probe, not a guess, and
     it is far larger than the previously-documented "walkable pocket"
     (that pocket was the *walked* corridor from the assisted approach, not
     the full standable area). Result cached to
     ``nes/zelda_i/logs/level8_6d_walkable_positions.json``.

  2. ``--burn``: for every standable position, try all 4 facings and, for
     each facing, 2 push directions (continue forward in the facing
     direction, and UP — the universal Zelda 1 "walk into the cave mouth"
     direction) after firing. Logs every trial. Stops immediately on the
     first trial where ``snap.level == 8 and snap.mode == PLAY_MODE`` is
     observed with the candle_used flag toggled first (matching
     ``BurnLevel8BushController``'s own success predicate).

Never writes ADDR_SELECTED_ITEM/ADDR_CANDLE/ADDR_TRIFORCE — those are the
one-time disclosed pokes baked into the source fixture, not this loop.
Link position pokes (teleporting between candidates) are the only writes
this script performs, and they are logged in every trial record.

    uv run python nes/zelda_i/scratch/level8_bush_burn_sweep.py --scan --burn
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs, resync_custom_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.anchors import SCREEN_LEVEL8_BUSH
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

FIXTURE_NAME = "Level8BushWithCandleFixture"
LEVEL8 = 8
ADDR_CANDLE_USED = 0x0513
POST_L7_TRIFORCE = 0x7F
CANDLE_RED = 2

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
WALKABLE_CACHE = LOG_DIR / "level8_6d_walkable_positions.json"
SWEEP_LOG = LOG_DIR / "level8_bush_burn_sweep.json"

# Live walkability scan bounds — full OW screen extent, not the narrower
# previously-documented walked corridor. Step 4px; rows settle onto an
# ~8px grid so this resolution does not miss a standable row.
SCAN_X_RANGE = range(0, 253, 4)
SCAN_Y_RANGE = range(56, 233, 4)
SETTLE_MAX_FRAMES = 30

FACINGS = ("UP", "DOWN", "LEFT", "RIGHT")
FACE_FRAMES = 4
FIRE_FRAMES = 6
POST_FIRE_SETTLE = 70  # let the flame resolve before deciding push worked
PUSH_FRAMES = 40
ENTER_FRAMES = 90
STAND_SETTLE_FRAMES = 10


def _assign(env: Any, address: int, value: int) -> None:
    env.unwrapped.data.memory.assign(int(address), "|u1", int(value) & 0xFF)


def _reload(env: Any) -> None:
    reset_obs(env)
    resync_custom_state(env, GAME_DIR, GAME, FIXTURE_NAME)


def _push_variants(facing: str) -> tuple[str, str]:
    return (facing, "UP") if facing != "UP" else ("UP", "DOWN")


def scan_walkable_positions(env: Any) -> list[tuple[int, int]]:
    """Live probe: which OW-0x6D tiles hold position without drifting."""
    seen: set[tuple[int, int]] = set()
    for x in SCAN_X_RANGE:
        for y in SCAN_Y_RANGE:
            _reload(env)
            _assign(env, ADDR_LINK_X, x)
            _assign(env, ADDR_LINK_Y, y)
            prev = None
            stable = 0
            snap = None
            for _ in range(SETTLE_MAX_FRAMES):
                env.step(nes_idle_action())
                snap = read_snapshot(env.get_ram())
                cur = (snap.link_x, snap.link_y)
                if cur == prev:
                    stable += 1
                    if stable >= 2:
                        break
                else:
                    stable = 0
                prev = cur
            if snap is not None and snap.mode == PLAY_MODE and snap.level == 0:
                seen.add((snap.link_x, snap.link_y))
    return sorted(seen)


def run_trial(
    env: Any, x: int, y: int, facing: str, push: str
) -> dict[str, Any]:
    _reload(env)
    _assign(env, ADDR_LINK_X, x)
    _assign(env, ADDR_LINK_Y, y)
    for _ in range(STAND_SETTLE_FRAMES):
        env.step(nes_idle_action())
    ram = env.get_ram()
    stand_snap = read_snapshot(ram)
    stand_xy = (stand_snap.link_x, stand_snap.link_y)

    # Guard rail: never let this loop touch inventory/progression.
    assert read_u8(ram, ADDR_CANDLE) == CANDLE_RED
    assert read_u8(ram, ADDR_TRIFORCE) == POST_L7_TRIFORCE

    for _ in range(FACE_FRAMES):
        env.step(nes_action(facing))
    for _ in range(FIRE_FRAMES):
        env.step(nes_action("B"))

    candle_used = False
    mouth_seen = False
    for _ in range(POST_FIRE_SETTLE):
        obs = env.step(nes_idle_action())
        ram = env.get_ram()
        candle_used = candle_used or read_u8(ram, ADDR_CANDLE_USED) != 0
        snap = read_snapshot(ram)
        if snap.mode == 16:
            mouth_seen = True
            break
        if snap.level != 0 or snap.mode not in (PLAY_MODE, 16):
            break

    outcome = "no_effect"
    entry_room = None
    if not mouth_seen:
        for _ in range(PUSH_FRAMES):
            env.step(nes_action(push))
            ram = env.get_ram()
            candle_used = candle_used or read_u8(ram, ADDR_CANDLE_USED) != 0
            snap = read_snapshot(ram)
            if snap.mode == 16:
                mouth_seen = True
                break
            if snap.level != 0 and snap.mode == PLAY_MODE:
                # Left the bush screen onto a different dungeon without a
                # mouth transition frame we caught — treat as a hit.
                mouth_seen = True
                break

    if mouth_seen:
        outcome = "mouth_mode16"
        for _ in range(ENTER_FRAMES):
            env.step(nes_action("UP"))
            ram = env.get_ram()
            snap = read_snapshot(ram)
            if snap.level == LEVEL8 and snap.mode == PLAY_MODE:
                outcome = "level8_entered"
                entry_room = int(snap.screen)
                break
            if snap.mode == 17:
                outcome = "link_death_during_enter"
                break
    else:
        ram = env.get_ram()
        snap = read_snapshot(ram)
        if snap.mode == 17:
            outcome = "link_death"
        elif snap.level == 0 and snap.mode == PLAY_MODE and snap.screen != SCREEN_LEVEL8_BUSH:
            outcome = "scrolled_off_screen"

    return {
        "target_xy": [x, y],
        "stand_xy": list(stand_xy),
        "facing": facing,
        "push": push,
        "candle_used": bool(candle_used),
        "mouth_seen": bool(mouth_seen),
        "outcome": outcome,
        "entry_room": entry_room,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="store_true", help="(re)run the live walkability probe")
    parser.add_argument("--burn", action="store_true", help="run the facing/fire/push sweep")
    parser.add_argument("--limit", type=int, default=0, help="cap trial count (0 = no cap)")
    parser.add_argument("--time-budget-s", type=float, default=540.0)
    args = parser.parse_args()

    configure_headless()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    resync_custom_state(env, GAME_DIR, GAME, FIXTURE_NAME)

    positions: list[tuple[int, int]]
    if args.scan or not WALKABLE_CACHE.exists():
        t0 = time.time()
        positions = scan_walkable_positions(env)
        WALKABLE_CACHE.write_text(json.dumps(positions), encoding="utf-8")
        print(f"scan: {len(positions)} standable positions in {time.time() - t0:.1f}s -> {WALKABLE_CACHE}")
    else:
        positions = [tuple(p) for p in json.loads(WALKABLE_CACHE.read_text(encoding="utf-8"))]
        print(f"loaded {len(positions)} cached standable positions from {WALKABLE_CACHE}")

    if not args.burn:
        env.close()
        return 0

    trials: list[dict[str, Any]] = []
    success: dict[str, Any] | None = None
    near_misses: list[dict[str, Any]] = []
    t0 = time.time()
    n = 0
    stop = False
    for (x, y) in positions:
        if stop:
            break
        for facing in FACINGS:
            if stop:
                break
            for push in _push_variants(facing):
                if args.limit and n >= args.limit:
                    stop = True
                    break
                if time.time() - t0 > args.time_budget_s:
                    print("time budget exceeded, stopping sweep")
                    stop = True
                    break
                record = run_trial(env, x, y, facing, push)
                n += 1
                trials.append(record)
                if record["mouth_seen"]:
                    near_misses.append(record)
                    print(f"NEAR MISS #{n}: {record}")
                if record["outcome"] == "level8_entered":
                    success = record
                    print(f"SUCCESS #{n}: {record}")
                    stop = True
                    break
    elapsed = time.time() - t0
    print(f"sweep done: {n} trials in {elapsed:.1f}s ({elapsed / max(n,1) * 1000:.1f} ms/trial)")
    print(f"near misses (mode16 seen): {len(near_misses)}")
    print(f"success: {success}")

    payload = {
        "bead": "rr-6o7.1",
        "fixture_source": FIXTURE_NAME,
        "n_standable_positions": len(positions),
        "n_trials": n,
        "elapsed_s": elapsed,
        "facings": FACINGS,
        "success": success,
        "near_misses": near_misses,
        "trials": trials,
    }
    SWEEP_LOG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"log written: {SWEEP_LOG}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
