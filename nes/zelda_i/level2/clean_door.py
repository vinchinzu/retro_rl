"""Clean heart-safe door path: 0x4A farm → 0x5A clear → 0x3C.

Isolated 2/2 from ``At4A`` (2026-08-06): farm to ≥3 filled, rejoin west/south
to 0x59 (stop early on play before hop-advance settles south), east @y≈140
into 0x5A, corridor kite 201f, then ``LEVEL2_CLEAN_FROM_5A_TO_3C`` with maze.

No health RAM writes. Assist must stay off.

The live runner ``run_clean_door_from_env`` is the authoritative frame sequence
(timing-sensitive). Keep step-controllers for unit tests / composition only.
"""

from __future__ import annotations

from typing import Any

from zelda_i.overworld.heart_farm import HeartFarmController
from zelda_i.level2.overworld import (
    LEVEL2_CLEAN_FROM_5A_TO_3C,
    OverworldToLevel2Controller,
)
from zelda_i.overworld.common import swing_action
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import PLAY_MODE, read_snapshot

REJOIN_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x49, "LEFT", align_y=141),
    ScreenHop(0x59, "DOWN", align_x=112),
)
EAST_Y = 140
DEFAULT_CLEAR_FRAMES = 201


def _kite_clear(env, frames: int, *, max_x: int = 160) -> list[str]:
    """Exact clear loop used by the 2/2 Clean probe (frame-sensitive)."""
    notes: list[str] = []
    start = -1
    for f in range(frames):
        snap = read_snapshot(env.get_ram())
        if start < 0:
            start = snap.filled_hearts
        if snap.mode == 17:
            notes.append("clear_death")
            return notes
        enemies = [
            o
            for o in snap.objects
            if o.slot >= 1 and o.type_id in (3, 7, 13) and 40 < o.y < 200
        ]
        near = [
            o
            for o in enemies
            if abs(o.x - snap.link_x) + abs(o.y - snap.link_y) < 40
        ]
        if near:
            n = near[0]
            dx, dy = n.x - snap.link_x, n.y - snap.link_y
            if abs(dx) > abs(dy):
                d = "LEFT" if dx > 0 else "RIGHT"
            else:
                d = "UP" if dy > 0 else "DOWN"
            env.step(swing_action(f, d, "k", period=5, hold=3).action)
        elif enemies:
            n = min(
                enemies,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dx, dy = n.x - snap.link_x, n.y - snap.link_y
            if abs(dx) >= abs(dy):
                d = "RIGHT" if dx > 0 else "LEFT"
            else:
                d = "DOWN" if dy > 0 else "UP"
            if d == "RIGHT" and snap.link_x > max_x:
                d = "LEFT"
            env.step(swing_action(f, d, "c", period=6, hold=3).action)
        else:
            d = "UP" if snap.link_y > 150 else "DOWN"
            env.step(swing_action(f, d, "i", period=12, hold=2).action)
    end = read_snapshot(env.get_ram()).filled_hearts
    notes.append(f"clear_{start}_to_{end}")
    return notes


def run_clean_door_from_env(
    env,
    obs: Any,
    *,
    farm_hearts_min: int = 3,
    farm_max_frames: int = 2500,
    corridor_clear_frames: int = DEFAULT_CLEAR_FRAMES,
    max_door_frames: int = 15000,
    trail: list[dict] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run Clean farm+clear+door path on an already-reset env at/near 0x4A.

    Returns ``(obs, report)`` with ``report["ok"]`` true on 0x3C play hearts>0.
    """
    notes: list[str] = []
    if trail is None:
        trail = []

    def _log(stage: str) -> None:
        s = read_snapshot(env.get_ram())
        trail.append(
            {
                "stage": stage,
                "screen": s.screen,
                "hearts": s.filled_hearts,
                "mode": s.mode,
                "x": s.link_x,
                "y": s.link_y,
                "health": s.health,
            }
        )

    # --- Farm on 0x4A ---
    farm = HeartFarmController(
        min_filled=farm_hearts_min,
        max_frames=farm_max_frames,
        farm_screen=0x4A,
    )
    for _ in range(farm_max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return obs, _fail_report(trail, notes, farm, "farm_death", env=env)
        if snap.screen != 0x4A:
            break
        if snap.filled_hearts >= farm_hearts_min:
            # Mark farm done so report peak/success stay honest.
            farm.step(snap)
            break
        act = farm.step(snap)
        obs, *_ = env.step(act.action)
        if farm.success:
            break
    snap = read_snapshot(env.get_ram())
    notes.append(f"farm_h{snap.filled_hearts}")
    _log("post_farm")

    # --- Rejoin to 0x59 (early stop on play y<200) ---
    nav = OverworldToLevel2Controller(hops=REJOIN_HOPS)
    for _ in range(4000):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return obs, _fail_report(trail, notes, farm, "rejoin_death", env=env)
        if (
            snap.screen == 0x59
            and snap.mode == PLAY_MODE
            and snap.link_y < 200
        ):
            break
        act = nav.step(snap)
        obs, *_ = env.step(act.action)
        if nav.phase.name == "FAILED":
            return obs, _fail_report(trail, notes, farm, "rejoin_failed", env=env)
    notes.append("rejoin_59")
    _log("at_59")

    # --- East @y≈140 into 0x5A ---
    for f in range(1200):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return obs, _fail_report(trail, notes, farm, "east_death", env=env)
        if snap.screen == 0x5A:
            notes.append(f"east_5a_h{snap.filled_hearts}")
            break
        if abs(snap.link_y - EAST_Y) > 5 and snap.link_x < 200:
            d = "DOWN" if snap.link_y < EAST_Y else "UP"
            obs, *_ = env.step(
                swing_action(f, d, "east_ay", period=6, hold=2).action
            )
        else:
            obs, *_ = env.step(
                swing_action(f, "RIGHT", "east", period=5, hold=3).action
            )
    snap = read_snapshot(env.get_ram())
    if snap.screen != 0x5A or snap.mode == 17:
        return obs, _fail_report(trail, notes, farm, "east_miss_5a", env=env)
    _log("at_5a")

    # --- Corridor clear ---
    if corridor_clear_frames > 0:
        notes.extend(_kite_clear(env, corridor_clear_frames))
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return obs, _fail_report(trail, notes, farm, "clear_death", env=env)
        _log("post_clear")

    # --- Door path with maze ---
    door = OverworldToLevel2Controller(
        hops=LEVEL2_CLEAN_FROM_5A_TO_3C,
        require_level2_screen=True,
    )
    for _ in range(max_door_frames):
        snap = read_snapshot(env.get_ram())
        if not trail or trail[-1].get("screen") != snap.screen:
            _log("door")
        if snap.mode == 17:
            return obs, _fail_report(
                trail, notes, farm, "door_death", door=door, env=env
            )
        if (
            snap.screen == 0x3C
            and snap.mode == PLAY_MODE
            and snap.filled_hearts > 0
        ):
            door.success = True
            notes.append("clean_door_0x3c")
            break
        act = door.step(snap)
        obs, *_ = env.step(act.action)
        if door.success and snap.screen == 0x3C:
            break
        if door.phase.name == "FAILED":
            return obs, _fail_report(
                trail, notes, farm, "door_failed", door=door, env=env
            )

    snap = read_snapshot(env.get_ram())
    ok = (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == 0x3C
        and snap.filled_hearts > 0
    )
    return obs, {
        "ok": ok,
        "track": "clean",
        "notes": notes,
        "farm": farm.report(),
        "door": door.report(),
        "trail": trail,
        "final": {
            "screen": snap.screen,
            "mode": snap.mode,
            "level": snap.level,
            "hearts": f"{snap.filled_hearts}/{snap.heart_containers}",
            "filled_hearts": snap.filled_hearts,
            "health": snap.health,
            "x": snap.link_x,
            "y": snap.link_y,
        },
    }


def _fail_report(trail, notes, farm, reason, door=None, env=None) -> dict[str, Any]:
    final = None
    if env is not None:
        snap = read_snapshot(env.get_ram())
        final = {
            "screen": snap.screen,
            "mode": snap.mode,
            "level": snap.level,
            "hearts": f"{snap.filled_hearts}/{snap.heart_containers}",
            "filled_hearts": snap.filled_hearts,
            "health": snap.health,
            "x": snap.link_x,
            "y": snap.link_y,
        }
    elif trail:
        last = trail[-1]
        final = {
            "screen": last.get("screen"),
            "mode": last.get("mode"),
            "level": 0,
            "hearts": f"{last.get('hearts')}/?",
            "filled_hearts": last.get("hearts"),
            "health": last.get("health"),
            "x": last.get("x"),
            "y": last.get("y"),
        }
    return {
        "ok": False,
        "track": "clean",
        "notes": list(notes) + [reason],
        "farm": farm.report() if farm else None,
        "door": door.report() if door else None,
        "trail": trail,
        "final": final,
        "fail": reason,
    }
