"""Probe: finish secret entrance (room 0x55) after uncle / fighter sword.

Goal (acceptance):
  1. Fighter sword + hold-up cleared
  2. South combat chamber (no stair soft-lock)
  3. Clear soldiers / open chest path
  4. Leave secret entrance (room id change OR outdoors transition)

Descriptive names only in reports; hex room ids are diagnostics only.

Usage:
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/probe_secret_entrance_exit.py
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/probe_secret_entrance_exit.py --mode explore
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, globals().get('_SNES_IMPORT_ROOT', _REPO_ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from alttp import primitives  # noqa: E402
from alttp.paths import FIGHTER_SWORD_STATE, RECORDINGS_DIR  # noqa: E402
from alttp.ram import (  # noqa: E402
    HYRULE_CASTLE_SECRET_ENTRANCE_ROOM,
    room_label,
    snapshot_to_diag,
)
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames  # noqa: E402
from alttp.opening_route.secret_entrance_clear import (  # noqa: E402
    SOUTH_CHAMBER_Y_MAX,
    approach_south_chamber,
    ensure_sword_control,
)

OUT_DIR = RECORDINGS_DIR / "probe_secret_exit"
SECRET = HYRULE_CASTLE_SECRET_ENTRANCE_ROOM


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def _shot(env: object, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = snapshot_env(env)
    try:
        rendered = env.render()  # type: ignore[attr-defined]
        if rendered is not None:
            Image.fromarray(np.asarray(rendered)).save(path)
    except Exception as exc:  # noqa: BLE001
        return {"png": str(path), "error": str(exc), **snapshot_to_diag(snap)}
    return {"png": str(path.name), **snapshot_to_diag(snap)}


def _stage(name: str, env: object, out: Path, stages: list[dict]) -> dict[str, Any]:
    info = _shot(env, out / f"{name}.png")
    info["stage"] = name
    # active hostiles near Link
    hostiles = primitives.sprites_of_type(
        env, primitives.CASTLE_HOSTILE_TYPES, max_distance=220
    )
    info["hostiles"] = [
        {
            "slot": s.slot,
            "type": f"0x{s.sprite_type:02x}",
            "hp": s.hp,
            "x": s.x,
            "y": s.y,
        }
        for s in hostiles
    ]
    stages.append(info)
    print(
        f"[{name}] room={room_label(info['room_base_id'])} "
        f"0x{info['room_base_id']:02X} indoors={info['indoors']} "
        f"xy=({info['link_x']},{info['link_y']}) keys={info['num_keys']} "
        f"hostiles={len(hostiles)}"
    )
    return info


def left_secret_entrance(snap) -> bool:
    """True when no longer in secret-entrance room (exit achieved)."""
    if not snap.indoors:
        return True
    return snap.room_base_id != SECRET


def open_chest_nearby(env: object, *, max_cycles: int = 40) -> primitives.PrimitiveResult:
    """Face and mash A near a chest; settle text/hold-up."""
    frames = 0
    start_keys = snapshot_env(env).num_keys
    start_rupees = None  # not tracked; success via hold-up/text or key change
    for cycle in range(max_cycles):
        snap = snapshot_env(env)
        if left_secret_entrance(snap):
            return primitives.PrimitiveResult(True, "left room during chest", frames, snap)
        if snap.num_keys != start_keys and snap.num_keys != 0xFF:
            primitives.settle_control(env)
            return primitives.PrimitiveResult(
                True, "key obtained", frames, snapshot_env(env)
            )
        if snap.is_hold_up_item or snap.is_text_mode:
            settle = primitives.settle_control(env, max_frames=300)
            frames += settle.frames
            # dismiss hold-up after chest
            from alttp.opening_route.castle_to_sword import dismiss_hold_up_item

            frames += dismiss_hold_up_item(env)
            return primitives.PrimitiveResult(
                True, "chest opened (hold-up/text)", frames, snapshot_env(env)
            )
        # face right then A (chest often east of platform approach), then left/up
        facing = ("RIGHT", "UP", "LEFT", "DOWN")[cycle % 4]
        step_frames(env, action_for(facing), 2)
        step_frames(env, action_for("A"), 3)
        step_frames(env, no_action(), 4)
        frames += 9
    return primitives.PrimitiveResult(False, "no chest response", frames, snapshot_env(env))


def try_path(env: object, script: list[tuple[tuple[str, ...], int]], label: str) -> dict:
    start = snapshot_env(env)
    result = primitives.run_script(
        env,
        script,
        stop_when=left_secret_entrance,
    )
    end = result.snapshot
    return {
        "label": label,
        "ok_exit": left_secret_entrance(end),
        "reason": result.reason,
        "frames": result.frames,
        "start_xy": [start.link_x, start.link_y],
        "end_xy": [end.link_x, end.link_y],
        "end_room": room_label(end.room_base_id),
        "end_room_hex": f"0x{end.room_base_id:02X}",
        "indoors": end.indoors,
        "keys": end.num_keys,
        "screen": f"0x{end.screen_id:02X}",
    }


def mode_clear(env: object, out: Path) -> dict[str, Any]:
    """Measured clear attempt: control → south → fight → chest → door search."""
    stages: list[dict] = []
    report: dict[str, Any] = {"mode": "clear", "stages": stages, "attempts": []}

    ready = ensure_sword_control(env)
    report["ensure_sword_control"] = {
        "ok": ready.ok,
        "detail": ready.detail,
        "frames": ready.frames,
    }
    _stage("00_sword_control", env, out, stages)
    if not ready.ok:
        report["blocker"] = ready.detail
        return report

    south = approach_south_chamber(env)
    report["approach_south_chamber"] = {
        "ok": south.ok,
        "detail": south.detail,
        "frames": south.frames,
    }
    _stage("01_south_chamber", env, out, stages)
    if not south.ok:
        report["blocker"] = south.detail
        return report

    # Cap y — if too deep south, walk north a bit before combat.
    snap = snapshot_env(env)
    if snap.link_y > SOUTH_CHAMBER_Y_MAX:
        primitives.run_script(env, ((("UP",), 40),))
        primitives.settle_control(env)
        _stage("01b_north_from_stair", env, out, stages)

    fight = primitives.fight_nearby(
        env,
        room=SECRET,
        max_distance=160,
        attack_distance=48,
        max_cycles=700,
        stop_when=lambda e: left_secret_entrance(snapshot_env(e)),
    )
    report["fight"] = {
        "ok": fight.ok,
        "reason": fight.reason,
        "frames": fight.frames,
        "defeated": list(fight.defeated_slots),
    }
    _stage("02_after_fight", env, out, stages)

    # Approach green platform / chest area. Measured roughly east of south chamber.
    # From ~2680,2925: move east-north onto platform.
    for label, wps in (
        (
            "toward_green_platform",
            [
                primitives.Waypoint(2720, 2900, tolerance=12, room=SECRET, label="mid"),
                primitives.Waypoint(2800, 2880, tolerance=12, room=SECRET, label="green_approach"),
                primitives.Waypoint(2860, 2870, tolerance=10, room=SECRET, label="on_green"),
            ],
        ),
    ):
        path = primitives.move_path(env, wps, max_frames_per_waypoint=500)
        report["attempts"].append(
            {
                "label": label,
                "ok": path.ok,
                "reason": path.reason,
                "frames": path.frames,
                "xy": [path.snapshot.link_x, path.snapshot.link_y],
            }
        )
        _stage("03_green_platform", env, out, stages)

    # Kill any remaining local hostiles on/near platform.
    fight2 = primitives.fight_nearby(
        env,
        room=SECRET,
        max_distance=120,
        attack_distance=48,
        max_cycles=400,
    )
    report["fight2"] = {
        "ok": fight2.ok,
        "reason": fight2.reason,
        "frames": fight2.frames,
        "defeated": list(fight2.defeated_slots),
    }
    _stage("04_cleared_near_chest", env, out, stages)

    # Open chest
    chest = open_chest_nearby(env)
    report["chest"] = {
        "ok": chest.ok,
        "reason": chest.reason,
        "frames": chest.frames,
        "keys": chest.snapshot.num_keys,
        "xy": [chest.snapshot.link_x, chest.snapshot.link_y],
    }
    _stage("05_post_chest", env, out, stages)

    # Ray-walk search for room exit from current position (avoid deep south stairs).
    # Snapshot restore not available — try short probes from chest area carefully.
    snap = snapshot_env(env)
    report["pre_exit_search"] = snapshot_to_diag(snap)

    exit_scripts: list[tuple[str, list[tuple[tuple[str, ...], int]]]] = [
        ("west_120", [(("LEFT",), 120)]),
        ("west_200", [(("LEFT",), 200)]),
        ("north_100", [(("UP",), 100)]),
        ("north_200", [(("UP",), 200)]),
        ("east_100", [(("RIGHT",), 100)]),
        ("nw_diagonal", [(("LEFT", "UP"), 80), (("UP",), 80)]),
        ("west_then_north", [(("LEFT",), 100), (("UP",), 120)]),
        ("north_then_west", [(("UP",), 80), (("LEFT",), 120)]),
        ("south_shallow_then_west", [(("DOWN",), 40), (("LEFT",), 150)]),
        ("climb_north_steps", [(("UP",), 60), (("RIGHT",), 40), (("UP",), 100)]),
        # Walkthrough: climb steps and exit — try up-east corridor
        ("platform_north_east", [(("UP",), 50), (("RIGHT",), 80), (("UP",), 120)]),
        ("long_west", [(("LEFT",), 300)]),
        ("long_north", [(("UP",), 300)]),
    ]

    # Reload FighterSword for each ray so we don't get stuck. Uses rebuild.
    for label, script in exit_scripts:
        # can't cheaply reload mid-run without env reset; use sequential from
        # current if still in room, else skip.
        if left_secret_entrance(snapshot_env(env)):
            report["exit"] = {
                "ok": True,
                "via": "prior_action",
                **snapshot_to_diag(snapshot_env(env)),
            }
            _stage("99_exited", env, out, stages)
            return report
        # skip if we already left
        attempt = try_path(env, script, label)
        report["attempts"].append(attempt)
        _stage(f"06_{label}", env, out, stages)
        if attempt["ok_exit"]:
            report["exit"] = attempt
            report["ok"] = True
            return report
        # if deep south stair pocket, nudge north
        s = snapshot_env(env)
        if s.link_y > SOUTH_CHAMBER_Y_MAX + 30:
            primitives.run_script(env, ((("UP",), 80),))
            primitives.settle_control(env)

    # Final fight + free roam west/north grid
    for y_off in (0, -40, -80, 40):
        for x_off in (0, -80, -160, -240, 80, 160):
            s = snapshot_env(env)
            if left_secret_entrance(s):
                report["exit"] = {"ok": True, "via": "grid", **snapshot_to_diag(s)}
                report["ok"] = True
                _stage("99_exited", env, out, stages)
                return report
            tx = max(2400, min(3200, s.link_x + x_off))
            ty = max(2600, min(SOUTH_CHAMBER_Y_MAX, s.link_y + y_off))
            wp = primitives.Waypoint(tx, ty, tolerance=14, room=SECRET, label=f"g_{x_off}_{y_off}")
            res = primitives.move_to(env, wp, max_frames=240)
            if left_secret_entrance(res.snapshot):
                report["exit"] = {
                    "ok": True,
                    "via": f"grid_{x_off}_{y_off}",
                    **snapshot_to_diag(res.snapshot),
                }
                report["ok"] = True
                _stage("99_exited", env, out, stages)
                return report

    _stage("98_stuck", env, out, stages)
    report["ok"] = False
    report["blocker"] = (
        f"still in secret entrance xy=({snapshot_env(env).link_x},"
        f"{snapshot_env(env).link_y}) keys={snapshot_env(env).num_keys}"
    )
    return report


def mode_explore(env: object, out: Path) -> dict[str, Any]:
    """Reload-based ray probes from post-sword and from south chamber."""
    stages: list[dict] = []
    report: dict[str, Any] = {"mode": "explore", "rays": []}

    def reload_to(phase: str) -> None:
        env.reset()  # type: ignore[attr-defined]
        primitives.settle_control(env)
        ready = ensure_sword_control(env)
        if not ready.ok:
            raise RuntimeError(ready.detail)
        if phase == "south":
            south = approach_south_chamber(env)
            if not south.ok:
                raise RuntimeError(south.detail)
            s = snapshot_env(env)
            if s.link_y > SOUTH_CHAMBER_Y_MAX:
                primitives.run_script(env, ((("UP",), 40),))

    rays_from_sword = [
        ("S_from_uncle", [(("DOWN",), 200)]),
        ("L100_D250", [(("LEFT",), 100), (("DOWN",), 250)]),  # measured south
        ("L150_D200", [(("LEFT",), 150), (("DOWN",), 200)]),
        ("W_from_uncle", [(("LEFT",), 200)]),
        ("E_from_uncle", [(("RIGHT",), 200)]),
        ("N_from_uncle", [(("UP",), 100)]),
        ("SE_from_uncle", [(("RIGHT", "DOWN"), 180)]),
        ("SW_from_uncle", [(("LEFT", "DOWN"), 180)]),
    ]

    for label, script in rays_from_sword:
        reload_to("sword")
        _stage(f"ray_start_{label}", env, out, stages)
        attempt = try_path(env, script, f"sword_{label}")
        report["rays"].append(attempt)
        _stage(f"ray_end_{label}", env, out, stages)
        print(f"  ray {label}: exit={attempt['ok_exit']} end={attempt['end_xy']} room={attempt['end_room']}")

    # From south: fight first then rays
    reload_to("south")
    fight = primitives.fight_nearby(env, room=SECRET, max_distance=160, max_cycles=600)
    report["south_fight"] = {
        "ok": fight.ok,
        "reason": fight.reason,
        "defeated": list(fight.defeated_slots),
    }
    _stage("south_cleared", env, out, stages)

    # Save "south cleared" coords for path design
    south_xy = (snapshot_env(env).link_x, snapshot_env(env).link_y)
    report["south_cleared_xy"] = list(south_xy)

    south_rays = [
        ("W200", [(("LEFT",), 200)]),
        ("W300", [(("LEFT",), 300)]),
        ("N150", [(("UP",), 150)]),
        ("N250", [(("UP",), 250)]),
        ("E150", [(("RIGHT",), 150)]),
        ("E250", [(("RIGHT",), 250)]),
        ("NW", [(("LEFT", "UP"), 150)]),
        ("NE", [(("RIGHT", "UP"), 150)]),
        ("W_N", [(("LEFT",), 150), (("UP",), 150)]),
        ("N_W", [(("UP",), 100), (("LEFT",), 150)]),
        ("E_N", [(("RIGHT",), 120), (("UP",), 120)]),
        ("to_chest_then_N", [
            (("RIGHT",), 100),
            (("UP",), 40),
            (("RIGHT",), 60),
            (("A",), 4),
            (("NONE",), 60),
            (("UP",), 120),
        ]),
        ("to_chest_then_W", [
            (("RIGHT",), 100),
            (("UP",), 40),
            (("RIGHT",), 60),
            (("A",), 4),
            (("NONE",), 60),
            (("LEFT",), 200),
        ]),
        ("shallow_S_then_W", [(("DOWN",), 30), (("LEFT",), 250)]),
        # Avoid stair pocket: never down>60 from south chamber
        ("stairs_test_S80", [(("DOWN",), 80)]),
        ("stairs_test_S150", [(("DOWN",), 150)]),
    ]

    for label, script in south_rays:
        reload_to("south")
        primitives.fight_nearby(env, room=SECRET, max_distance=160, max_cycles=500)
        # keep y safe
        s = snapshot_env(env)
        if s.link_y > SOUTH_CHAMBER_Y_MAX:
            primitives.run_script(env, ((("UP",), 40),))
        attempt = try_path(env, script, f"south_{label}")
        report["rays"].append(attempt)
        _stage(f"sray_{label}", env, out, stages)
        print(
            f"  sray {label}: exit={attempt['ok_exit']} "
            f"end={attempt['end_xy']} room={attempt['end_room']} "
            f"indoors={attempt['indoors']}"
        )
        if attempt["ok_exit"]:
            report["first_exit"] = attempt

    exits = [r for r in report["rays"] if r.get("ok_exit")]
    report["ok"] = bool(exits)
    report["exit_count"] = len(exits)
    return report


def mode_reload_exit_hunt(env_factory, out: Path) -> dict[str, Any]:
    """Each candidate path starts from a fresh FighterSword load."""
    stages: list[dict] = []
    report: dict[str, Any] = {"mode": "reload_exit_hunt", "candidates": []}

    # Full candidate scripts from sword control → hoped exit
    candidates: list[tuple[str, list[tuple[tuple[str, ...], int]]]] = [
        # Measured south then free explore patterns
        (
            "measured_south_fight_west",
            [
                (("LEFT",), 100),
                (("DOWN",), 250),
                # fight via later fight_nearby
            ],
        ),
    ]

    # Build richer candidate list as sequences of move segments + fight + open
    plans: list[dict[str, Any]] = [
        {
            "name": "south_clear_chest_west",
            "south": True,
            "fight": True,
            "chest_wps": [
                (2780, 2890),
                (2860, 2870),
            ],
            "open_chest": True,
            "after": [
                (("LEFT",), 200),
                (("UP",), 100),
                (("LEFT",), 150),
            ],
        },
        {
            "name": "south_clear_chest_north",
            "south": True,
            "fight": True,
            "chest_wps": [(2860, 2870)],
            "open_chest": True,
            "after": [
                (("UP",), 200),
                (("LEFT",), 100),
                (("UP",), 150),
            ],
        },
        {
            "name": "south_clear_east_corridor",
            "south": True,
            "fight": True,
            "chest_wps": [(2900, 2880), (2940, 2860)],
            "open_chest": True,
            "after": [
                (("RIGHT",), 150),
                (("UP",), 200),
                (("RIGHT",), 100),
            ],
        },
        {
            "name": "south_clear_west_corridor_long",
            "south": True,
            "fight": True,
            "chest_wps": [(2860, 2870)],
            "open_chest": True,
            "after": [
                (("LEFT",), 80),
                (("DOWN",), 40),
                (("LEFT",), 250),
                (("UP",), 120),
                (("LEFT",), 200),
            ],
        },
        {
            "name": "south_clear_up_steps_guess",
            "south": True,
            "fight": True,
            "chest_wps": [(2860, 2865)],
            "open_chest": True,
            "after": [
                (("UP",), 40),
                (("LEFT",), 60),
                (("UP",), 200),
                (("LEFT",), 80),
                (("UP",), 150),
            ],
        },
        {
            "name": "south_no_chest_west_north",
            "south": True,
            "fight": True,
            "chest_wps": [],
            "open_chest": False,
            "after": [
                (("LEFT",), 180),
                (("UP",), 180),
                (("LEFT",), 120),
                (("UP",), 100),
            ],
        },
        {
            "name": "south_shallow_only_west",
            "south": True,
            "fight": True,
            "chest_wps": [(2700, 2900)],
            "open_chest": False,
            "after": [
                (("LEFT",), 300),
                (("UP",), 50),
                (("LEFT",), 200),
            ],
        },
        {
            "name": "walkthrough_climb_exit",
            "south": True,
            "fight": True,
            "chest_wps": [(2860, 2870)],
            "open_chest": True,
            "after": [
                # after rupee chest, climb steps — green platform lip then north/west door
                (("LEFT",), 30),
                (("DOWN",), 20),
                (("LEFT",), 100),
                (("UP",), 80),
                (("LEFT",), 80),
                (("UP",), 200),
            ],
        },
    ]

    for plan in plans:
        env = env_factory()
        try:
            env.reset()
            primitives.settle_control(env)
            ready = ensure_sword_control(env)
            if not ready.ok:
                report["candidates"].append({"name": plan["name"], "ok": False, "err": ready.detail})
                continue
            if plan.get("south"):
                south = approach_south_chamber(env)
                if not south.ok:
                    report["candidates"].append(
                        {"name": plan["name"], "ok": False, "err": south.detail}
                    )
                    continue
                s = snapshot_env(env)
                if s.link_y > SOUTH_CHAMBER_Y_MAX:
                    primitives.run_script(env, ((("UP",), 40),))
            if plan.get("fight"):
                primitives.fight_nearby(
                    env,
                    room=SECRET,
                    max_distance=160,
                    attack_distance=48,
                    max_cycles=700,
                    stop_when=lambda e: left_secret_entrance(snapshot_env(e)),
                )
            for x, y in plan.get("chest_wps") or []:
                primitives.move_to(
                    env,
                    primitives.Waypoint(x, y, tolerance=12, room=SECRET, label="wp"),
                    max_frames=500,
                )
                if left_secret_entrance(snapshot_env(env)):
                    break
            if plan.get("open_chest") and not left_secret_entrance(snapshot_env(env)):
                open_chest_nearby(env)
            if not left_secret_entrance(snapshot_env(env)):
                after = plan.get("after") or []
                primitives.run_script(
                    env,
                    after,
                    stop_when=left_secret_entrance,
                )
            # extra fight if hostiles block door
            if not left_secret_entrance(snapshot_env(env)):
                primitives.fight_nearby(env, room=SECRET, max_distance=100, max_cycles=200)
                primitives.run_script(
                    env,
                    ((("LEFT",), 100), (("UP",), 100), (("RIGHT",), 80), (("UP",), 120)),
                    stop_when=left_secret_entrance,
                )

            snap = snapshot_env(env)
            exited = left_secret_entrance(snap)
            entry = {
                "name": plan["name"],
                "ok_exit": exited,
                "xy": [snap.link_x, snap.link_y],
                "room": room_label(snap.room_base_id),
                "room_hex": f"0x{snap.room_base_id:02X}",
                "indoors": snap.indoors,
                "keys": snap.num_keys,
                "screen": f"0x{snap.screen_id:02X}",
                "diag": snapshot_to_diag(snap),
            }
            shot_name = f"plan_{plan['name']}"
            _stage(shot_name, env, out, stages)
            report["candidates"].append(entry)
            print(
                f"PLAN {plan['name']}: exit={exited} "
                f"room={entry['room']} xy={entry['xy']} keys={entry['keys']}"
            )
            if exited:
                report["ok"] = True
                report["winner"] = entry
                # Continue a bit into next room for screenshots
                primitives.settle_control(env)
                primitives.run_script(env, ((("UP",), 40), (("LEFT",), 40), (("RIGHT",), 40)))
                primitives.settle_control(env)
                _stage("99_after_exit", env, out, stages)
                report["after_exit"] = snapshot_to_diag(snapshot_env(env))
                return report
        finally:
            env.close()

    report["ok"] = False
    report["blocker"] = "no candidate left secret entrance"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        default=FIGHTER_SWORD_STATE,
        help=f"Dev state (default {FIGHTER_SWORD_STATE})",
    )
    parser.add_argument(
        "--mode",
        choices=("clear", "explore", "hunt"),
        default="hunt",
        help="clear=one continuous run; explore=rays; hunt=reload candidates",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help=f"Output dir (default {OUT_DIR})",
    )
    args = parser.parse_args()
    _configure_headless()
    args.out.mkdir(parents=True, exist_ok=True)

    def factory():
        return build_boot_env(args.state)

    if args.mode == "hunt":
        report = mode_reload_exit_hunt(factory, args.out)
    else:
        env = factory()
        try:
            env.reset()
            primitives.settle_control(env)
            if args.mode == "explore":
                report = mode_explore(env, args.out)
            else:
                report = mode_clear(env, args.out)
        finally:
            env.close()

    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k not in {"stages"}}, indent=2))
    print(f"Wrote {report_path}")
    print(f"Screenshots in {args.out}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
