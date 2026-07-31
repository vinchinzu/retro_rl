"""Tiered probe: courtyard secret pocket → main castle door (room 0x61).

Per docs/TRIGGER_HANDOFF.md, keep tiers separate:

  route     — any walkable escape from the hedge pocket toward courtyard
  approach  — local window at the main castle door (coords TBD)
  trigger   — exact door entry frames / facing → indoors main hall

Natural predecessor: secret-entrance stairs exit (``sword_to_zelda``).

Usage:
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/probe_courtyard_main_door.py
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/probe_courtyard_main_door.py --tier route
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/probe_courtyard_main_door.py --tier bfs
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/probe_courtyard_main_door.py --tier scripts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alttp import primitives  # noqa: E402
from alttp.opening_route.anchors import (  # noqa: E402
    COURTYARD_SECRET_POCKET_TOLERANCE,
    COURTYARD_SECRET_POCKET_X,
    COURTYARD_SECRET_POCKET_Y,
)
from alttp.opening_route.sword_to_zelda import run_from_sword  # noqa: E402
from alttp.paths import FIGHTER_SWORD_STATE, RECORDINGS_DIR  # noqa: E402
from alttp.ram import (  # noqa: E402
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_SCREEN,
    room_label,
    snapshot_to_diag,
)
from alttp.startup import (  # noqa: E402
    action_for,
    build_boot_env,
    no_action,
    snapshot_env,
    step_frames,
)

OUT_DIR = RECORDINGS_DIR / "probe_courtyard_door"

# Escape from pocket: leave the tight landing window (route tier success).
POCKET_ESCAPE_TOL = max(COURTYARD_SECRET_POCKET_TOLERANCE, 64)

# Candidate main-door approach guesses (world coords on screen 0x1B).
# Refined after route discovery; main south hall is room 0x61.
DOOR_APPROACH_CANDIDATES: tuple[tuple[str, int, int], ...] = (
    ("door_center_n", 2200, 1600),
    ("door_center", 2200, 1650),
    ("door_west", 2100, 1650),
    ("door_east", 2300, 1650),
    ("door_far_n", 2200, 1550),
    ("door_nnw", 2150, 1580),
    ("door_nne", 2250, 1580),
    # From secret hole x≈2432 the courtyard is west; try mid courtyard
    ("mid_court", 2150, 1700),
    ("west_court", 2050, 1700),
    ("north_court", 2200, 1620),
)


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
            return {"png": path.name, **snapshot_to_diag(snap)}
    except Exception as exc:  # noqa: BLE001
        return {"png": str(path), "error": str(exc), **snapshot_to_diag(snap)}
    return {"png": path.name, **snapshot_to_diag(snap)}


def _in_pocket(snap) -> bool:
    if snap.indoors or snap.dark_world or snap.screen_id != HYRULE_CASTLE_SCREEN:
        return False
    return (
        abs(snap.link_x - COURTYARD_SECRET_POCKET_X) <= POCKET_ESCAPE_TOL
        and abs(snap.link_y - COURTYARD_SECRET_POCKET_Y) <= POCKET_ESCAPE_TOL
    )


def _escaped_pocket(snap) -> bool:
    """Route-tier: outdoors on 0x1B, sword, clearly outside landing window."""
    if snap.indoors or snap.dark_world:
        return False
    if snap.screen_id != HYRULE_CASTLE_SCREEN:
        return True  # left castle screen entirely — still a route result
    if not snap.has_fighter_sword:
        return False
    dx = abs(snap.link_x - COURTYARD_SECRET_POCKET_X)
    dy = abs(snap.link_y - COURTYARD_SECRET_POCKET_Y)
    return dx > POCKET_ESCAPE_TOL or dy > POCKET_ESCAPE_TOL


def _entered_main_hall(snap) -> bool:
    return (
        snap.indoors
        and (not snap.dark_world)
        and snap.room_base_id == HYRULE_CASTLE_MAIN_HALL_ROOM
    )


def _reentered_secret(snap) -> bool:
    """UP from pocket re-enters secret entrance 0x55 — not a success."""
    from alttp.ram import SECRET_PASSAGE_ROOM

    return (
        snap.indoors
        and (not snap.dark_world)
        and snap.room_base_id == SECRET_PASSAGE_ROOM
    )


def _entered_progress_indoor(snap) -> bool:
    """Indoors into castle progress (not secret-entrance re-entry)."""
    return (
        snap.indoors
        and (not snap.dark_world)
        and not _reentered_secret(snap)
    )


def reach_pocket(env: object) -> dict[str, Any]:
    """FighterSword → secret-entrance clear → courtyard pocket."""
    primitives.settle_control(env)
    result = run_from_sword(env, source="state_load_dev")
    snap = result.snapshot
    ok = bool(
        result.ok
        and result.acceptance.get("left_secret_entrance")
        and (not snap.indoors)
        and snap.screen_id == HYRULE_CASTLE_SCREEN
        and snap.has_fighter_sword
    )
    return {
        "ok": ok,
        "phase": result.phase,
        "frames": result.frames,
        "blocker": result.blocker,
        "diag": snapshot_to_diag(snap),
    }


def capture_pocket_state(env: object) -> bytes:
    return env.em.get_state()  # type: ignore[attr-defined]


def restore_pocket(env: object, state: bytes) -> None:
    env.em.set_state(state)  # type: ignore[attr-defined]
    # One no-op frame so RAM/render catch up.
    step_frames(env, no_action(), 1)
    primitives.settle_control(env, max_frames=60)


def swing_cut(env: object, facing: str | None = None, *, swings: int = 2) -> int:
    """Face optional direction and swing sword to cut bushes."""
    frames = 0
    if facing:
        step_frames(env, action_for(facing), 2)
        frames += 2
    for _ in range(swings):
        step_frames(env, action_for("B"), 4)
        step_frames(env, no_action(), 4)
        frames += 8
    return frames


def try_script(
    env: object,
    script: list[tuple[tuple[str, ...], int]],
    *,
    cut_on_stuck: bool = False,
) -> dict[str, Any]:
    start = snapshot_env(env)
    frames = 0
    if cut_on_stuck:
        # Walk script with periodic swings when position freezes.
        prev = (start.link_x, start.link_y)
        stuck = 0
        for buttons, n in script:
            remaining = n
            while remaining > 0:
                chunk = min(4, remaining)
                step_frames(env, action_for(*buttons) if buttons != ("NONE",) else no_action(), chunk)
                frames += chunk
                remaining -= chunk
                snap = snapshot_env(env)
                if _entered_progress_indoor(snap) or _escaped_pocket(snap):
                    break
                if _reentered_secret(snap):
                    break
                xy = (snap.link_x, snap.link_y)
                if xy == prev:
                    stuck += 1
                    if stuck >= 3:
                        # Swing toward movement intent
                        face = buttons[0] if buttons and buttons[0] in {
                            "UP", "DOWN", "LEFT", "RIGHT"
                        } else None
                        frames += swing_cut(env, face, swings=2)
                        stuck = 0
                else:
                    stuck = 0
                    prev = xy
            if (
                _entered_progress_indoor(snapshot_env(env))
                or _escaped_pocket(snapshot_env(env))
                or _reentered_secret(snapshot_env(env))
            ):
                break
    else:
        res = primitives.run_script(
            env,
            script,
            stop_when=lambda s: (
                _entered_progress_indoor(s)
                or _escaped_pocket(s)
                or _reentered_secret(s)
            ),
        )
        frames = res.frames

    end = snapshot_env(env)
    primitives.settle_control(env, max_frames=40)
    end = snapshot_env(env)
    return {
        "start_xy": [start.link_x, start.link_y],
        "end_xy": [end.link_x, end.link_y],
        "frames": frames,
        "escaped_pocket": _escaped_pocket(end) and not end.indoors,
        "indoors": end.indoors,
        "reentered_secret": _reentered_secret(end),
        "room": room_label(end.room_base_id),
        "room_hex": f"0x{end.room_base_id:02X}",
        "screen": f"0x{end.screen_id:02X}",
        "main_hall": _entered_main_hall(end),
        "progress_indoor": _entered_progress_indoor(end),
        "dx": end.link_x - COURTYARD_SECRET_POCKET_X,
        "dy": end.link_y - COURTYARD_SECRET_POCKET_Y,
        "diag": snapshot_to_diag(end),
    }


def tier_scripts(env: object, pocket_state: bytes, out: Path) -> dict[str, Any]:
    """Try explicit open-loop paths with optional bush-cutting."""
    report: dict[str, Any] = {"tier": "scripts", "attempts": []}
    plans: list[tuple[str, list[tuple[tuple[str, ...], int]], bool]] = [
        # Cardinal rays from pad
        ("DOWN_120", [(("DOWN",), 120)], False),
        ("DOWN_200", [(("DOWN",), 200)], False),
        ("LEFT_120", [(("LEFT",), 120)], False),
        ("LEFT_200", [(("LEFT",), 200)], False),
        ("LEFT_400", [(("LEFT",), 400)], False),
        ("RIGHT_80", [(("RIGHT",), 80)], False),
        ("UP_40", [(("UP",), 40)], False),  # expected re-enter stairs
        # Off pad then west corridor
        ("S40_W200", [(("DOWN",), 40), (("LEFT",), 200)], False),
        ("S40_W400", [(("DOWN",), 40), (("LEFT",), 400)], False),
        ("S20_W100_S40_W200", [
            (("DOWN",), 20), (("LEFT",), 100),
            (("DOWN",), 40), (("LEFT",), 200),
        ], False),
        ("S40_W100_N40_W200", [
            (("DOWN",), 40), (("LEFT",), 100),
            (("UP",), 40), (("LEFT",), 200),
        ], False),
        ("S60_W60_S60_W200", [
            (("DOWN",), 60), (("LEFT",), 60),
            (("DOWN",), 60), (("LEFT",), 200),
        ], False),
        # With bush cutting
        ("S_cut_W", [(("DOWN",), 40), (("LEFT",), 200)], True),
        ("S_cut_S_W", [
            (("DOWN",), 30), (("DOWN",), 80), (("LEFT",), 200),
        ], True),
        ("W_cut_S_W", [
            (("LEFT",), 40), (("DOWN",), 40), (("LEFT",), 200), (("DOWN",), 100),
        ], True),
        ("S_cut_long", [(("DOWN",), 300)], True),
        ("W_cut_long", [(("LEFT",), 400)], True),
        # Off-pad then micro-explore with cuts
        ("offpad_W_cut", [(("DOWN",), 30), (("LEFT",), 80), (("LEFT",), 200)], True),
        ("offpad_SW_cut", [
            (("DOWN",), 30), (("LEFT",), 40), (("DOWN",), 40),
            (("LEFT",), 40), (("DOWN",), 40), (("LEFT",), 80),
            (("DOWN",), 60), (("LEFT",), 120),
        ], True),
        ("offpad_WSW_cut", [
            (("DOWN",), 28), (("LEFT",), 60), (("DOWN",), 20),
            (("LEFT",), 60), (("DOWN",), 20), (("LEFT",), 100),
            (("UP",), 20), (("LEFT",), 100), (("UP",), 100),
        ], True),
        # A-lift bushes (no gloves needed for small green)
        ("S_lift_A", [(("DOWN",), 24), (("A",), 6), (("NONE",), 20), (("DOWN",), 40),
                       (("A",), 6), (("NONE",), 20), (("DOWN",), 40),
                       (("LEFT",), 100)], False),
        ("S_lift_then_W", [
            (("DOWN",), 28), (("A",), 6), (("NONE",), 16), (("DOWN",), 20),
            (("LEFT",), 40), (("A",), 6), (("NONE",), 16), (("LEFT",), 80),
            (("DOWN",), 40), (("LEFT",), 120),
        ], False),
        # Hedge-maze style: S then alternate W/S with cuts
        ("maze_S_W_S_W", [
            (("DOWN",), 50), (("LEFT",), 80),
            (("DOWN",), 50), (("LEFT",), 80),
            (("DOWN",), 50), (("LEFT",), 120),
            (("UP",), 80), (("LEFT",), 100),
        ], True),
        ("maze_S_W_N_W_N", [
            (("DOWN",), 40), (("LEFT",), 100),
            (("UP",), 30), (("LEFT",), 100),
            (("UP",), 80), (("LEFT",), 100),
            (("UP",), 120),
        ], True),
        ("maze_S_W_W_N_castle", [
            (("DOWN",), 50), (("LEFT",), 150),
            (("LEFT",), 150), (("UP",), 200),
            (("LEFT",), 80), (("UP",), 200),
        ], True),
        # Push hard west then north toward door
        ("W300_N300", [(("LEFT",), 300), (("UP",), 300)], True),
        ("S30_W300_N400", [
            (("DOWN",), 30), (("LEFT",), 300), (("UP",), 400),
        ], True),
        ("S30_W200_N100_W100_N300", [
            (("DOWN",), 30), (("LEFT",), 200),
            (("UP",), 100), (("LEFT",), 100),
            (("UP",), 300),
        ], True),
        # Try east (unlikely) and south-east
        ("S40_E100", [(("DOWN",), 40), (("RIGHT",), 100)], True),
        ("S100_W50_S100", [
            (("DOWN",), 100), (("LEFT",), 50), (("DOWN",), 100),
        ], True),
    ]

    best_dist = 0
    best_name = ""
    for name, script, cut in plans:
        restore_pocket(env, pocket_state)
        attempt = try_script(env, script, cut_on_stuck=cut)
        attempt["name"] = name
        attempt["cut"] = cut
        report["attempts"].append(attempt)
        dist = abs(attempt["dx"]) + abs(attempt["dy"])
        print(
            f"  {name}: esc={attempt['escaped_pocket']} "
            f"prog={attempt['progress_indoor']} main={attempt['main_hall']} "
            f"secret={attempt['reentered_secret']} "
            f"end={attempt['end_xy']} d=({attempt['dx']},{attempt['dy']}) "
            f"room={attempt['room']}"
        )
        _shot(env, out / f"script_{name}.png")
        if not attempt["reentered_secret"] and dist > best_dist:
            best_dist = dist
            best_name = name
        if attempt["main_hall"]:
            report["ok"] = True
            report["winner"] = attempt
            report["tier_result"] = "trigger"
            # Keep scanning other scripts for alternate paths; don't return.
            continue
        if attempt["progress_indoor"]:
            report.setdefault("progress_hits", []).append(attempt)
            report["ok"] = True
            report.setdefault("winner", attempt)
            report.setdefault("tier_result", "trigger_other_room")
            continue
        if attempt["escaped_pocket"]:
            report.setdefault("escape_hits", []).append(attempt)

    if report.get("tier_result") == "trigger":
        return report
    if report.get("progress_hits"):
        report["ok"] = True
        report["tier_result"] = "trigger_other_room"
        report["winner"] = report["progress_hits"][0]
        return report
    if report.get("escape_hits"):
        report["ok"] = True
        report["tier_result"] = "route"
        report["winner"] = report["escape_hits"][0]
    else:
        report["ok"] = False
        report["tier_result"] = "stuck"
        report["best_name"] = best_name
        report["best_dist"] = best_dist
    return report


def tier_bfs(
    env: object,
    pocket_state: bytes,
    out: Path,
    *,
    quant: int = 8,
    max_nodes: int = 400,
    step_frames_n: int = 10,
) -> dict[str, Any]:
    """State-restore BFS over outdoor movement + bush swings.

    Each node restores pocket base then replays the path prefix so we do not
    need full per-node emulator snapshots (cheaper: store action chain).
    For correctness under bush destruction (irreversible), we store full
    emulator state at each frontier node.
    """
    report: dict[str, Any] = {"tier": "bfs", "quant": quant, "points": []}
    # Frontier: (em_state, actions_label, x, y)
    restore_pocket(env, pocket_state)
    start = snapshot_env(env)
    start_state = capture_pocket_state(env)
    start_key = (start.link_x // quant, start.link_y // quant)

    visited: set[tuple[int, int]] = {start_key}
    # Store state bytes only for frontier; cap memory.
    queue: deque[tuple[bytes, str, int, int]] = deque(
        [(start_state, "", start.link_x, start.link_y)]
    )
    points: list[list[int]] = [[start.link_x, start.link_y]]
    extremes = {
        "min_x": start.link_x,
        "max_x": start.link_x,
        "min_y": start.link_y,
        "max_y": start.link_y,
    }
    # Actions: move cardinal, or swing+nudge in a facing.
    ACTIONS: list[tuple[str, tuple[str, ...], int, bool]] = [
        ("U", ("UP",), step_frames_n, False),
        ("D", ("DOWN",), step_frames_n, False),
        ("L", ("LEFT",), step_frames_n, False),
        ("R", ("RIGHT",), step_frames_n, False),
        ("sU", ("UP",), 4, True),
        ("sD", ("DOWN",), 4, True),
        ("sL", ("LEFT",), 4, True),
        ("sR", ("RIGHT",), 4, True),
    ]

    expanded = 0
    found_escape: dict[str, Any] | None = None
    found_indoor: dict[str, Any] | None = None

    while queue and expanded < max_nodes:
        state, path, _x, _y = queue.popleft()
        expanded += 1
        for label, buttons, n, do_swing in ACTIONS:
            env.em.set_state(state)  # type: ignore[attr-defined]
            step_frames(env, no_action(), 1)
            frames = 0
            if do_swing:
                frames += swing_cut(env, buttons[0], swings=2)
            step_frames(env, action_for(*buttons), n)
            frames += n
            # Settle brief transitions
            snap = snapshot_env(env)
            if snap.submodule != 0 or snap.game_mode not in (0x07, 0x09):
                for _ in range(30):
                    step_frames(env, no_action(), 2)
                    snap = snapshot_env(env)
                    if snap.has_control:
                        break

            if snap.game_mode == 0x12:
                continue  # dead

            new_path = path + label
            info = {
                "path": new_path,
                "xy": [snap.link_x, snap.link_y],
                "indoors": snap.indoors,
                "room": room_label(snap.room_base_id),
                "room_hex": f"0x{snap.room_base_id:02X}",
                "screen": f"0x{snap.screen_id:02X}",
                "escaped": _escaped_pocket(snap),
                "main_hall": _entered_main_hall(snap),
            }

            if _entered_main_hall(snap):
                found_indoor = info
                report["trigger"] = info
                print(f"  BFS MAIN HALL via {new_path} xy={info['xy']}")
                _shot(env, out / "bfs_main_hall.png")
                queue.clear()
                break
            if _reentered_secret(snap):
                # Known trap — do not expand or treat as progress.
                continue
            if _entered_progress_indoor(snap):
                if found_indoor is None:
                    found_indoor = info
                    print(
                        f"  BFS indoor {info['room']} via {new_path} "
                        f"xy={info['xy']}"
                    )
                    _shot(env, out / f"bfs_indoor_{info['room']}.png")
                # Don't expand from indoors for outdoor BFS
                continue

            if not snap.has_control:
                continue

            key = (snap.link_x // quant, snap.link_y // quant)
            if key in visited:
                continue
            visited.add(key)
            points.append([snap.link_x, snap.link_y])
            extremes["min_x"] = min(extremes["min_x"], snap.link_x)
            extremes["max_x"] = max(extremes["max_x"], snap.link_x)
            extremes["min_y"] = min(extremes["min_y"], snap.link_y)
            extremes["max_y"] = max(extremes["max_y"], snap.link_y)

            if _escaped_pocket(snap) and found_escape is None:
                found_escape = info
                print(f"  BFS ESCAPE via {new_path} xy={info['xy']}")
                _shot(env, out / "bfs_escape.png")

            # Keep exploring even after escape (want door)
            child_state = capture_pocket_state(env)
            queue.append((child_state, new_path, snap.link_x, snap.link_y))

    report["expanded"] = expanded
    report["visited"] = len(visited)
    report["points"] = points
    report["extremes"] = extremes
    report["escape"] = found_escape
    report["indoor"] = found_indoor
    report["ok"] = bool(found_indoor or found_escape)
    if found_indoor and found_indoor.get("main_hall"):
        report["tier_result"] = "trigger"
    elif found_indoor:
        report["tier_result"] = "trigger_other_room"
    elif found_escape:
        report["tier_result"] = "route"
    else:
        report["tier_result"] = "stuck"

    # Final overview shot from farthest west point if any
    if points:
        west = min(points, key=lambda p: p[0])
        north = min(points, key=lambda p: p[1])
        south = max(points, key=lambda p: p[1])
        report["westmost"] = west
        report["northmost"] = north
        report["southmost"] = south
        print(
            f"  BFS extremes x=[{extremes['min_x']},{extremes['max_x']}] "
            f"y=[{extremes['min_y']},{extremes['max_y']}] "
            f"visited={len(visited)} result={report['tier_result']}"
        )
    return report


def tier_wall_follow(
    env: object, pocket_state: bytes, out: Path
) -> dict[str, Any]:
    """Right-hand and left-hand wall followers with bush cutting."""
    report: dict[str, Any] = {"tier": "wall_follow", "runs": []}
    # Order of facings for right-hand rule (turn right preferred).
    facings = ["UP", "RIGHT", "DOWN", "LEFT"]

    def run_follow(hand: str, max_steps: int = 250) -> dict[str, Any]:
        restore_pocket(env, pocket_state)
        # Prefer starting by stepping off pad south first.
        primitives.run_script(env, ((("DOWN",), 24),))
        facing_i = facings.index("DOWN")
        trail: list[list[int]] = []
        for step in range(max_steps):
            snap = snapshot_env(env)
            trail.append([snap.link_x, snap.link_y])
            if _entered_main_hall(snap):
                return {
                    "hand": hand,
                    "ok": True,
                    "tier_result": "trigger",
                    "steps": step,
                    "end": snapshot_to_diag(snap),
                    "trail_len": len(trail),
                }
            if _reentered_secret(snap):
                # Back out of stairs and keep following.
                step_frames(env, action_for("DOWN"), 40)
                primitives.settle_control(env, max_frames=60)
                facing_i = facings.index("DOWN")
                continue
            if _entered_progress_indoor(snap):
                return {
                    "hand": hand,
                    "ok": True,
                    "tier_result": "trigger_other_room",
                    "steps": step,
                    "end": snapshot_to_diag(snap),
                    "trail_len": len(trail),
                }
            if _escaped_pocket(snap) and step > 5:
                # Continue toward door but record escape
                pass

            # Try turn (right or left), then straight, then other, then back.
            if hand == "right":
                order = [1, 0, 3, 2]  # right, straight, left, back
            else:
                order = [3, 0, 1, 2]  # left, straight, right, back

            moved = False
            for turn in order:
                fi = (facing_i + turn) % 4
                face = facings[fi]
                before = (snap.link_x, snap.link_y)
                # swing then walk
                swing_cut(env, face, swings=1)
                step_frames(env, action_for(face), 8)
                snap2 = snapshot_env(env)
                if (snap2.link_x, snap2.link_y) != before:
                    facing_i = fi
                    moved = True
                    break
                # restore facing attempt didn't move — try next
                snap = snap2
            if not moved:
                # spin swing all directions
                for face in facings:
                    swing_cut(env, face, swings=2)
                step_frames(env, action_for(facings[facing_i]), 6)

            # transition settle
            s = snapshot_env(env)
            if s.submodule != 0 or s.game_mode not in (0x07, 0x09):
                for _ in range(40):
                    step_frames(env, no_action(), 2)
                    s = snapshot_env(env)
                    if s.has_control:
                        break

        end = snapshot_env(env)
        _shot(env, out / f"wall_{hand}_end.png")
        return {
            "hand": hand,
            "ok": _escaped_pocket(end) or _entered_progress_indoor(end),
            "tier_result": (
                "trigger"
                if _entered_main_hall(end)
                else (
                    "trigger_other_room"
                    if _entered_progress_indoor(end)
                    else ("route" if _escaped_pocket(end) else "stuck")
                )
            ),
            "steps": max_steps,
            "end": snapshot_to_diag(end),
            "escaped": _escaped_pocket(end),
            "extremes": {
                "min_x": min(p[0] for p in trail),
                "max_x": max(p[0] for p in trail),
                "min_y": min(p[1] for p in trail),
                "max_y": max(p[1] for p in trail),
            },
            "trail_sample": trail[:: max(1, len(trail) // 20)],
        }

    for hand in ("right", "left"):
        print(f"  wall-follow {hand}...")
        run = run_follow(hand)
        report["runs"].append(run)
        print(
            f"    result={run['tier_result']} end_xy="
            f"({run['end'].get('link_x')},{run['end'].get('link_y')}) "
            f"room={run['end'].get('room_hex')}"
        )
        if run.get("tier_result") in {"trigger", "trigger_other_room", "route"}:
            report["ok"] = True
            report["winner"] = run
            if run["tier_result"] != "route":
                return report

    report["ok"] = any(r.get("ok") for r in report["runs"])
    report["tier_result"] = (
        report.get("winner", {}).get("tier_result")
        or max(
            (r.get("tier_result") or "stuck" for r in report["runs"]),
            key=lambda t: {"trigger": 3, "trigger_other_room": 2, "route": 1, "stuck": 0}[t],
        )
    )
    return report


def tier_approach_from_escape(
    env: object,
    pocket_state: bytes,
    escape_script: list[tuple[tuple[str, ...], int]],
    out: Path,
) -> dict[str, Any]:
    """After a known escape path, try door approach candidates."""
    report: dict[str, Any] = {"tier": "approach", "candidates": []}
    restore_pocket(env, pocket_state)
    esc = try_script(env, escape_script, cut_on_stuck=True)
    report["escape"] = esc
    if not esc["escaped_pocket"] and not esc["progress_indoor"]:
        report["ok"] = False
        report["blocker"] = "escape script failed"
        return report
    if esc["main_hall"]:
        report["ok"] = True
        report["tier_result"] = "trigger"
        report["winner"] = esc
        return report

    _shot(env, out / "approach_start.png")
    for name, x, y in DOOR_APPROACH_CANDIDATES:
        # Restore pocket + re-run escape for each door candidate.
        restore_pocket(env, pocket_state)
        try_script(env, escape_script, cut_on_stuck=True)
        wp = primitives.Waypoint(x, y, tolerance=16, label=name)
        # Outdoor move_to: room check disabled (room=None)
        res = primitives.move_to(env, wp, max_frames=800, stuck_cycles=40)
        # If stuck, swing and retry short
        if not res.ok:
            swing_cut(env, "UP", swings=3)
            swing_cut(env, "LEFT", swings=2)
            res2 = primitives.move_to(env, wp, max_frames=400, stuck_cycles=30)
            res = res2
        # Try walking into door (UP) if near candidate
        step_frames(env, action_for("UP"), 40)
        primitives.settle_control(env, max_frames=80)
        snap = snapshot_env(env)
        entry = {
            "name": name,
            "target": [x, y],
            "move_ok": res.ok,
            "move_reason": res.reason,
            "xy": [snap.link_x, snap.link_y],
            "main_hall": _entered_main_hall(snap),
            "progress_indoor": _entered_progress_indoor(snap),
            "reentered_secret": _reentered_secret(snap),
            "room": room_label(snap.room_base_id),
            "room_hex": f"0x{snap.room_base_id:02X}",
            "diag": snapshot_to_diag(snap),
        }
        report["candidates"].append(entry)
        print(
            f"  approach {name}: move={res.ok} prog={entry['progress_indoor']} "
            f"main={entry['main_hall']} xy={entry['xy']} room={entry['room']}"
        )
        _shot(env, out / f"approach_{name}.png")
        if entry["main_hall"]:
            report["ok"] = True
            report["tier_result"] = "trigger"
            report["winner"] = entry
            return report
        if entry["progress_indoor"]:
            report["ok"] = True
            report["tier_result"] = "trigger_other_room"
            report["winner"] = entry
            return report

    report["ok"] = False
    report["tier_result"] = "approach_miss"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=FIGHTER_SWORD_STATE)
    parser.add_argument(
        "--tier",
        choices=("all", "scripts", "bfs", "wall", "route"),
        default="all",
        help="Which probe tier to run (default all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help=f"Output dir (default {OUT_DIR})",
    )
    parser.add_argument("--bfs-nodes", type=int, default=350)
    parser.add_argument("--bfs-quant", type=int, default=8)
    args = parser.parse_args()
    _configure_headless()
    args.out.mkdir(parents=True, exist_ok=True)

    env = build_boot_env(args.state)
    report: dict[str, Any] = {
        "probe": "courtyard_main_door",
        "state": args.state,
        "tiers": {},
    }
    try:
        env.reset()
        print("=== reach courtyard pocket ===")
        pocket = reach_pocket(env)
        report["pocket"] = pocket
        print(
            f"  pocket ok={pocket['ok']} phase={pocket['phase']} "
            f"xy=({pocket['diag'].get('link_x')},{pocket['diag'].get('link_y')})"
        )
        _shot(env, args.out / "00_pocket.png")
        if not pocket["ok"]:
            report["ok"] = False
            report["blocker"] = pocket.get("blocker") or "failed to reach pocket"
            (args.out / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            return 1

        pocket_state = capture_pocket_state(env)
        # Also save a dev .state file for faster future probes
        try:
            state_path = (
                Path(__file__).resolve().parents[1]
                / "custom_integrations"
                / "Zelda3-Snes"
                / "CourtyardSecretPocket.state"
            )
            # Use integration's expected format via env if available
            if hasattr(env, "em") and hasattr(env.em, "get_state"):
                state_path.write_bytes(pocket_state)
                report["saved_pocket_state"] = str(state_path)
                print(f"  saved {state_path.name} ({len(pocket_state)} bytes)")
        except Exception as exc:  # noqa: BLE001
            report["saved_pocket_state_error"] = str(exc)

        tiers = args.tier
        run_scripts = tiers in {"all", "scripts", "route"}
        run_bfs = tiers in {"all", "bfs", "route"}
        run_wall = tiers in {"all", "wall", "route"}

        if run_scripts:
            print("=== tier: scripts ===")
            report["tiers"]["scripts"] = tier_scripts(env, pocket_state, args.out)

        if run_bfs:
            print("=== tier: bfs ===")
            report["tiers"]["bfs"] = tier_bfs(
                env,
                pocket_state,
                args.out,
                quant=args.bfs_quant,
                max_nodes=args.bfs_nodes,
            )

        if run_wall:
            print("=== tier: wall_follow ===")
            report["tiers"]["wall"] = tier_wall_follow(env, pocket_state, args.out)

        # If any tier found an escape path string, try approach candidates.
        escape_script: list[tuple[tuple[str, ...], int]] | None = None
        scripts_tier = report["tiers"].get("scripts") or {}
        if scripts_tier.get("escape_hits"):
            # Rebuild a coarse script from the winner name is hard; use a
            # strong generic west-north push if escape worked.
            winner = scripts_tier["escape_hits"][0]
            print(f"=== escape hit via {winner.get('name')} — approach tier ===")
            # Map known winners to scripts
            name = winner.get("name", "")
            name_to_script = {
                "S40_W200": [(("DOWN",), 40), (("LEFT",), 200)],
                "S40_W400": [(("DOWN",), 40), (("LEFT",), 400)],
                "LEFT_400": [(("LEFT",), 400)],
                "W300_N300": [(("LEFT",), 300), (("UP",), 300)],
                "S30_W300_N400": [
                    (("DOWN",), 30), (("LEFT",), 300), (("UP",), 400),
                ],
                "maze_S_W_W_N_castle": [
                    (("DOWN",), 50), (("LEFT",), 150),
                    (("LEFT",), 150), (("UP",), 200),
                    (("LEFT",), 80), (("UP",), 200),
                ],
            }
            escape_script = name_to_script.get(
                name,
                [(("DOWN",), 40), (("LEFT",), 300), (("UP",), 200)],
            )
            report["tiers"]["approach"] = tier_approach_from_escape(
                env, pocket_state, escape_script, args.out
            )
        elif (report["tiers"].get("bfs") or {}).get("escape"):
            # BFS path is a string of U/D/L/R — convert roughly
            path = report["tiers"]["bfs"]["escape"]["path"]
            print(f"=== bfs escape path {path} — converting ===")
            # Replay path as approach start already at escape; try door UP pushes
            restore_pocket(env, pocket_state)
            # Replay path tokens
            token_map = {
                "U": (("UP",), 10),
                "D": (("DOWN",), 10),
                "L": (("LEFT",), 10),
                "R": (("RIGHT",), 10),
            }
            i = 0
            script: list[tuple[tuple[str, ...], int]] = []
            while i < len(path):
                if path[i] == "s" and i + 1 < len(path):
                    # swing+nudge: skip swing encoding, just move
                    mv = path[i + 1]
                    if mv in token_map:
                        script.append(token_map[mv])
                    i += 2
                    continue
                if path[i] in token_map:
                    script.append(token_map[path[i]])
                i += 1
            report["tiers"]["approach"] = tier_approach_from_escape(
                env, pocket_state, script, args.out
            )

        # Aggregate
        any_ok = any(t.get("ok") for t in report["tiers"].values())
        report["ok"] = any_ok
        best = "stuck"
        rank = {
            "trigger": 4,
            "trigger_other_room": 3,
            "route": 2,
            "approach_miss": 1,
            "stuck": 0,
        }
        for t in report["tiers"].values():
            tr = t.get("tier_result") or ("route" if t.get("ok") else "stuck")
            if rank.get(tr, 0) > rank.get(best, 0):
                best = tr
        report["best_tier_result"] = best

    finally:
        env.close()

    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Compact summary print
    summary = {
        "ok": report.get("ok"),
        "best_tier_result": report.get("best_tier_result"),
        "pocket_xy": [
            report.get("pocket", {}).get("diag", {}).get("link_x"),
            report.get("pocket", {}).get("diag", {}).get("link_y"),
        ],
        "tiers": {
            k: {
                "ok": v.get("ok"),
                "tier_result": v.get("tier_result"),
                "extremes": v.get("extremes"),
                "winner": (
                    (v.get("winner") or {}).get("name")
                    or (v.get("winner") or {}).get("path")
                    or (v.get("winner") or {}).get("hand")
                ),
            }
            for k, v in report.get("tiers", {}).items()
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {report_path}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
