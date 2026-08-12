#!/usr/bin/env python3
"""Navigate to upper-right outdoor mountain pond and dump tiles/screenshots.

Goal: lock coordinates for the light-blue spa pond next to the wooden shed
(not west cave 0x29, not mid-right tent camp pond).

Examples:

    HEADLESS=1 uv run python -m harvest.scripts.outdoor_spa_probe
    HEADLESS=1 uv run python -m harvest.scripts.outdoor_spa_probe \\
      --state latest_backup_sunday_go_to_mountain_20260427_152011
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.maps.map_config import ROUTES
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.nav import make_action
from harvest.tasks.hot_spring import MOUNTAIN_TILEMAP, read_player_action, read_stamina

DEFAULT_STATE = "latest_backup_sunday_go_to_mountain_20260427_152011"
OUT_DIR = PROJECT_DIR / "recordings" / "spa_outdoor_true"


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default=DEFAULT_STATE)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--nav-timeout", type=int, default=14000)
    return p.parse_args()


def _snap(ram, frame: int, label: str = "") -> dict:
    px = int(read_ram_value(ram, "player_x"))
    py = int(read_ram_value(ram, "player_y"))
    return {
        "frame": frame,
        "label": label,
        "tilemap": int(read_ram_value(ram, "tilemap")),
        "stamina": int(read_stamina(ram)),
        "px": px,
        "py": py,
        "tx": px // 16,
        "ty": py // 16,
        "action": int(read_player_action(ram)),
        "game_state": int(read_ram_value(ram, "game_state")),
        "hour": int(read_ram_value(ram, "hour")),
        "minute": int(read_ram_value(ram, "minute")),
    }


def _save_png(obs, path: Path) -> None:
    if obs is None:
        return
    arr = np.asarray(obs)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        Image.fromarray(arr[..., :3].astype("uint8")).save(path)


def _dump_tiles(ram, path: Path, radius: int = 12) -> None:
    px = int(read_ram_value(ram, "player_x"))
    py = int(read_ram_value(ram, "player_y"))
    tx, ty = px // 16, py // 16
    lines = [f"player px=({px},{py}) tile=({tx},{ty})"]
    # Best-effort raw map array (64-wide rows is common in this ROM).
    map_addr = 0x09B6  # !current_map_array in decomp RamMap (verify if needed)
    try:
        from harvest.core.ram_catalog import field_spec

        map_addr = field_spec("current_map_array").address
    except Exception:
        pass
    width = 64
    for y in range(max(0, ty - radius), ty + radius + 1):
        row = []
        for x in range(max(0, tx - radius), tx + radius + 1):
            idx = map_addr + y * width + x
            if 0 <= idx < len(ram):
                val = int(ram[idx])
                mark = "*" if (x, y) == (tx, ty) else " "
                row.append(f"{val:02X}{mark}")
            else:
                row.append("?? ")
        lines.append(f"Y{y:02d} " + " ".join(row))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    _configure_headless()
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = make_harvest_env(state=args.state, render_mode="rgb_array")
    obs, info = env.reset()
    ram = env.get_ram()
    snaps: list[dict] = []
    snaps.append(_snap(ram, 0, "start"))
    _save_png(obs, args.out / "00_start.png")
    print(f"[SPA] start {_snap(ram, 0)}")

    tilemap0 = int(read_ram_value(ram, "tilemap"))
    if tilemap0 == MOUNTAIN_TILEMAP:
        waypoints = list(
            ROUTES.get("mountain_entry_to_outdoor_spa")
            or ROUTES.get("mountain_entry_to_spa")
            or []
        )
        waypoints = [w for w in waypoints if int(getattr(w, "tilemap", 0x10)) == 0x10]
    else:
        # Farm / path / house: full multi-map route.
        waypoints = list(ROUTES.get("farm_to_spa") or [])
        # Exit house/shed first is owned by HotSpringStaminaTask; this probe
        # expects an outdoor farm start when not already on mountain.
    print(f"[SPA] route hops={len(waypoints)} from map=0x{tilemap0:02X}")

    if not waypoints:
        print("[SPA] no outdoor spa route")
        return 1

    nav = MultiMapNavTask(
        name="to_outdoor_spa",
        waypoints=waypoints,
        timeout=args.nav_timeout,
        initial_settle_frames=15,
    )
    world = WorldState(frame=0, ram=ram, info=info or {}, obs=obs)
    nav.reset(world)
    frame = 0
    status = TaskStatus.RUNNING
    while status == TaskStatus.RUNNING and frame < args.nav_timeout:
        result = nav.step(world)
        status = result.status
        action = make_action()
        if result.action is not None:
            action = getattr(result.action, "action", result.action)
        step = env.step(action)
        obs = step[0]
        info = step[4] if len(step) > 4 else {}
        ram = env.get_ram()
        world = WorldState(frame=frame, ram=ram, info=info or {}, obs=obs)
        frame += 1
        if frame % 200 == 0:
            s = _snap(ram, frame, f"nav_{frame}")
            snaps.append(s)
            print(
                f"[SPA] nav f={frame} map=0x{s['tilemap']:02X} "
                f"tile=({s['tx']},{s['ty']}) px=({s['px']},{s['py']})"
            )

    snaps.append(_snap(ram, frame, "nav_end"))
    _save_png(obs, args.out / "01_nav_end.png")
    print(f"[SPA] nav done status={status} f={frame} {_snap(ram, frame)}")

    # Nudge around provisional stand, screenshot + try B/A at each stop.
    sequences = [
        ("n20", {"up": True}, 20),
        ("e15", {"right": True}, 15),
        ("n15", {"up": True}, 15),
        ("e20", {"right": True}, 20),
        ("n20b", {"up": True}, 20),
        ("w10", {"left": True}, 10),
        ("n15b", {"up": True}, 15),
        ("e15b", {"right": True}, 15),
        ("s8", {"down": True}, 8),
        ("e10", {"right": True}, 10),
        ("n10", {"up": True}, 10),
        ("w20", {"left": True}, 20),
        ("n25", {"up": True}, 25),
        ("e25", {"right": True}, 25),
        ("final", {}, 8),
    ]
    for i, (label, btn, n) in enumerate(sequences, start=2):
        for _ in range(n):
            step = env.step(make_action(**btn))
            obs = step[0]
            ram = env.get_ram()
            frame += 1
        s = _snap(ram, frame, label)
        snaps.append(s)
        _save_png(obs, args.out / f"{i:02d}_{label}.png")
        print(
            f"[SPA] {label} tile=({s['tx']},{s['ty']}) px=({s['px']},{s['py']}) "
            f"map=0x{s['tilemap']:02X} stam={s['stamina']}"
        )

    # At final stand: face each direction + B-alone + A, log action/stam.
    stam0 = int(read_stamina(ram))
    interact_log = []
    for face in ("up", "right", "left", "down"):
        for _ in range(8):
            step = env.step(make_action(**{face: True}))
            obs = step[0]
            ram = env.get_ram()
            frame += 1
        for _ in range(4):
            step = env.step(make_action())
            ram = env.get_ram()
            frame += 1
        acts = []
        for _ in range(12):
            step = env.step(make_action(b=True))
            ram = env.get_ram()
            acts.append(int(read_player_action(ram)))
            frame += 1
        for _ in range(20):
            step = env.step(make_action())
            ram = env.get_ram()
            frame += 1
        acts_a = []
        for _ in range(8):
            step = env.step(make_action(**{face: True}))
            ram = env.get_ram()
            frame += 1
        for _ in range(4):
            step = env.step(make_action())
            ram = env.get_ram()
            frame += 1
        for _ in range(10):
            step = env.step(make_action(a=True))
            ram = env.get_ram()
            acts_a.append(int(read_player_action(ram)))
            frame += 1
        for _ in range(16):
            step = env.step(make_action())
            ram = env.get_ram()
            frame += 1
        stam1 = int(read_stamina(ram))
        entry = {
            "face": face,
            "stam0": stam0,
            "stam1": stam1,
            "b_acts": acts,
            "a_acts": acts_a,
            "saw_jump": 3 in acts,
            "pos": _snap(ram, frame, f"interact_{face}"),
        }
        interact_log.append(entry)
        _save_png(obs, args.out / f"interact_{face}.png")
        print(
            f"[SPA] interact face={face} stam {stam0}->{stam1} "
            f"jump={entry['saw_jump']} b_peak={max(acts) if acts else -1}"
        )
        stam0 = stam1

    _dump_tiles(ram, args.out / "tiles_end.txt")
    report = {
        "snaps": snaps,
        "nav_status": str(status),
        "frames": frame,
        "interact": interact_log,
        "final": _snap(ram, frame, "final"),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[SPA] wrote {args.out}")
    env.close()
    return 0 if status == TaskStatus.SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
