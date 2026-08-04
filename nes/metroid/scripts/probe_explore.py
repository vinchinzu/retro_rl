"""Manual-ish exploration probe: scripted button holds while logging map cells.

Used to discover the real start→morph map path before hardening the controller.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metroid.menus import boot_to_level1_script
from metroid.paths import GAME, GAME_DIR, RECORDINGS_DIR
from metroid.ram import is_level1_ready, read_snapshot
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png

# Default open-explore macro after boot: walk/jump right and drop.
DEFAULT_PLAN = (
    ("RIGHT", 180),
    ("RIGHT", "A", 40),
    ("RIGHT", 120),
    ("DOWN", 60),
    ("RIGHT", 90),
    ("DOWN", 90),
    ("RIGHT", 120),
    ("DOWN", 120),
    ("LEFT", 60),
    ("RIGHT", 90),
    ("DOWN", 180),
    ("RIGHT", 180),
    ("LEFT", 90),
    ("DOWN", 120),
    ("RIGHT", 200),
    ("A", 20),  # sometimes fires B in NES layout — intentional probe
    ("RIGHT", 200),
    ("DOWN", 200),
    ("LEFT", 150),
    ("RIGHT", 150),
)


def _parse_plan(text: str) -> list[tuple]:
    """Parse 'RIGHT:60,RIGHT+A:30,DOWN:40' style plans."""
    plan: list[tuple] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            buttons, frames_s = token.rsplit(":", 1)
            frames = int(frames_s)
        else:
            buttons, frames = token, 30
        names = tuple(b for b in buttons.replace("+", ",").split(",") if b)
        if len(names) == 1:
            plan.append((names[0], frames))
        else:
            plan.append((*names, frames))
    return plan


def _boot(env) -> tuple[object, int]:
    frame = 0
    obs = None
    stable = 0
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
        if is_level1_ready(env.get_ram(), float(obs.mean())):
            stable += 1
            if stable >= 15:
                return obs, frame
        else:
            stable = 0
    return obs, frame


def run(*, plan_text: str | None = None, max_frames: int = 8000) -> int:
    configure_headless()
    plan = _parse_plan(plan_text) if plan_text else DEFAULT_PLAN
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        env.reset()
        obs, boot_frames = _boot(env)
        snap0 = read_snapshot(env.get_ram())
        events: list[dict] = [
            {
                "event": "boot",
                "frame": boot_frames,
                "map": list(snap0.map_cell),
                "xy": [snap0.samus_x, snap0.samus_y],
                "area": snap0.area,
                "status": snap0.samus_status,
                "missiles": snap0.missiles_enabled,
                "item_pause": snap0.item_pause,
            }
        ]
        last_cell = snap0.map_cell
        frame = boot_frames
        for step in plan:
            *buttons, hold = step
            action = nes_action(*buttons) if buttons else nes_idle_action()
            for _ in range(int(hold)):
                obs, *_ = env.step(action)
                frame += 1
                snap = read_snapshot(env.get_ram())
                if snap.map_cell != last_cell or snap.item_pause or snap.missiles_enabled:
                    events.append(
                        {
                            "frame": frame,
                            "map": list(snap.map_cell),
                            "xy": [snap.samus_x, snap.samus_y],
                            "dir": snap.samus_dir,
                            "door": snap.in_door,
                            "status": snap.samus_status,
                            "item_pause": snap.item_pause,
                            "missiles": snap.missiles_enabled,
                            "health": snap.health_units,
                            "buttons": list(buttons),
                        }
                    )
                    last_cell = snap.map_cell
                if frame >= max_frames:
                    break
            if frame >= max_frames:
                break

        snap = read_snapshot(env.get_ram())
        png = save_rgb_png(obs, RECORDINGS_DIR / "probe_explore.png")
        out = {
            "boot_frames": boot_frames,
            "final": {
                "map": list(snap.map_cell),
                "xy": [snap.samus_x, snap.samus_y],
                "status": snap.samus_status,
                "missiles": snap.missiles_enabled,
                "item_pause": snap.item_pause,
            },
            "events": events,
            "screenshot": str(png),
        }
        path = RECORDINGS_DIR / "probe_explore.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(
            f"done frames={frame} map={snap.map_cell} missiles={snap.missiles_enabled} "
            f"item_pause={snap.item_pause} events={len(events)} -> {path}"
        )
        return 0
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Comma plan e.g. RIGHT:120,DOWN:60,RIGHT+A:40",
    )
    parser.add_argument("--max-frames", type=int, default=8000)
    args = parser.parse_args()
    raise SystemExit(run(plan_text=args.plan, max_frames=args.max_frames))


if __name__ == "__main__":
    main()
