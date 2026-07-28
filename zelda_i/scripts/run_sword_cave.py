"""Run the wooden sword cave segment (M3 / natural-entry).

Examples::

    # Isolated from Level1.state
    uv run python zelda_i/scripts/run_sword_cave.py

    # Natural entry: power-on boot then segment (no state load)
    uv run python zelda_i/scripts/run_sword_cave.py --natural-entry

    # Multiple trials
    uv run python zelda_i/scripts/run_sword_cave.py --trials 3
"""

# Script execution adds the repository root before importing local packages.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env
from retro_harness.nes import nes_idle_action
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.menus import boot_to_level1_script
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import is_level1_ready, parse_game_state, read_snapshot
from zelda_i.sword_cave import (
    SEGMENT_MAX_FRAMES,
    SwordCaveController,
    sword_segment_success,
)


def _boot_to_ready(env) -> tuple[object, int]:
    frame = 0
    obs = None
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
        if is_level1_ready(env.get_ram(), obs_mean=float(obs.mean())):
            return obs, frame
    return obs, frame


def run_once(
    *,
    natural_entry: bool = False,
    max_frames: int = SEGMENT_MAX_FRAMES,
    tag: str = "sword_cave",
) -> dict:
    configure_headless()
    start_state = "NONE" if natural_entry else "Level1"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = SwordCaveController()
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        boot_frames = 0
        if natural_entry:
            obs, boot_frames = _boot_to_ready(env)
        else:
            obs, *_ = env.step(nes_idle_action())

        snap0 = read_snapshot(env.get_ram())
        entry = {
            "natural_entry": natural_entry,
            "mode": snap0.mode,
            "screen": snap0.screen,
            "sword": snap0.sword,
            "x": snap0.link_x,
            "y": snap0.link_y,
            "boot_frames": boot_frames,
        }

        for _ in range(max_frames):
            ram = env.get_ram()
            snap = read_snapshot(ram)
            action = controller.step(snap)
            obs, *_ = env.step(action.action)
            if controller.success or controller.phase.name == "FAILED":
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        state = parse_game_state(ram, frame=controller.frames)
        ok = bool(
            sword_segment_success(ram)
            or (controller.success and snap.sword >= 1 and snap.overworld)
        )
        png = RECORDINGS_DIR / f"{tag}_{'natural' if natural_entry else 'isolated'}.png"
        save_rgb_png(obs, png)
        report = {
            "ok": ok,
            "entry": entry,
            "controller": controller.report(),
            "final": {
                "mode": snap.mode,
                "screen": snap.screen,
                "sword": snap.sword,
                "x": snap.link_x,
                "y": snap.link_y,
                "overworld": snap.overworld,
                "in_cave": snap.in_cave,
                "game_mode": state.mode.name,
            },
            "screenshot": str(png),
        }
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--natural-entry",
        action="store_true",
        help="Boot from power-on instead of loading Level1.state",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    args = parser.parse_args(argv)

    reports = []
    for i in range(args.trials):
        tag = f"sword_cave_t{i}"
        rep = run_once(
            natural_entry=args.natural_entry,
            max_frames=args.max_frames,
            tag=tag,
        )
        reports.append(rep)
        print(
            f"trial={i} ok={rep['ok']} frames={rep['controller']['frames']} "
            f"sword={rep['final']['sword']} screen={rep['final']['screen']:02X} "
            f"phase={rep['controller']['phase']}"
        )

    out = RECORDINGS_DIR / (
        "sword_cave_natural.json" if args.natural_entry else "sword_cave_isolated.json"
    )
    payload = {
        "segment": "sword_cave",
        "natural_entry": args.natural_entry,
        "trials": args.trials,
        "successes": sum(1 for r in reports if r["ok"]),
        "reports": reports,
    }
    write_json_report(out, payload)
    print(f"wrote {out}")
    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
