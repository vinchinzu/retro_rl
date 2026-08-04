"""Run the Level 1 entrance → east-room first-key segment.

Examples::

    # Isolated from the naturally-produced Level1Entrance.state checkpoint
    uv run python zelda_i/scripts/run_level1_first_key.py

    # Power-on → sword → Level 1 → first dungeon key, with no state load
    uv run python zelda_i/scripts/run_level1_first_key.py --natural-entry
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.chain import run_natural_to_level1
from zelda_i.level1 import (
    SEGMENT_MAX_FRAMES,
    Level1FirstKeyController,
    level1_first_key_success,
)
from zelda_i.overworld_nav import OverworldToLevel1Controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.sword_cave import SwordCaveController


def run_once(
    *,
    natural_entry: bool = False,
    max_frames: int = SEGMENT_MAX_FRAMES,
    tag: str = "level1_first_key",
    save_checkpoint: bool = False,
) -> dict:
    configure_headless()
    start_state = "NONE" if natural_entry else "Level1Entrance"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = Level1FirstKeyController()
    sword: SwordCaveController | None = None
    nav: OverworldToLevel1Controller | None = None
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        boot_frames = 0
        prefix_ok = True
        if natural_entry:
            obs, boot_frames, sword, nav, _end = run_natural_to_level1(env)
            prefix_ok = sword.success and nav.success
        else:
            obs, *_ = env.step(nes_idle_action())

        snap0 = read_snapshot(env.get_ram())
        entry = {
            "natural_entry": natural_entry,
            "boot_frames": boot_frames,
            "mode": snap0.mode,
            "level": snap0.level,
            "room": snap0.screen,
            "keys": snap0.keys,
            "health": snap0.health,
            "x": snap0.link_x,
            "y": snap0.link_y,
        }

        if prefix_ok:
            for _ in range(max_frames):
                obs, *_ = env.step(
                    controller.step(read_snapshot(env.get_ram())).action
                )
                if controller.success or controller.phase.name == "FAILED":
                    break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = prefix_ok and level1_first_key_success(ram)
        checkpoint = None
        if ok and save_checkpoint:
            checkpoint = str(
                save_state(env, GAME_DIR, GAME, "Level1FirstKey")
            )
        label = "natural" if natural_entry else "isolated"
        png = RECORDINGS_DIR / f"{tag}_{label}.png"
        save_rgb_png(obs, png)
        return {
            "ok": ok,
            "stage": "level1_first_key" if ok else "failed",
            "entry": entry,
            "prefix_ok": prefix_ok,
            "sword": sword.report() if sword else None,
            "nav": nav.report() if nav else None,
            "level1": controller.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "keys": snap.keys,
                "health": snap.health,
                "rupees": snap.rupees,
                "x": snap.link_x,
                "y": snap.link_y,
                "room_obj_count": snap.room_obj_count,
                "live_stalfos": sum(
                    1
                    for obj in snap.objects
                    if 1 <= obj.slot <= 10
                    and obj.type_id == 0x2A
                    and obj.hp > 0
                ),
            },
            "checkpoint": checkpoint,
            "screenshot": str(png),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--natural-entry",
        action="store_true",
        help="Boot from power-on instead of loading Level1Entrance.state",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Save successful endpoint as Level1FirstKey.state",
    )
    args = parser.parse_args(argv)

    reports = []
    for i in range(args.trials):
        report = run_once(
            natural_entry=args.natural_entry,
            max_frames=args.max_frames,
            tag=f"level1_first_key_t{i}",
            save_checkpoint=args.save_state,
        )
        reports.append(report)
        final = report["final"]
        print(
            f"trial={i} ok={report['ok']} prefix_ok={report['prefix_ok']} "
            f"room={final['room']:02X} keys={final['keys']} "
            f"level1_frames={report['level1']['frames']} "
            f"phase={report['level1']['phase']}"
        )

    label = "natural" if args.natural_entry else "isolated"
    out = RECORDINGS_DIR / f"level1_first_key_{label}.json"
    payload = {
        "segment": "level1_first_key",
        "natural_entry": args.natural_entry,
        "runtime_class": "bronze",
        "intervention_class": "clean",
        "trials": args.trials,
        "successes": sum(1 for report in reports if report["ok"]),
        "reports": reports,
    }
    write_json_report(out, payload)
    print(f"wrote {out}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
