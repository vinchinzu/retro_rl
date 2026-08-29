"""Run sword cave + overworld path to Level 1 entrance.

Examples::

    # Isolated from Level1.state (sword segment then overworld)
    uv run python zelda_i/scripts/run_to_level1.py

    # Natural entry: power-on boot, no state load
    uv run python zelda_i/scripts/run_to_level1.py --natural-entry

    # Stop on overworld 0x37 (do not require dungeon interior)
    uv run python zelda_i/scripts/run_to_level1.py --screen-only
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.route.chain import boot_to_ready
from zelda_i.overworld.nav import (
    SEGMENT_MAX_FRAMES,
    OverworldToLevel1Controller,
    level1_entrance_success,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import parse_game_state, read_snapshot
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.sword_cave import SwordCaveController, sword_segment_success

def run_once(
    *,
    natural_entry: bool = False,
    require_dungeon: bool = True,
    max_frames: int = SEGMENT_MAX_FRAMES,
    tag: str = "to_level1",
) -> dict:
    configure_headless()
    start_state = "NONE" if natural_entry else "Level1"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    sword = SwordCaveController()
    nav = OverworldToLevel1Controller(require_dungeon=require_dungeon)
    try:
        obs, _ = reset_obs(env)
        boot_frames = 0
        if natural_entry:
            obs, boot_frames = boot_to_ready(env)
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

        for _ in range(SWORD_MAX):
            obs, *_ = env.step(sword.step(read_snapshot(env.get_ram())).action)
            if sword.success or sword.phase.name == "FAILED":
                break

        ram = env.get_ram()
        sword_ok = bool(
            sword_segment_success(ram)
            or (sword.success and read_snapshot(ram).sword >= 1)
        )
        if not sword_ok:
            snap = read_snapshot(ram)
            png = RECORDINGS_DIR / f"{tag}_sword_fail.png"
            save_rgb_png(obs, png)
            return {
                "ok": False,
                "stage": "sword_cave",
                "entry": entry,
                "sword": sword.report(),
                "nav": nav.report(),
                "final": {
                    "mode": snap.mode,
                    "screen": snap.screen,
                    "level": snap.level,
                    "sword": snap.sword,
                    "x": snap.link_x,
                    "y": snap.link_y,
                },
                "screenshot": str(png),
            }

        # Nav EAST_77 aligns y≈140 from cave exit (~64,77); no fixed DOWN hold.
        for _ in range(max_frames):
            obs, *_ = env.step(nav.step(read_snapshot(env.get_ram())).action)
            if nav.success or nav.phase.name == "FAILED":
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        state = parse_game_state(ram, frame=sword.frames + nav.frames)
        ok = level1_entrance_success(ram, require_dungeon=require_dungeon)
        if not ok and nav.success:
            ok = True
        label = "natural" if natural_entry else "isolated"
        suffix = "dungeon" if require_dungeon else "screen"
        png = RECORDINGS_DIR / f"{tag}_{label}_{suffix}.png"
        save_rgb_png(obs, png)
        return {
            "ok": ok,
            "stage": "level1" if ok else "overworld",
            "entry": entry,
            "sword": sword.report(),
            "nav": nav.report(),
            "final": {
                "mode": snap.mode,
                "screen": snap.screen,
                "level": snap.level,
                "sword": snap.sword,
                "x": snap.link_x,
                "y": snap.link_y,
                "overworld": snap.overworld,
                "game_mode": state.mode.name,
            },
            "screenshot": str(png),
            "require_dungeon": require_dungeon,
        }
    finally:
        env.close()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--natural-entry",
        action="store_true",
        help="Boot from power-on instead of loading Level1.state",
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Success = reach overworld screen 0x37 (skip dungeon door)",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    args = parser.parse_args(argv)

    require_dungeon = not args.screen_only
    reports = []
    for i in range(args.trials):
        tag = f"to_level1_t{i}"
        rep = run_once(
            natural_entry=args.natural_entry,
            require_dungeon=require_dungeon,
            max_frames=args.max_frames,
            tag=tag,
        )
        reports.append(rep)
        fin = rep["final"]
        print(
            f"trial={i} ok={rep['ok']} stage={rep.get('stage')} "
            f"sword_frames={rep['sword']['frames']} nav_frames={rep['nav']['frames']} "
            f"screen={fin['screen']:02X} level={fin['level']} "
            f"phase={rep['nav']['phase']}"
        )

    label = "natural" if args.natural_entry else "isolated"
    suffix = "dungeon" if require_dungeon else "screen"
    out = RECORDINGS_DIR / f"to_level1_{label}_{suffix}.json"
    payload = {
        "segment": "to_level1",
        "natural_entry": args.natural_entry,
        "require_dungeon": require_dungeon,
        "trials": args.trials,
        "successes": sum(1 for r in reports if r["ok"]),
        "reports": reports,
    }
    write_json_report(out, payload)
    print(f"wrote {out}")
    return 0 if all(r["ok"] for r in reports) else 1

if __name__ == "__main__":
    raise SystemExit(main())
