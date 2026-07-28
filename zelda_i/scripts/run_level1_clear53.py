"""Clear Level 1 room 0x53 and collect its fixed room key.

Examples::

    # Isolated from naturally-produced Level1Cleared63.state
    uv run python zelda_i/scripts/run_level1_clear53.py

    # Power-on → sword → first key → clear 0x63 → clear/key 0x53
    uv run python zelda_i/scripts/run_level1_clear53.py --natural-entry
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
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.chain import run_natural_to_level1
from zelda_i.level1 import (
    CLEAR_53_MAX_FRAMES,
    CLEAR_63_MAX_FRAMES,
    SEGMENT_MAX_FRAMES as FIRST_KEY_MAX_FRAMES,
)
from zelda_i.level1 import (
    STALFOS_OBJECT_TYPE,
    UNLOCK_NORTH_MAX_FRAMES,
    Level1Clear53Controller,
    Level1Clear63Controller,
    Level1FirstKeyController,
    Level1UnlockNorthController,
    level1_room_53_cleared,
)
from zelda_i.overworld_nav import OverworldToLevel1Controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.sword_cave import SwordCaveController


def _live_stalfos_count(snap) -> int:
    return sum(
        1
        for obj in snap.objects
        if 1 <= obj.slot <= 10
        and obj.type_id == STALFOS_OBJECT_TYPE
        and obj.hp > 0
    )


def run_once(
    *,
    natural_entry: bool = False,
    first_key_max_frames: int = FIRST_KEY_MAX_FRAMES,
    north_max_frames: int = UNLOCK_NORTH_MAX_FRAMES,
    clear63_max_frames: int = CLEAR_63_MAX_FRAMES,
    clear53_max_frames: int = CLEAR_53_MAX_FRAMES,
    tag: str = "level1_clear53",
    save_checkpoint: bool = False,
) -> dict:
    configure_headless()
    start_state = "NONE" if natural_entry else "Level1Cleared63"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    first_key: Level1FirstKeyController | None = None
    north: Level1UnlockNorthController | None = None
    clear63: Level1Clear63Controller | None = None
    clear53 = Level1Clear53Controller()
    sword: SwordCaveController | None = None
    nav: OverworldToLevel1Controller | None = None
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        boot_frames = 0
        prefix_ok = True
        if natural_entry:
            obs, boot_frames, sword, nav = run_natural_to_level1(env)
            prefix_ok = sword.success and nav.success
            first_key = Level1FirstKeyController()
            if prefix_ok:
                for _ in range(first_key_max_frames):
                    obs, *_ = env.step(
                        first_key.step(read_snapshot(env.get_ram())).action
                    )
                    if first_key.success or first_key.phase.name == "FAILED":
                        break
                prefix_ok = prefix_ok and first_key.success

            north = Level1UnlockNorthController()
            if prefix_ok:
                for _ in range(north_max_frames):
                    obs, *_ = env.step(
                        north.step(read_snapshot(env.get_ram())).action
                    )
                    if north.success or north.phase.name == "FAILED":
                        break
                prefix_ok = prefix_ok and north.success

            clear63 = Level1Clear63Controller()
            if prefix_ok:
                for _ in range(clear63_max_frames):
                    obs, *_ = env.step(
                        clear63.step(read_snapshot(env.get_ram())).action
                    )
                    if clear63.success or clear63.phase.name == "FAILED":
                        break
                prefix_ok = prefix_ok and clear63.success
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
            "rupees": snap0.rupees,
            "bombs": snap0.bombs,
            "health": snap0.health,
            "x": snap0.link_x,
            "y": snap0.link_y,
            "room_item_id": snap0.room_item_id,
            "live_stalfos": _live_stalfos_count(snap0),
        }

        if prefix_ok:
            for _ in range(clear53_max_frames):
                obs, *_ = env.step(
                    clear53.step(read_snapshot(env.get_ram())).action
                )
                if clear53.success or clear53.phase.name == "FAILED":
                    break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = prefix_ok and level1_room_53_cleared(ram)
        checkpoint = None
        if ok and save_checkpoint:
            checkpoint = str(save_state(env, GAME_DIR, GAME, "Level1Cleared53"))
        label = "natural" if natural_entry else "isolated"
        png = RECORDINGS_DIR / f"{tag}_{label}.png"
        save_rgb_png(obs, png)
        return {
            "ok": ok,
            "stage": "level1_room_53_cleared" if ok else "failed",
            "entry": entry,
            "prefix_ok": prefix_ok,
            "sword": sword.report() if sword else None,
            "nav": nav.report() if nav else None,
            "first_key": first_key.report() if first_key else None,
            "north": north.report() if north else None,
            "clear63": clear63.report() if clear63 else None,
            "clear53": clear53.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "keys": snap.keys,
                "rupees": snap.rupees,
                "bombs": snap.bombs,
                "health": snap.health,
                "x": snap.link_x,
                "y": snap.link_y,
                "room_item_id": snap.room_item_id,
                "room_all_dead": snap.room_all_dead,
                "room_obj_count": snap.room_obj_count,
                "cur_opened_doors": snap.cur_opened_doors,
                "open_doorway_mask": snap.open_doorway_mask,
                "live_stalfos": _live_stalfos_count(snap),
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
        help="Boot from power-on instead of loading Level1Cleared63.state",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--first-key-max-frames",
        type=int,
        default=FIRST_KEY_MAX_FRAMES,
    )
    parser.add_argument(
        "--north-max-frames",
        type=int,
        default=UNLOCK_NORTH_MAX_FRAMES,
    )
    parser.add_argument(
        "--clear63-max-frames",
        type=int,
        default=CLEAR_63_MAX_FRAMES,
    )
    parser.add_argument(
        "--clear53-max-frames",
        type=int,
        default=CLEAR_53_MAX_FRAMES,
    )
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Save successful endpoint as Level1Cleared53.state",
    )
    args = parser.parse_args(argv)

    reports = []
    for i in range(args.trials):
        report = run_once(
            natural_entry=args.natural_entry,
            first_key_max_frames=args.first_key_max_frames,
            north_max_frames=args.north_max_frames,
            clear63_max_frames=args.clear63_max_frames,
            clear53_max_frames=args.clear53_max_frames,
            tag=f"level1_clear53_t{i}",
            save_checkpoint=args.save_state,
        )
        reports.append(report)
        final = report["final"]
        print(
            f"trial={i} ok={report['ok']} prefix_ok={report['prefix_ok']} "
            f"room={final['room']:02X} live={final['live_stalfos']} "
            f"keys={final['keys']} all_dead={final['room_all_dead']} "
            f"clear53_frames={report['clear53']['frames']} "
            f"phase={report['clear53']['phase']}"
        )

    label = "natural" if args.natural_entry else "isolated"
    out = RECORDINGS_DIR / f"level1_clear53_{label}.json"
    payload = {
        "segment": "level1_clear53",
        "natural_entry": args.natural_entry,
        "runtime_class": "bronze",
        "intervention_class": "clean",
        "trials": args.trials,
        "successes": sum(1 for report in reports if report["ok"]),
        "reward": "small_key",
        "onward_doors": {
            "south": "0x63 open",
            "west": "0x52 open",
            "east": "0x54 open",
            "north": "closed",
        },
        "reports": reports,
    }
    write_json_report(out, payload)
    print(f"wrote {out}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
