"""Clear the eight Keese in Level 1 room 0x54.

This is the first segment using the data-driven ``DungeonRoomSpec`` controller
and reusable natural milestone chain.
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
from zelda_i.chain import run_natural_to_milestone
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
    ROOM_54_SPEC,
    dungeon_room_cleared,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


def run_once(
    *,
    natural_entry: bool = False,
    tag: str = "level1_clear54",
    save_checkpoint: bool = False,
) -> dict:
    configure_headless()
    start_state = "NONE" if natural_entry else "Level1Cleared53"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    prefix = None
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        if natural_entry:
            prefix = run_natural_to_milestone(env, milestone="clear53")
            obs = prefix.obs
            prefix_ok = prefix.success
        else:
            obs, *_ = env.step(nes_idle_action())
            prefix_ok = True

        entry = read_snapshot(env.get_ram())
        if prefix_ok:
            for _ in range(ROOM_54_SPEC.max_frames):
                action = controller.step(read_snapshot(env.get_ram()))
                obs, *_ = env.step(action.action)
                if controller.success or controller.phase is DungeonPhase.FAILED:
                    break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = prefix_ok and dungeon_room_cleared(ram, ROOM_54_SPEC)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level1Cleared54")
            checkpoint = str(checkpoint_path)
            provenance = str(
                write_state_provenance(
                    checkpoint_path,
                    source_state_path=(
                        None
                        if natural_entry
                        else GAME_DIR
                        / "custom_integrations"
                        / GAME
                        / "Level1Cleared53.state"
                    ),
                    request={
                        "segment": "level1_clear54",
                        "natural_entry": natural_entry,
                    },
                    selected_trial=controller.report(),
                    natural_entry=natural_entry,
                )
            )
        label = "natural" if natural_entry else "isolated"
        screenshot = RECORDINGS_DIR / f"{tag}_{label}.png"
        save_rgb_png(obs, screenshot)
        return {
            "ok": ok,
            "natural_entry": natural_entry,
            "prefix_ok": prefix_ok,
            "prefix": prefix.report() if prefix else None,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
                "health": entry.health,
            },
            "controller": controller.report(),
            "final": {
                "mode": snap.mode,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "keys": snap.keys,
                "rupees": snap.rupees,
                "bombs": snap.bombs,
                "health": snap.health,
                "room_item_id": snap.room_item_id,
                "room_all_dead": snap.room_all_dead,
                "live_keese": len(ROOM_54_SPEC.live_enemies(snap)),
            },
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(screenshot),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-entry", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    args = parser.parse_args(argv)

    reports = [
        run_once(
            natural_entry=args.natural_entry,
            tag=f"level1_clear54_t{trial}",
            save_checkpoint=args.save_state,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report["final"]
        print(
            f"trial={trial} ok={report['ok']} prefix_ok={report['prefix_ok']} "
            f"room={final['room']:02X} live={final['live_keese']} "
            f"frames={report['controller']['frames']} "
            f"phase={report['controller']['phase']}"
        )

    label = "natural" if args.natural_entry else "isolated"
    output = RECORDINGS_DIR / f"level1_clear54_{label}.json"
    write_json_report(
        output,
        {
            "segment": "level1_clear54",
            "natural_entry": args.natural_entry,
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "reward": "no known inventory change; RoomItemId=0x16",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
