"""Isolated pure: Level 3 entry 0x7c west → 0x7b (6 Zols + key).

Clean track from ``Level3Entrance``. West door uses LEFT+UP diagonal residual
(see ``level3_dungeon.west_door_step``). Stop: ``level3_room_7b_key_success``.

Examples::

    uv run python nes/zelda_i/scripts/run_level3_west_key.py --trials 2
    uv run python nes/zelda_i/scripts/run_level3_west_key.py --save-state
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
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level3_dungeon import (
    ROOM_7B_SPEC,
    Level3WestKeyController,
    level3_room_7b_key_success,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


def run_once(
    *,
    tag: str = "level3_west_key",
    save_checkpoint: bool = False,
    start_state: str = "Level3Entrance",
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = Level3WestKeyController()
    max_frames = (
        controller.door.max_frames + ROOM_7B_SPEC.max_frames
    )
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())

        for _ in range(max_frames):
            action = controller.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(action.action)
            if controller.success or controller.phase == "failed":
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level3_room_7b_key_success(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level3WestKey")
            checkpoint = str(checkpoint_path)
            provenance = str(
                write_state_provenance(
                    checkpoint_path,
                    source_state_path=(
                        GAME_DIR
                        / "custom_integrations"
                        / GAME
                        / f"{start_state}.state"
                    ),
                    request={
                        "segment": "level3_west_key",
                        "natural_entry": False,
                        "start_state": start_state,
                        "intervention_class": "clean",
                    },
                    selected_trial=controller.report(),
                    natural_entry=False,
                )
            )
        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        return {
            "ok": ok,
            "natural_entry": False,
            "start_state": start_state,
            "intervention_class": "clean",
            "track": "clean",
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
                "health": entry.health,
                "doors": entry.cur_opened_doors,
            },
            "controller": controller.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "keys": snap.keys,
                "health": snap.health,
                "room_item_id": snap.room_item_id,
                "room_all_dead": snap.room_all_dead,
                "cur_opened_doors": snap.cur_opened_doors,
                "live_zols": len(ROOM_7B_SPEC.live_enemies(snap)),
            },
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(screenshot),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--from-state", default="Level3Entrance")
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"level3_west_key_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report["final"]
        print(
            f"trial={trial} ok={report['ok']} "
            f"room={final['room']:02X} keys={final['keys']} "
            f"live={final['live_zols']} "
            f"frames={report['controller']['frames']} "
            f"phase={report['controller']['phase']}"
        )

    output = RECORDINGS_DIR / "level3_west_key_isolated.json"
    write_json_report(
        output,
        {
            "segment": "level3_west_key",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "track": "clean",
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "level3_room_7b_key_success",
            "spec_id": ROOM_7B_SPEC.spec_id,
            "room_graph": {
                "0x7c": {
                    "role": "entry",
                    "west": "0x7b",
                    "west_policy": "LEFT+UP diagonal at wall after y≈149 band",
                },
                "0x7b": {
                    "role": "west_key",
                    "enemies": "6× Zol type 0x13 (TYPE_AND_HP)",
                    "item": "RoomItemId 0x19 small key",
                },
            },
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
