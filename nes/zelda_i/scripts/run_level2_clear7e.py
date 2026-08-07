"""Isolated pure: clear Level 2 east key room 0x7e (5 Ropes + key 0x19).

Default start: ``Level2Entrance`` (0x7d south spawn). Entry uses diamond-nav
waypoints (y≈157 wall-first, then y≈141 RIGHT) — not sealed-door, mid-room
solids block naive center-corridor RIGHT.

Stop: ``level2_room_7e_key_success`` (cleared + keys≥1).

Examples::

    uv run python nes/zelda_i/scripts/run_level2_clear7e.py --trials 2
    uv run python nes/zelda_i/scripts/run_level2_clear7e.py --save-state --trials 2
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
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
    ROOM_7E_SPEC,
    level2_room_7e_key_success,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


def run_once(
    *,
    tag: str = "level2_clear7e",
    save_checkpoint: bool = False,
    start_state: str = "Level2Entrance",
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = GenericDungeonRoomController(ROOM_7E_SPEC)
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())

        for _ in range(ROOM_7E_SPEC.max_frames):
            action = controller.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(action.action)
            if controller.success or controller.phase is DungeonPhase.FAILED:
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level2_room_7e_key_success(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level2EastKey")
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
                        "segment": "level2_clear7e",
                        "natural_entry": False,
                        "start_state": start_state,
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
                "live_ropes": len(ROOM_7E_SPEC.live_enemies(snap)),
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
    parser.add_argument("--from-state", default="Level2Entrance")
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"level2_clear7e_t{trial}",
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
            f"live={final['live_ropes']} "
            f"xy=({final['x']},{final['y']}) "
            f"frames={report['controller']['frames']} "
            f"phase={report['controller']['phase']} "
            f"max_live={report['controller']['max_live_enemies']}"
        )

    output = RECORDINGS_DIR / "level2_clear7e_isolated.json"
    write_json_report(
        output,
        {
            "segment": "level2_clear7e",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "track": "clean",
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "level2_room_7e_key_success",
            "spec_id": ROOM_7E_SPEC.spec_id,
            "entry_policy": "diamond-nav y≈157 wall-first then y≈141 RIGHT",
            "key": "fixed RoomItemId 0x19; pickup near (136,141)",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
