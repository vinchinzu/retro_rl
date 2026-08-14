"""Isolated pure: clear Level 2 room 0x6d (5 Ropes) from Level2Entrance.

Uses ``ROOM_6D_SPEC`` / ``GenericDungeonRoomController``. Stop:
``level2_room_6d_cleared`` (RoomAllDead≥20, left door bit 0x02).

Examples::

    uv run python nes/zelda_i/scripts/run_level2_clear6d.py --trials 2
    uv run python nes/zelda_i/scripts/run_level2_clear6d.py --save-state
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
    ROOM_6D_SPEC,
    level2_room_6d_cleared,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot

def run_once(
    *,
    tag: str = "level2_clear6d",
    save_checkpoint: bool = False,
    start_state: str = "Level2Entrance",
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = GenericDungeonRoomController(ROOM_6D_SPEC)
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())

        for _ in range(ROOM_6D_SPEC.max_frames):
            action = controller.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(action.action)
            if controller.success or controller.phase is DungeonPhase.FAILED:
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level2_room_6d_cleared(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level2RopesCleared")
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
                        "segment": "level2_clear6d",
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
                "live_ropes": len(ROOM_6D_SPEC.live_enemies(snap)),
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
            tag=f"level2_clear6d_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report["final"]
        print(
            f"trial={trial} ok={report['ok']} "
            f"room={final['room']:02X} live={final['live_ropes']} "
            f"doors={final['cur_opened_doors']:#04x} "
            f"all_dead={final['room_all_dead']} "
            f"frames={report['controller']['frames']} "
            f"phase={report['controller']['phase']}"
        )

    output = RECORDINGS_DIR / "level2_clear6d_isolated.json"
    write_json_report(
        output,
        {
            "segment": "level2_clear6d",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "track": "clean",
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "level2_room_6d_cleared",
            "spec_id": ROOM_6D_SPEC.spec_id,
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report["ok"] for report in reports) else 1

if __name__ == "__main__":
    raise SystemExit(main())
