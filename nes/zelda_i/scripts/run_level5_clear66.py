"""Isolated pure: clear Level 5 room 0x66 (3× type 0x30 Gibdo) from entry.

Default start: ``L5_Room_66`` (already in room, south mouth).
``--from-entrance`` / ``Level5Entrance``: walk north from 0x76 then clear.

Stop: ``level5_room_66_cleared`` (3 dead, RoomAllDead≥20, east door bit 0x08).

Examples::

    uv run python nes/zelda_i/scripts/run_level5_clear66.py --trials 2
    uv run python nes/zelda_i/scripts/run_level5_clear66.py --from-entrance --save-state
    uv run python nes/zelda_i/scripts/run_level5_clear66.py --from-state Level5Entrance --trials 2
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
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_dungeon import (
    GIBDO_OBJECT_TYPE,
    ROOM_66_SPEC,
    level5_room_66_cleared,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot

def run_once(
    *,
    tag: str = "level5_clear66",
    save_checkpoint: bool = False,
    start_state: str = "L5_Room_66",
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = GenericDungeonRoomController(ROOM_66_SPEC)
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())

        for _ in range(ROOM_66_SPEC.max_frames):
            action = controller.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(action.action)
            if controller.success or controller.phase is DungeonPhase.FAILED:
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level5_room_66_cleared(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level5Cleared66")
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
                        "segment": "level5_clear66",
                        "natural_entry": False,
                        "start_state": start_state,
                    },
                    selected_trial=controller.report(),
                    natural_entry=False,
                )
            )
        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        live_gibdos = len(ROOM_66_SPEC.live_enemies(snap))
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
                "objects": [
                    {
                        "slot": o.slot,
                        "type": o.type_id,
                        "hp": o.hp,
                        "x": o.x,
                        "y": o.y,
                    }
                    for o in entry.objects
                    if 1 <= o.slot <= 10 and o.type_id == GIBDO_OBJECT_TYPE
                ],
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
                "live_gibdos": live_gibdos,
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
    parser.add_argument(
        "--from-state",
        default=None,
        help="Start state name (default L5_Room_66, or Level5Entrance with --from-entrance)",
    )
    parser.add_argument(
        "--from-entrance",
        action="store_true",
        help="Start from Level5Entrance (0x76) and walk north into 0x66",
    )
    args = parser.parse_args(argv)

    if args.from_state is not None:
        start_state = args.from_state
    elif args.from_entrance:
        start_state = "Level5Entrance"
    else:
        start_state = "L5_Room_66"

    reports = [
        run_once(
            tag=f"l5_clear66_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=start_state,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report["final"]
        print(
            f"trial={trial} ok={report['ok']} "
            f"room={final['room']:02X} live={final['live_gibdos']} "
            f"doors={final['cur_opened_doors']:#04x} "
            f"all_dead={final['room_all_dead']} "
            f"health={final['health']} "
            f"frames={report['controller']['frames']} "
            f"phase={report['controller']['phase']}"
        )

    tag_suffix = "entrance" if start_state == "Level5Entrance" else "isolated"
    output = RECORDINGS_DIR / f"l5_clear66_{tag_suffix}.json"
    write_json_report(
        output,
        {
            "segment": "level5_clear66",
            "natural_entry": False,
            "start_state": start_state,
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "track": "clean",
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "level5_room_66_cleared",
            "spec_id": ROOM_66_SPEC.spec_id,
            "enemy_type": GIBDO_OBJECT_TYPE,
            "enemy_type_hex": f"0x{GIBDO_OBJECT_TYPE:02x}",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report["ok"] for report in reports) else 1

if __name__ == "__main__":
    raise SystemExit(main())
