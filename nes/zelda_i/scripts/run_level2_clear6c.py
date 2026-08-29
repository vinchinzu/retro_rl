"""Isolated pure: clear Level 2 west key room 0x6c (6 Ropes + key 0x19).

Default start: ``Level2RopesCleared`` (0x6d cleared, left door open).
Optional ``--from-entrance`` runs 0x6d clear then 0x6c in one env (still
checkpoint-isolated from Level2Entrance, not natural OW).

Stop: ``level2_room_6c_key_success`` (cleared + keys≥1).

Examples::

    uv run python nes/zelda_i/scripts/run_level2_clear6c.py --trials 2
    uv run python nes/zelda_i/scripts/run_level2_clear6c.py --from-entrance --trials 2
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
)
from zelda_i.level2_dungeon import (
    ROOM_6C_SPEC,
    ROOM_6D_SPEC,
    level2_room_6c_key_success,
    level2_room_6d_cleared,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot

def _run_controller(env, controller, max_frames: int):
    obs = None
    for _ in range(max_frames):
        action = controller.step(read_snapshot(env.get_ram()))
        obs, *_ = env.step(action.action)
        if controller.success or controller.phase is DungeonPhase.FAILED:
            break
    return obs

def run_once(
    *,
    tag: str = "level2_clear6c",
    save_checkpoint: bool = False,
    from_entrance: bool = False,
    start_state: str | None = None,
) -> dict:
    configure_headless()
    if start_state is None:
        start_state = "Level2Entrance" if from_entrance else "Level2RopesCleared"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    clear6d: GenericDungeonRoomController | None = None
    controller = GenericDungeonRoomController(ROOM_6C_SPEC)
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())

        prefix_ok = True
        if from_entrance or entry.screen == 0x7D:
            clear6d = GenericDungeonRoomController(ROOM_6D_SPEC)
            obs = _run_controller(env, clear6d, ROOM_6D_SPEC.max_frames)
            prefix_ok = level2_room_6d_cleared(env.get_ram())
            if not prefix_ok:
                snap = read_snapshot(env.get_ram())
                screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
                save_rgb_png(obs, screenshot)
                return {
                    "ok": False,
                    "prefix_ok": False,
                    "from_entrance": from_entrance,
                    "start_state": start_state,
                    "error": "failed_clear_0x6d",
                    "clear6d": clear6d.report() if clear6d else None,
                    "entry": {
                        "room": entry.screen,
                        "x": entry.link_x,
                        "y": entry.link_y,
                        "keys": entry.keys,
                        "health": entry.health,
                    },
                    "final": {
                        "mode": snap.mode,
                        "level": snap.level,
                        "room": snap.screen,
                        "keys": snap.keys,
                        "room_all_dead": snap.room_all_dead,
                        "cur_opened_doors": snap.cur_opened_doors,
                    },
                    "screenshot": str(screenshot),
                }

        if prefix_ok:
            obs = _run_controller(env, controller, ROOM_6C_SPEC.max_frames)

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = prefix_ok and level2_room_6c_key_success(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level2WestKey")
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
                        "segment": "level2_clear6c",
                        "from_entrance": from_entrance,
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
            "from_entrance": from_entrance,
            "start_state": start_state,
            "prefix_ok": prefix_ok,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
                "health": entry.health,
                "doors": entry.cur_opened_doors,
            },
            "clear6d": clear6d.report() if clear6d else None,
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
                "live_ropes": len(ROOM_6C_SPEC.live_enemies(snap)),
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
        "--from-entrance",
        action="store_true",
        help="Start Level2Entrance and clear 0x6d then 0x6c",
    )
    parser.add_argument("--from-state", default=None)
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"level2_clear6c_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            from_entrance=args.from_entrance,
            start_state=args.from_state,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report.get("final") or {}
        print(
            f"trial={trial} ok={report.get('ok')} "
            f"prefix_ok={report.get('prefix_ok')} "
            f"room={final.get('room', 0):02X} keys={final.get('keys')} "
            f"live={final.get('live_ropes')} "
            f"frames={report.get('controller', {}).get('frames')} "
            f"phase={report.get('controller', {}).get('phase')}"
        )

    stem = (
        "level2_clear6c_from_entrance_isolated"
        if args.from_entrance
        else "level2_clear6c_isolated"
    )
    output = RECORDINGS_DIR / f"{stem}.json"
    write_json_report(
        output,
        {
            "segment": "level2_clear6c",
            "natural_entry": False,
            "from_entrance": args.from_entrance,
            "start_state": args.from_state
            or ("Level2Entrance" if args.from_entrance else "Level2RopesCleared"),
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "track": "clean",
            "trials": args.trials,
            "successes": sum(1 for report in reports if report.get("ok")),
            "stop_predicate": "level2_room_6c_key_success",
            "spec_id": ROOM_6C_SPEC.spec_id,
            "key": "fixed RoomItemId 0x19; pickup near (136,141)",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report.get("ok") for report in reports) else 1

if __name__ == "__main__":
    raise SystemExit(main())
