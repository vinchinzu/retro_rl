"""Isolated pure: Level 6 entry 0x79 → east 0x7a key collect.

Default start: ``Level6Entrance`` (room-ready 0x79). Uses
``Level6EntryRightController`` (no A while aligning) then
``GenericDungeonRoomController`` + ``ROOM_7A_SPEC`` for 5× type 0x24 + key 0x19.

Optional ``--from-state L6Room_7a`` skips the RIGHT hop.

Stop: ``level6_room_7a_key_success`` (keys≥1, no live 0x24).

Examples::

    uv run python nes/zelda_i/scripts/run_level6_east_key.py --trials 2
    uv run python nes/zelda_i/scripts/run_level6_east_key.py --save-state --trials 2
    uv run python nes/zelda_i/scripts/run_level6_east_key.py --from-state L6Room_7a
    uv run python nes/zelda_i/scripts/run_level6_east_key.py --infinite-life --trials 2
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
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level6_dungeon import (
    ROOM_7A_SPEC,
    Level6EastKeyController,
    level6_room_7a_key_success,
    make_east_key_controller,
)
from zelda_i.level6_overworld import (
    LEVEL6_EAST_KEY_ROOM,
    LEVEL6_ENTRY_ROOM,
    Level6EntryRightController,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot


def _run_right(env, max_frames: int = 4000, assist=None):
    controller = Level6EntryRightController(max_frames=max_frames)
    obs = None
    for frame in range(max_frames):
        if assist is not None:
            assist.apply_env(env, frame=frame)
        action = controller.step(read_snapshot(env.get_ram()))
        obs, *_ = env.step(action.action)
        if controller.success or controller.phase.name == "FAILED":
            break
    return obs, controller


def run_once(
    *,
    tag: str = "level6_east_key",
    save_checkpoint: bool = False,
    start_state: str = "Level6Entrance",
    infinite_life: bool = False,
    probe_doors: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    right_ctl: Level6EntryRightController | None = None
    controller: Level6EastKeyController = make_east_key_controller()
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())

        prefix_ok = True
        if entry.screen == LEVEL6_ENTRY_ROOM and entry.mode == PLAY_MODE:
            obs, right_ctl = _run_right(env, assist=assist)
            mid = read_snapshot(env.get_ram())
            prefix_ok = (
                mid.level == 6
                and mid.screen == LEVEL6_EAST_KEY_ROOM
                and mid.mode == PLAY_MODE
            )
            if not prefix_ok:
                screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
                save_rgb_png(obs, screenshot)
                return {
                    "ok": False,
                    "prefix_ok": False,
                    "start_state": start_state,
                    "error": "failed_entry_right_to_0x7a",
                    "entry": {
                        "room": entry.screen,
                        "x": entry.link_x,
                        "y": entry.link_y,
                        "keys": entry.keys,
                        "health": entry.health,
                    },
                    "right": right_ctl.report() if right_ctl else None,
                    "final": {
                        "mode": mid.mode,
                        "level": mid.level,
                        "room": mid.screen,
                        "x": mid.link_x,
                        "y": mid.link_y,
                        "keys": mid.keys,
                        "health": mid.health,
                    },
                    "screenshot": str(screenshot),
                    "assist": assist.report() if assist else None,
                }
        elif entry.screen != LEVEL6_EAST_KEY_ROOM:
            return {
                "ok": False,
                "prefix_ok": False,
                "start_state": start_state,
                "error": f"unexpected_start_room_0x{entry.screen:02x}",
                "entry": {
                    "room": entry.screen,
                    "x": entry.link_x,
                    "y": entry.link_y,
                    "keys": entry.keys,
                },
            }

        if prefix_ok:
            obs = None
            for frame in range(ROOM_7A_SPEC.max_frames):
                if assist is not None:
                    assist.apply_env(env, frame=frame)
                action = controller.step(read_snapshot(env.get_ram()))
                obs, *_ = env.step(action.action)
                if controller.success or controller.phase is DungeonPhase.FAILED:
                    break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = prefix_ok and level6_room_7a_key_success(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level6EastKey")
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
                        "segment": "level6_east_key",
                        "start_state": start_state,
                        "infinite_life": infinite_life,
                    },
                    selected_trial=controller.report(),
                    natural_entry=False,
                )
            )

        door_probe = None
        if ok and probe_doors:
            # Idle a few frames then sample open doors / room item for graph.
            for frame in range(30):
                if assist is not None:
                    assist.apply_env(env, frame=10000 + frame)
                obs, *_ = env.step(nes_idle_action())
            after = read_snapshot(env.get_ram())
            door_probe = {
                "room": after.screen,
                "keys": after.keys,
                "cur_opened_doors": after.cur_opened_doors,
                "open_doorway_mask": after.open_doorway_mask,
                "room_item_id": after.room_item_id,
                "room_all_dead": after.room_all_dead,
                "x": after.link_x,
                "y": after.link_y,
            }

        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        return {
            "ok": ok,
            "natural_entry": False,
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
            "right": right_ctl.report() if right_ctl else None,
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
                "live_wizzrobes": len(ROOM_7A_SPEC.live_enemies(snap)),
            },
            "door_probe": door_probe,
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(screenshot),
            "assist": assist.report() if assist else None,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--from-state", default="Level6Entrance")
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (not Clean STATUS)",
    )
    parser.add_argument(
        "--probe-doors",
        action="store_true",
        help="After success, sample open-door bits for graph notes",
    )
    args = parser.parse_args(argv)

    track = "assisted" if args.infinite_life else "clean"
    reports = [
        run_once(
            tag=f"level6_east_key_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            probe_doors=args.probe_doors and trial == 0,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report.get("final") or {}
        print(
            f"trial={trial} ok={report.get('ok')} "
            f"prefix_ok={report.get('prefix_ok')} "
            f"room={final.get('room', 0):02X} keys={final.get('keys')} "
            f"live={final.get('live_wizzrobes')} "
            f"xy=({final.get('x')},{final.get('y')}) "
            f"frames={report.get('controller', {}).get('frames')} "
            f"phase={report.get('controller', {}).get('phase')} "
            f"max_live={report.get('controller', {}).get('max_live_enemies')}"
        )

    stem = (
        "level6_east_key_assisted_isolated"
        if args.infinite_life
        else "level6_east_key_isolated"
    )
    output = RECORDINGS_DIR / f"{stem}.json"
    write_json_report(
        output,
        {
            "segment": "level6_east_key",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": "survival" if args.infinite_life else "clean",
            "track": track,
            "trials": args.trials,
            "successes": sum(1 for report in reports if report.get("ok")),
            "stop_predicate": "level6_room_7a_key_success",
            "spec_id": ROOM_7A_SPEC.spec_id,
            "entry_policy": "Level6EntryRightController y≈157→x≈208→y≈144 RIGHT",
            "key": "fixed RoomItemId 0x19; 5× type 0x24; pickup near (136,141)",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report.get("ok") for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
