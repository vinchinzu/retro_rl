"""Arrive Level 5 room 0x56 (north of 0x66) from Level5EastKey.

Documented hop after the east key: 0x77 LEFT → 0x76 UP → 0x66 free UP →
0x56. Arrival only (no combat clear). Does not poke keys or doors.

Examples::

    uv run python nes/zelda_i/scripts/run_level5_north56.py \
        --from-state Level5EastKey --keep-keys --infinite-life --save-state --trials 1
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
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_dungeon import (
    GIBDO_OBJECT_TYPE,
    ROOM_L5_NORTH_56,
    level5_room_56_arrived,
)
from zelda_i.level5_path import make_west65_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_CANDLE, ADDR_MAP, ADDR_WHISTLE, read_snapshot, read_u8


def _track_labels(infinite_life: bool) -> tuple[str, str]:
    if infinite_life:
        return "assisted", "survival"
    return "clean", "clean"


def _objects(snap) -> list[dict]:
    out = []
    for obj in snap.objects:
        if not (1 <= obj.slot <= 12) or obj.type_id in (0, 0xFF):
            continue
        out.append(
            {
                "slot": obj.slot,
                "type": obj.type_id,
                "type_hex": f"0x{obj.type_id:02x}",
                "name": object_name(obj.type_id),
                "hp": obj.hp,
                "x": obj.x,
                "y": obj.y,
            }
        )
    return out


def run_once(
    *,
    tag: str = "level5_north56",
    save_checkpoint: bool = False,
    start_state: str = "Level5EastKey",
    infinite_life: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    controller = make_west65_controller()
    track, intervention_class = _track_labels(infinite_life)
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = read_snapshot(env.get_ram())
        trail: list[dict] = []
        last_room = entry.screen

        for frame in range(controller.max_frames):
            if assist is not None:
                assist.apply_env(env, frame=frame)
            snap = read_snapshot(env.get_ram())
            action = controller.step(snap)
            obs, *_ = env.step(action.action)
            after = read_snapshot(env.get_ram())
            if after.screen != last_room:
                trail.append(
                    {
                        "event": "transition",
                        "frame": frame + 1,
                        "room": after.screen,
                        "room_hex": f"0x{after.screen:02x}",
                        "mode": after.mode,
                        "x": after.link_x,
                        "y": after.link_y,
                        "keys": after.keys,
                        "reason": action.reason,
                    }
                )
                last_room = after.screen
            elif frame % 250 == 0:
                trail.append(
                    {
                        "event": "sample",
                        "frame": frame + 1,
                        "room": after.screen,
                        "room_hex": f"0x{after.screen:02x}",
                        "mode": after.mode,
                        "x": after.link_x,
                        "y": after.link_y,
                        "keys": after.keys,
                        "reason": action.reason,
                    }
                )
            if controller.success or controller.failed:
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level5_room_56_arrived(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level5North56")
            checkpoint = str(checkpoint_path)
            provenance = str(
                write_state_provenance(
                    checkpoint_path,
                    source_state_path=(
                        GAME_DIR / "custom_integrations" / GAME / f"{start_state}.state"
                    ),
                    request={
                        "segment": "level5_north56",
                        "predecessor_entry": True,
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
            "track": track,
            "intervention_class": intervention_class,
            "start_state": start_state,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "doors": entry.cur_opened_doors,
                "keys": entry.keys,
                "bombs": entry.bombs,
            },
            "trail": trail,
            "controller": controller.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "room_hex": f"0x{snap.screen:02x}",
                "x": snap.link_x,
                "y": snap.link_y,
                "keys": snap.keys,
                "bombs": snap.bombs,
                "health": snap.health,
                "room_item_id": snap.room_item_id,
                "room_all_dead": snap.room_all_dead,
                "cur_opened_doors": snap.cur_opened_doors,
                "whistle": int(read_u8(ram, ADDR_WHISTLE)),
                "candle": int(read_u8(ram, ADDR_CANDLE)),
                "map": int(read_u8(ram, ADDR_MAP)),
                "objects": _objects(snap),
                "live_gibdo": len(
                    [
                        obj
                        for obj in snap.objects
                        if 1 <= obj.slot <= 12
                        and obj.type_id == GIBDO_OBJECT_TYPE
                        and obj.hp > 0
                    ]
                ),
            },
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
    parser.add_argument("--from-state", default="Level5EastKey")
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (ASSIST_CONTRACT). Not Clean STATUS.",
    )
    parser.add_argument(
        "--keep-keys",
        action="store_true",
        help="Safety no-op: this runner never pokes keys.",
    )
    args = parser.parse_args(argv)
    _ = args.keep_keys

    reports = [
        run_once(
            tag=f"l5_north56_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
            infinite_life=args.infinite_life,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report.get("final", {})
        print(
            f"trial={trial} ok={report['ok']} track={report.get('track')} "
            f"room={final.get('room_hex')} keys={final.get('keys')} "
            f"xy={final.get('x')},{final.get('y')} "
            f"frames={report['controller'].get('frames')} "
            f"notes={report['controller'].get('notes')}"
        )

    track, intervention_class = _track_labels(args.infinite_life)
    output = RECORDINGS_DIR / f"l5_north56_{track}.json"
    write_json_report(
        output,
        {
            "segment": "level5_north56",
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": intervention_class,
            "track": track,
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "level5_room_56_arrived",
            "room_id": ROOM_L5_NORTH_56,
            "room_hex": f"0x{ROOM_L5_NORTH_56:02x}",
            "key_poke": False,
            "door_poke": False,
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
