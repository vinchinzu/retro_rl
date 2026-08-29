"""Isolated pure: clear Level 2 compass room 0x6f (6× Gel + compass).

Default start: ``Level2EastKey`` (0x7e, keys≥1). Prefix:

1. LEFT → 0x7d, free east alcove (reverse diamond), UP → 0x6d
2. RIGHT → 0x6e west entry; clear 3 Ropes (``ROOM_6E_SPEC``)
3. Key door RIGHT via wall-vertical-push band y≈113 (``ROOM_6F_SPEC.entry``)
4. Clear 6× Gel TYPE-only; collect compass at ~(200, 101)

Stop: ``level2_room_6f_compass_success`` (gels dead + ``ADDR_COMPASS`` L2 bit).

Examples::

    uv run python nes/zelda_i/scripts/run_level2_clear6f.py --trials 2
    uv run python nes/zelda_i/scripts/run_level2_clear6f.py --save-state --trials 2
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
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
    ROOM_6E_SPEC,
    ROOM_6F_SPEC,
    level2_room_6e_cleared,
    level2_room_6f_compass_success,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

def _run_controller(env, controller, max_frames: int):
    obs = None
    for _ in range(max_frames):
        action = controller.step(read_snapshot(env.get_ram()))
        obs, *_ = env.step(action.action)
        if controller.success or controller.phase is DungeonPhase.FAILED:
            break
    return obs

def _act(env, direction: str | None) -> None:
    env.step(nes_action(direction) if direction else nes_idle_action())

def _idle(env, n: int = 1) -> None:
    for _ in range(n):
        env.step(nes_idle_action())

def _nav_east_key_to_6e(env) -> bool:
    """Level2EastKey (0x7e) → free 0x7d → 0x6d → 0x6e west entry."""

    def snap():
        return read_snapshot(env.get_ram())

    for _ in range(400):
        s = snap()
        if s.screen == 0x7D and s.mode == PLAY_MODE:
            break
        _act(env, "LEFT")

    # Reverse diamond: free east alcove of 0x7d (LEFT×6, UP, LEFT cycles).
    cycle = ["LEFT"] * 6 + ["UP"] * 12 + ["LEFT"] * 20
    for i in range(500):
        if snap().link_x <= 150:
            break
        _act(env, cycle[i % len(cycle)])

    for _ in range(400):
        s = snap()
        if abs(s.link_x - 120) <= 6 and abs(s.link_y - 141) <= 8:
            break
        if abs(s.link_x - 120) > 6:
            _act(env, "RIGHT" if s.link_x < 120 else "LEFT")
        else:
            _act(env, "DOWN" if s.link_y < 141 else "UP")

    for _ in range(900):
        s = snap()
        if s.screen == 0x6D and s.mode == PLAY_MODE:
            break
        if s.mode != PLAY_MODE or abs(s.link_x - 120) <= 8:
            _act(env, "UP")
        else:
            _act(env, "RIGHT" if s.link_x < 120 else "LEFT")

    for _ in range(600):
        s = snap()
        if s.screen == 0x6E and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE or abs(s.link_y - 141) <= 6:
            _act(env, "RIGHT")
        else:
            _act(env, "DOWN" if s.link_y < 141 else "UP")
    return snap().screen == 0x6E and snap().mode == PLAY_MODE

def _clear_6e_keep_mid(env) -> GenericDungeonRoomController:
    """Clear 3 ropes without parking in west/north alcoves."""
    controller = GenericDungeonRoomController(ROOM_6E_SPEC)
    controller.phase = DungeonPhase.FIGHT
    for _ in range(ROOM_6E_SPEC.max_frames):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE:
            if s.link_x < 56:
                _act(env, "RIGHT")
                continue
            if s.link_y < 105:
                _act(env, "DOWN")
                continue
            if s.link_y > 185:
                _act(env, "UP")
                continue
        action = controller.step(s)
        env.step(action.action)
        if controller.success or controller.phase is DungeonPhase.FAILED:
            break
    return controller

def _enter_6f_key_door(env) -> bool:
    """0x6e key-RIGHT: band y≈113 → wall x≥200 → vertical y≈141 → pure RIGHT.

    Do **not** LEFT-nudge at the wall (door_y LEFT re-enters the diamond solid
    on y=141). Matches ROOM_6F_SPEC.entry waypoints, executed as holds.
    """

    def snap():
        return read_snapshot(env.get_ram())

    # Band mid.
    for _ in range(400):
        s = snap()
        if abs(s.link_y - 113) <= 4 and 90 <= s.link_x <= 160:
            break
        if s.mode != PLAY_MODE:
            _act(env, "RIGHT" if s.transitioning else None)
            continue
        if abs(s.link_y - 113) > 4:
            _act(env, "DOWN" if s.link_y < 113 else "UP")
        elif s.link_x < 90:
            _act(env, "RIGHT")
        elif s.link_x > 160:
            _act(env, "LEFT")
        else:
            _act(env, "RIGHT")

    # Wall on band.
    for _ in range(400):
        s = snap()
        if s.link_x >= 200:
            break
        if s.mode != PLAY_MODE:
            _act(env, "RIGHT")
            continue
        if abs(s.link_y - 113) > 6:
            _act(env, "DOWN" if s.link_y < 113 else "UP")
        else:
            _act(env, "RIGHT")

    # Vertical at wall to door_y (no LEFT).
    for _ in range(120):
        s = snap()
        if abs(s.link_y - 141) <= 2 and s.link_x >= 195:
            break
        if s.mode != PLAY_MODE:
            _act(env, "RIGHT")
            continue
        if abs(s.link_y - 141) > 2:
            _act(env, "DOWN" if s.link_y < 141 else "UP")
        else:
            _act(env, "RIGHT")

    # Pure push RIGHT through key door.
    for _ in range(500):
        s = snap()
        if s.screen == 0x6F and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            _act(env, "RIGHT")
            continue
        if abs(s.link_y - 141) > 4:
            _act(env, "DOWN" if s.link_y < 141 else "UP")
        else:
            _act(env, "RIGHT")
    return snap().screen == 0x6F and snap().mode == PLAY_MODE

def run_once(
    *,
    tag: str = "level2_clear6f",
    save_checkpoint: bool = False,
    start_state: str = "Level2EastKey",
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    clear6e: GenericDungeonRoomController | None = None
    controller = GenericDungeonRoomController(ROOM_6F_SPEC)
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())

        prefix_ok = True
        prefix_error = None
        if entry.screen == 0x6F and entry.mode == PLAY_MODE:
            # Already in target room (e.g. Level2Compass reload for combat-only).
            controller.phase = DungeonPhase.FIGHT
            obs = _run_controller(env, controller, ROOM_6F_SPEC.max_frames)
        else:
            if entry.screen == 0x7E:
                if not _nav_east_key_to_6e(env):
                    prefix_ok = False
                    prefix_error = "failed_nav_to_0x6e"
            elif entry.screen == 0x6E:
                pass
            elif entry.screen == 0x6D:
                for _ in range(600):
                    s = read_snapshot(env.get_ram())
                    if s.screen == 0x6E and s.mode == PLAY_MODE:
                        break
                    if s.mode != PLAY_MODE or abs(s.link_y - 141) <= 6:
                        _act(env, "RIGHT")
                    else:
                        _act(env, "DOWN" if s.link_y < 141 else "UP")
                if read_snapshot(env.get_ram()).screen != 0x6E:
                    prefix_ok = False
                    prefix_error = "failed_6d_to_6e"
            else:
                prefix_ok = False
                prefix_error = f"unsupported_start_room_0x{entry.screen:02x}"

            if prefix_ok:
                _idle(env, 120)
                clear6e = _clear_6e_keep_mid(env)
                prefix_ok = level2_room_6e_cleared(env.get_ram())
                if not prefix_ok:
                    prefix_error = "failed_clear_0x6e"

            if prefix_ok:
                if not _enter_6f_key_door(env):
                    prefix_ok = False
                    prefix_error = "failed_key_door_0x6e_to_0x6f"

            if prefix_ok:
                # Already in 0x6f: fight gels + walk compass (no re-entry route).
                controller.phase = DungeonPhase.FIGHT
                obs = _run_controller(env, controller, ROOM_6F_SPEC.max_frames)

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = prefix_ok and level2_room_6f_compass_success(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level2Compass")
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
                        "segment": "level2_clear6f",
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
            "prefix_ok": prefix_ok,
            "prefix_error": prefix_error,
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
                "health": entry.health,
                "compass": entry.compass,
                "doors": entry.cur_opened_doors,
            },
            "clear6e": clear6e.report() if clear6e else None,
            "controller": controller.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "keys": snap.keys,
                "health": snap.health,
                "compass": snap.compass,
                "room_item_id": snap.room_item_id,
                "room_all_dead": snap.room_all_dead,
                "cur_opened_doors": snap.cur_opened_doors,
                "live_gels": len(ROOM_6F_SPEC.live_enemies(snap)),
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
    parser.add_argument("--from-state", default="Level2EastKey")
    args = parser.parse_args(argv)

    reports = [
        run_once(
            tag=f"level2_clear6f_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=args.from_state,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report.get("final") or {}
        print(
            f"trial={trial} ok={report.get('ok')} "
            f"prefix_ok={report.get('prefix_ok')} "
            f"room={final.get('room', 0):02X} compass={final.get('compass')} "
            f"live={final.get('live_gels')} "
            f"xy=({final.get('x')},{final.get('y')}) "
            f"frames={report.get('controller', {}).get('frames')} "
            f"phase={report.get('controller', {}).get('phase')} "
            f"max_live={report.get('controller', {}).get('max_live_enemies')}"
        )

    output = RECORDINGS_DIR / "level2_clear6f_isolated.json"
    write_json_report(
        output,
        {
            "segment": "level2_clear6f",
            "natural_entry": False,
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "intervention_class": "clean",
            "track": "clean",
            "trials": args.trials,
            "successes": sum(1 for report in reports if report.get("ok")),
            "stop_predicate": "level2_room_6f_compass_success",
            "spec_id": ROOM_6F_SPEC.spec_id,
            "entry_policy": (
                "0x6e WEST entry; clear 3 ropes; band y≈113 wall x≥200 "
                "vertical to y≈141 pure RIGHT (key door); no door_y LEFT"
            ),
            "compass": (
                "RoomItemId 0x16; ADDR_COMPASS bitfield; L2 bit → value 2; "
                "east-wall sweep waypoints ~(192–208, y≈101)"
            ),
            "gels": "6× type 0x15 TYPE-only (hp=0 alive)",
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    return 0 if all(report.get("ok") for report in reports) else 1

if __name__ == "__main__":
    raise SystemExit(main())
