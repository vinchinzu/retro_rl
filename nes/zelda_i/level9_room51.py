"""Level 9 play 0x51 dump and north-door dest probe into uncleared 0x41.

Fixture inventory + 0x61 neighbor-scroll stay route-ineligible. After
materialize there are no object / room / door / inventory / progression /
capacity writes. 0x41 doors are never staged.
"""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_ganon import LEVEL9
from zelda_i.level9_path import NORTH_DOOR_X
from zelda_i.level9_stairs import (
    ROOM41,
    ROOM41_ROM_EAST,
    ROOM41_ROM_NORTH,
    ROOM41_ROM_SOUTH,
    ROOM41_ROM_WEST,
    ROOM50,
    ROOM50_ROM_EAST,
    ROOM50_ROM_NORTH,
    ROOM51,
    ROOM51_ROM_EAST,
    ROOM51_ROM_NORTH,
    ROOM51_ROM_SECRET,
    ROOM51_ROM_SOUTH,
    ROOM51_ROM_WEST,
    ROOM61,
    ROOM61_ROM_EAST,
    ROOM61_ROM_NORTH,
    ROOM61_ROM_SOUTH,
    ROOM61_ROM_WEST,
    chase_sword_step,
    dest_report,
    in_room_51,
    live_combat_objects,
    rom_door_name,
    stair_loader_for,
    walk_to_step,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _idle, _step
from zelda_i.level9_stair_run import dump_room_tiles, materialize_stair_room

BEAD = "rr-sz8.4"
CLEAR_MAX_FRAMES = 2500
DEST_PROBE_FRAMES = 1800
ROOM51_SOUTH_Y = 189
ROOM51_WEST_X = 48
ROOM51_EAST_X = 208
ROOM51_MID_Y = 141
# Thread the statue diamond: center UP to y=133, west to x=104, UP past
# the north vertex (live stick 120,117), then the north door band.
ROOM51_THREAD_Y = 133
ROOM51_THREAD_X = 144
ROOM51_NORTH_BAND_Y = 93
ROOM51_DOOR_X_TOL = 1

_DOOR_STAND = {
    "UP": (NORTH_DOOR_X, 77),
    "DOWN": (NORTH_DOOR_X, ROOM51_SOUTH_Y),
    "LEFT": (ROOM51_WEST_X, ROOM51_MID_Y),
    "RIGHT": (ROOM51_EAST_X, ROOM51_MID_Y),
}


def room51_rom_north_is_open() -> bool:
    return ROOM51_ROM_NORTH == 0 and ROOM41_ROM_SOUTH == 7


def room51_is_rom_predecessor_of_41() -> bool:
    """ROM only: 0x51 north open pairs 0x41 south shutter. Live dest separate."""
    return room51_rom_north_is_open()


def room51_loader_avoids_41() -> bool:
    """True when the 0x51 neighbor-scroll does not stage 0x41 doors."""
    return stair_loader_for(ROOM51).from_room != ROOM41


def room51_to_41_step(snap: ZeldaSnapshot) -> FrameAction:
    """Door-column UP through the 0x51 north open door → hypothesized 0x41.

    No door poke on 0x41.
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.transitioning or snap.mode in (4, 6, 7):
        return FrameAction(nes_action("UP"), "room41_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM41:
        return FrameAction(nes_idle_action(), "room41_arrived")
    if snap.screen != ROOM51:
        return FrameAction(
            nes_idle_action(),
            f"unexpected_room_0x{snap.screen:02x}",
        )
    x = int(snap.link_x)
    y = int(snap.link_y)
    if y < ROOM51_THREAD_Y and abs(x - NORTH_DOOR_X) > 4:
        return FrameAction(nes_action("DOWN"), "room51_drop_to_thread")
    if y > ROOM51_THREAD_Y and abs(x - NORTH_DOOR_X) <= 8:
        return FrameAction(nes_action("UP"), "room51_center_to_thread")
    if y >= ROOM51_THREAD_Y and x > ROOM51_THREAD_X + 2:
        return FrameAction(nes_action("LEFT"), "room51_to_thread_x")
    if y >= ROOM51_THREAD_Y and x < ROOM51_THREAD_X - 2:
        return FrameAction(nes_action("RIGHT"), "room51_to_thread_x")
    if y > ROOM51_NORTH_BAND_Y and abs(x - ROOM51_THREAD_X) <= 4:
        return FrameAction(nes_action("UP"), "room51_climb_thread")
    if abs(x - NORTH_DOOR_X) > ROOM51_DOOR_X_TOL:
        direction = "LEFT" if x > NORTH_DOOR_X else "RIGHT"
        return FrameAction(nes_action(direction), "room51_align_x")
    return FrameAction(nes_action("UP"), "room51_push_north")


def _rom_door_row(
    north: int, south: int, west: int, east: int, *, secret: int | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "north": north,
        "south": south,
        "west": west,
        "east": east,
        "north_name": rom_door_name(north),
        "south_name": rom_door_name(south),
        "west_name": rom_door_name(west),
        "east_name": rom_door_name(east),
    }
    if secret is not None:
        row["secret"] = secret
        row["secret_name"] = "all_dead" if secret == 1 else str(secret)
    return row


def _probe_room51_north(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    """Controller-only north open-door push. Does not write doors on 0x51 or 0x41."""
    start = read_snapshot(env.get_ram())
    transition = None
    samples: list[dict[str, Any]] = []
    last_reason = ""
    for frame_i in range(DEST_PROBE_FRAMES):
        snap = read_snapshot(env.get_ram())
        if (
            snap.screen == ROOM41
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            transition = {
                "from_room": ROOM51,
                "direction": "UP",
                "to_room": int(snap.screen),
                "mode": int(snap.mode),
                "objects": dest_report(snap)["objects"],
            }
            break
        if (
            snap.screen != ROOM51
            or snap.mode != PLAY_MODE
            or snap.transitioning
        ):
            last_reason = "hold_up_scroll"
            _step(env, nes_action("UP"), assist=assist, total=total)
            continue
        frame = room51_to_41_step(snap)
        last_reason = frame.reason
        if frame_i % 250 == 0:
            samples.append(
                {
                    "i": frame_i,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "reason": frame.reason,
                }
            )
        _step(env, frame.action, assist=assist, total=total)
    end = read_snapshot(env.get_ram())
    return {
        "start_room": int(start.screen),
        "end_room": int(end.screen),
        "end_mode": int(end.mode),
        "link": {"x": end.link_x, "y": end.link_y},
        "doors": dest_report(end)["doors"],
        "mask": dest_report(end)["mask"],
        "entered_other_room": transition is not None,
        "entered_0x41": int(end.screen) == ROOM41,
        "transition": transition,
        "still_in_51": bool(in_room_51(end)),
        "stuck_y": int(end.link_y),
        "last_reason": last_reason,
        "samples": samples,
        "dest": dest_report(end),
    }


def _probe_room51_dir(
    env: Any,
    direction: str,
    *,
    dest_room: int | None,
    total: list[int],
    assist: Any = None,
) -> dict[str, Any]:
    """Walk to a door stand and hold. Never writes door bits."""
    start = read_snapshot(env.get_ram())
    stand_x, stand_y = _DOOR_STAND[direction]
    transition = None
    for _ in range(DEST_PROBE_FRAMES):
        snap = read_snapshot(env.get_ram())
        left_51 = int(snap.screen) != ROOM51 and snap.mode == PLAY_MODE
        if left_51 and not snap.transitioning:
            transition = {
                "from_room": ROOM51,
                "direction": direction,
                "to_room": int(snap.screen),
                "mode": int(snap.mode),
                "objects": dest_report(snap)["objects"],
            }
            break
        if snap.mode != PLAY_MODE or snap.transitioning or int(snap.screen) != ROOM51:
            _step(env, nes_action(direction), assist=assist, total=total)
            continue
        if dest_room is not None and int(snap.screen) == dest_room:
            break
        if abs(int(snap.link_x) - stand_x) > 4 or abs(int(snap.link_y) - stand_y) > 6:
            # Diamond 0x51: x-first for sides (south band), y-first for N/S aisle.
            y_first = direction in ("UP", "DOWN")
            if direction in ("LEFT", "RIGHT") and int(snap.link_y) < ROOM51_SOUTH_Y - 8:
                frame = walk_to_step(
                    snap, int(snap.link_x), ROOM51_SOUTH_Y, y_first=True
                )
            else:
                frame = walk_to_step(snap, stand_x, stand_y, y_first=y_first)
            _step(env, frame.action, assist=assist, total=total)
            continue
        _step(env, nes_action(direction), assist=assist, total=total)
    end = read_snapshot(env.get_ram())
    return {
        "direction": direction,
        "start_room": int(start.screen),
        "end_room": int(end.screen),
        "end_mode": int(end.mode),
        "link": {"x": end.link_x, "y": end.link_y},
        "doors": dest_report(end)["doors"],
        "mask": dest_report(end)["mask"],
        "entered_other_room": transition is not None,
        "entered_dest": dest_room is not None and int(end.screen) == dest_room,
        "transition": transition,
        "still_in_51": bool(in_room_51(end)),
        "dest": dest_report(end),
    }


def _clear_room51(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    cooldown = 0
    frames = 0
    for _ in range(CLEAR_MAX_FRAMES):
        snap = read_snapshot(env.get_ram())
        if not in_room_51(snap):
            break
        if not live_combat_objects(snap):
            break
        frame, cooldown = chase_sword_step(snap, cooldown)
        _step(env, frame.action, assist=assist, total=total)
        frames += 1
    _idle(env, 16, assist=assist, total=total)
    end = read_snapshot(env.get_ram())
    return {
        "frames": frames,
        "combat_left": len(live_combat_objects(end)),
        "doors": dest_report(end)["doors"],
        "mask": dest_report(end)["mask"],
        "screen": int(end.screen),
        "link": {"x": end.link_x, "y": end.link_y},
        "room_all_dead": dest_report(end)["room_all_dead"],
    }


def _rematerialize(
    env: Any, *, door_staging: bool, total: list[int]
) -> tuple[Any, Any, Any, bool]:
    env.close()
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total[:] = [0]
    obs, loader, loaded = materialize_stair_room(
        env, ROOM51, total=total, door_staging=door_staging
    )
    return env, obs, loader, loaded


def dump_room_51(*, tag: str = "l9_room51_dump") -> dict[str, Any]:
    """Live 0x51 dump + dest probes. Stages 0x61, never 0x41 doors."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "room": "0x51",
        "rom_doors": {
            "0x51": _rom_door_row(
                ROOM51_ROM_NORTH,
                ROOM51_ROM_SOUTH,
                ROOM51_ROM_WEST,
                ROOM51_ROM_EAST,
                secret=ROOM51_ROM_SECRET,
            ),
            "0x41": _rom_door_row(
                ROOM41_ROM_NORTH,
                ROOM41_ROM_SOUTH,
                ROOM41_ROM_WEST,
                ROOM41_ROM_EAST,
            ),
            "0x61": _rom_door_row(
                ROOM61_ROM_NORTH,
                ROOM61_ROM_SOUTH,
                ROOM61_ROM_WEST,
                ROOM61_ROM_EAST,
            ),
            "0x50": {
                **_rom_door_row(
                    ROOM50_ROM_NORTH,
                    1,
                    1,
                    ROOM50_ROM_EAST,
                ),
                "note": "east shutter pairs 0x51 west; north key is dirty 0x40 — not this chain",
            },
            "0x40": {
                "note": "dirty; keep out of the 0x51 → 0x41 predecessor chain",
            },
        },
        "rom_north_is_open": room51_rom_north_is_open(),
        "rom_predecessor": room51_is_rom_predecessor_of_41(),
        "loader_avoids_41": room51_loader_avoids_41(),
        "door_poke_on_41": False,
    }

    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        total = [0]
        obs, loader, loaded = materialize_stair_room(
            env, ROOM51, total=total, door_staging=True
        )
        report["loader"] = {
            "label": loader.label,
            "from_room": f"0x{loader.from_room:02X}",
            "direction": loader.direction,
            "link_xy": [loader.link_x, loader.link_y],
            "door_staging_on_from_room": True,
            "from_room_is_41": loader.from_room == ROOM41,
            "from_room_is_61": loader.from_room == ROOM61,
            "note": (
                "0x0F/0x0F is written on 0x61 so the south-open scroll can start; "
                "0x41 doors are not poked."
            ),
        }
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x51"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        snap = read_snapshot(env.get_ram())
        report["settled"] = dest_report(snap)
        report["link_start"] = {"x": snap.link_x, "y": snap.link_y}
        settle_png = RECORDINGS_DIR / f"{tag}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

        tiles = dump_room_tiles(env, total=total)
        tile_json = RECORDINGS_DIR / f"{tag}_tiles.json"
        write_json_report(tile_json, tiles)
        report["tiles"] = {
            "stair_hits": tiles["stair_hits"],
            "mouth_hits": tiles["mouth_hits"],
            "tile_counts": tiles["tile_counts"],
        }
        report["tile_json"] = str(tile_json)

        env, obs, _, loaded = _rematerialize(env, door_staging=True, total=total)
        if not loaded:
            report["error"] = "rematerialize failed 0x51 before north"
            return report
        north = _probe_room51_north(env, total=total, assist=assist)
        report["north_probe_uncleared"] = north
        idle = _idle(env, 16, assist=assist, total=total)
        north_png = RECORDINGS_DIR / f"{tag}_north_probe.png"
        save_rgb_png(idle, north_png)
        report["north_probe_png"] = str(north_png)
        dest_snap = read_snapshot(env.get_ram())
        if int(dest_snap.screen) == ROOM41 and dest_snap.mode == PLAY_MODE:
            report["dest_screen"] = int(dest_snap.screen)
            report["dest_mode"] = int(dest_snap.mode)
            report["lands_0x41"] = True
            dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
            save_rgb_png(idle, dest_png)
            report["dest_png"] = str(dest_png)
            report["dest_objects"] = dest_report(dest_snap)["objects"]
            report["dest_settled"] = dest_report(dest_snap)

        for direction, dest_room, key in (
            ("DOWN", ROOM61, "south_probe_uncleared"),
            ("LEFT", ROOM50, "west_probe_uncleared"),
            ("RIGHT", None, "east_probe_uncleared"),
        ):
            env, obs, _, loaded = _rematerialize(env, door_staging=True, total=total)
            if not loaded:
                report[key] = {"error": "rematerialize failed"}
                continue
            probed = _probe_room51_dir(
                env, direction, dest_room=dest_room, total=total, assist=assist
            )
            report[key] = probed
            idle = _idle(env, 1, assist=assist, total=total)
            png = RECORDINGS_DIR / f"{tag}_{direction.lower()}_probe.png"
            save_rgb_png(idle, png)
            report[f"{key}_png"] = str(png)

        env, obs, _, loaded = _rematerialize(env, door_staging=True, total=total)
        if loaded:
            report["clear"] = _clear_room51(env, total=total, assist=assist)
            idle = _idle(env, 1, assist=assist, total=total)
            report["after_clear"] = dest_report(read_snapshot(env.get_ram()))
            clear_png = RECORDINGS_DIR / f"{tag}_after_clear.png"
            save_rgb_png(idle, clear_png)
            report["after_clear_png"] = str(clear_png)
            west_clear = _probe_room51_dir(
                env, "LEFT", dest_room=ROOM50, total=total, assist=assist
            )
            report["west_probe_cleared"] = west_clear
            idle = _idle(env, 1, assist=assist, total=total)
            west_clear_png = RECORDINGS_DIR / f"{tag}_west_cleared.png"
            save_rgb_png(idle, west_clear_png)
            report["west_probe_cleared_png"] = str(west_clear_png)
            if not report.get("lands_0x41"):
                env, obs, _, loaded = _rematerialize(
                    env, door_staging=True, total=total
                )
                if loaded:
                    _clear_room51(env, total=total, assist=assist)
                    north_clear = _probe_room51_north(
                        env, total=total, assist=assist
                    )
                    report["north_probe_cleared"] = north_clear
                    idle = _idle(env, 1, assist=assist, total=total)
                    clear_dest_png = RECORDINGS_DIR / f"{tag}_cleared_dest.png"
                    save_rgb_png(idle, clear_dest_png)
                    report["cleared_dest_png"] = str(clear_dest_png)
                    dest_snap = read_snapshot(env.get_ram())
                    report["cleared_dest_screen"] = int(dest_snap.screen)
                    report["cleared_dest_mode"] = int(dest_snap.mode)
                    if int(dest_snap.screen) == ROOM41 and dest_snap.mode == PLAY_MODE:
                        report["dest_screen"] = int(dest_snap.screen)
                        report["dest_mode"] = int(dest_snap.mode)
                        report["lands_0x41"] = True
                        dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
                        save_rgb_png(idle, dest_png)
                        report["dest_png"] = str(dest_png)
                        report["dest_objects"] = dest_report(dest_snap)["objects"]
                        report["dest_settled"] = dest_report(dest_snap)

        env, obs, loader_np, loaded_np = _rematerialize(
            env, door_staging=False, total=total
        )
        report["no_door_poke_settle"] = {
            "loaded": loaded_np,
            "loader": loader_np.label,
            "final": compact_snapshot(read_snapshot(env.get_ram())),
        }
        if loaded_np:
            no_poke_png = RECORDINGS_DIR / f"{tag}_no_door_poke_settle.png"
            save_rgb_png(obs, no_poke_png)
            report["no_door_poke_settle"]["settled"] = dest_report(
                read_snapshot(env.get_ram())
            )
            report["no_door_poke_settle"]["png"] = str(no_poke_png)
            north_np = _probe_room51_north(env, total=total, assist=assist)
            report["north_probe_no_door_poke"] = north_np
            idle = _idle(env, 1, assist=assist, total=total)
            np_png = RECORDINGS_DIR / f"{tag}_no_door_poke_north.png"
            save_rgb_png(idle, np_png)
            report["north_probe_no_door_poke_png"] = str(np_png)
            dest_snap = read_snapshot(env.get_ram())
            if int(dest_snap.screen) == ROOM41 and dest_snap.mode == PLAY_MODE:
                report["lands_0x41"] = True
                report["dest_screen"] = int(dest_snap.screen)
                report["dest_mode"] = int(dest_snap.mode)
                if "dest_objects" not in report:
                    report["dest_objects"] = dest_report(dest_snap)["objects"]
                    report["dest_settled"] = dest_report(dest_snap)

        entered = bool(
            (report.get("north_probe_uncleared") or {}).get("entered_0x41")
            or (report.get("north_probe_cleared") or {}).get("entered_0x41")
            or (report.get("north_probe_no_door_poke") or {}).get("entered_0x41")
            or report.get("lands_0x41")
        )
        report["lands_0x41"] = bool(entered)
        dest_uncleared = bool(
            (report.get("dest_settled") or {}).get("objects")
        )
        report["dest_uncleared_0x41"] = bool(
            entered and dest_uncleared
        )
        report["how_north_opens"] = (
            "already_open"
            if entered
            and (report["settled"]["doors"].get("north") or room51_rom_north_is_open())
            else "shutter_after_clear"
            if entered
            else "visually_open_statue_blocked"
        )
        if entered:
            report["shutter_contract"] = (
                "0x51 north is ROM-open; 0x41 south is shutter. "
                "Controller UP from live 0x51 lands play 0x41 with no 0x41 door poke. "
                "0x41 start objects are the dest snapshot (traps + Like-Likes expected)."
            )
            report["next_candidate"] = (
                "0x61 south-open (current 0x51 loader) or 0x50 east shutter "
                "after 0x51 all_dead. Keep 0x40 out of this chain."
            )
        else:
            last = (
                report.get("north_probe_cleared")
                or report.get("north_probe_no_door_poke")
                or report.get("north_probe_uncleared")
                or {}
            )
            report["dest_screen"] = last.get("end_room")
            report["dest_mode"] = last.get("end_mode")
            report["shutter_contract"] = (
                "0x51 north is ROM-open and visually black; 0x41 south is "
                "shutter. After Like-Like clear the west shutter opens "
                "(all_dead, doors raw=2). North walk is blocked by the statue "
                "diamond (center stick 120,117; thread 104/144 blocked). "
                "No 0x41 door poke."
            )
            report["next_candidate"] = (
                "0x51 is the identified south predecessor of 0x41 (ROM + "
                "visual north open) but the live walk is not earned. Next: "
                "thread the statue diamond from the south-door spawn after "
                "clear, or materialize 0x61 (south-open pred of 0x51). "
                "Keep 0x40 out of this chain."
            )
        report["ok"] = bool(loaded)
        return report
    finally:
        env.close()


__all__ = [
    "BEAD",
    "ROOM51_SOUTH_Y",
    "dump_room_51",
    "in_room_51",
    "room51_is_rom_predecessor_of_41",
    "room51_loader_avoids_41",
    "room51_rom_north_is_open",
    "room51_to_41_step",
]
