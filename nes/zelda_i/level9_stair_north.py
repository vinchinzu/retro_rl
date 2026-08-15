"""Level 9 play 0x21 / 0x41 dumps and south-shutter compose-to-credits."""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_ganon import (
    ADDR_SELECTED_ITEM,
    B_ITEM_ARROWS,
    B_ITEM_BOMBS,
    GanonFightController,
    LEVEL9,
    credits_rolling,
    final_ending_screen,
    ganon_defeated,
)
from zelda_i.level9_patra import (
    FinalPatraFightController,
    NORTH_DOOR,
    PATRA_EYE_COUNT,
    PATRA_MAX_FRAMES,
    final_patra_live,
    final_patra_north_door_earned,
    patra_action,
    patra_body,
    patra_eyes,
)
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES
from zelda_i.level9_stairs import (
    BLACK_MOUTH_TILE,
    BLOCK_PUSH_STANDS,
    CELLAR_MODE,
    ITEM_CELLAR_MODE,
    LEVEL9_CELLAR_ROOMS,
    LEVEL9_STAIR_PAIRS,
    PATRA_STAIR_SOURCE,
    PLAY_STAIR_CANDIDATES,
    STAIR_STANDS,
    STAIR_TILE_HI,
    STAIR_TILE_LO,
    StairLoader,
    cellar_dest_for,
    cellar_exit_step,
    cellar_for_play_room,
    cellar_mouth_xy,
    in_patra_cellar,
    is_patra_cellar_source,
    chase_sword_step,
    dest_report,
    room03_stairs_step,
    room03_like_like_blocks_push,
    room03_west_block_pushed,
    in_stair_source,
    landed_final_patra,
    live_combat_objects,
    on_stair_tile,
    on_warp_tile,
    paired_stair_dest,
    play_rooms_entering_cellar,
    in_room_04,
    in_room_13,
    in_room_30,
    in_room_21,
    in_room_31,
    in_room_40,
    in_room_41,
    room30_loader_avoids_04,
    room30_rom_neighbors,
    room30_rom_secret_is_block_stairs,
    room30_stairs_step,
    room30_block_secret_open,
    in_cellar_67,
    room40_is_rom_predecessor_of_30,
    room40_loader_avoids_30,
    room40_rom_north_is_key,
    room40_to_30_step,
    make_room04_bomb_west_controller,
    make_room31_bomb_west_controller,
    pause_select_next_b_item_script,
    room04_bomb_west_approach_step,
    room31_bomb_west_approach_step,
    room21_is_rom_predecessor_of_31,
    room21_loader_avoids_31,
    room21_rom_south_is_shutter,
    room21_to_31_step,
    room41_is_rom_predecessor_of_31,
    room41_loader_avoids_31,
    room41_rom_north_is_open,
    room41_to_31_step,
    room31_is_rom_predecessor_of_30,
    room31_loader_avoids_30,
    room31_rom_west_is_bomb,
    room03_rom_neighbors,
    room04_is_rom_predecessor_of_03,
    room04_rom_west_is_bomb,
    room13_is_clean_predecessor_of_03,
    room13_to_03_step,
    ROOM03,
    ROOM03_ROM_EAST,
    ROOM03_ROM_NORTH,
    ROOM03_ROM_SOUTH,
    ROOM03_ROM_WEST,
    ROOM03_PUSH_X,
    ROOM03_PUSH_Y,
    ROOM03_STAIR_X,
    ROOM03_STAIR_Y,
    ROOM04,
    ROOM04_BOMB_WEST_STAND,
    ROOM30,
    ROOM30_ROM_EAST,
    ROOM30_ROM_NORTH,
    ROOM30_ROM_SECRET,
    ROOM30_ROM_SOUTH,
    ROOM30_ROM_WEST,
    ROOM30_STAIR_X,
    ROOM30_STAIR_Y,
    ROOM40,
    ROOM40_ROM_NORTH,
    ROOM40_ROM_SOUTH,
    ROOM40_ROM_WEST,
    ROOM40_ROM_EAST,
    ROOM20,
    ROOM20_ROM_SOUTH,
    ROOM2F,
    ROOM11,
    ROOM21,
    ROOM21_ROM_EAST,
    ROOM21_ROM_NORTH,
    ROOM21_ROM_SOUTH,
    ROOM21_ROM_WEST,
    ROOM31,
    ROOM31_BOMB_WEST_STAND,
    ROOM31_ROM_EAST,
    ROOM31_ROM_NORTH,
    ROOM31_ROM_SOUTH,
    ROOM31_ROM_WEST,
    ROOM41,
    ROOM41_ROM_EAST,
    ROOM41_ROM_NORTH,
    ROOM41_ROM_SOUTH,
    ROOM41_ROM_WEST,
    ROOM51,
    ROOM04_ROM_EAST,
    ROOM04_ROM_NORTH,
    ROOM04_ROM_SOUTH,
    ROOM04_ROM_WEST,
    ROOM13,
    ROOM13_ROM_EAST,
    ROOM13_ROM_NORTH,
    ROOM13_ROM_SOUTH,
    ROOM13_ROM_WEST,
    stair_loader_for,
    stair_transition_modes,
    take_stairs_step,
    walk_to_step,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_NEXT_SCREEN,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.scripts.run_level9_ganon import (
    FIXTURE_SOURCE,
    FULL_LOADOUT,
    _assign,
    _checkpoint_result,
    _collect_power_triforce,
    _enter_ganon,
    _enter_zelda,
    _idle,
    _rescue_zelda,
    _save_checkpoint,
    _step,
)
from zelda_i.scripts.run_level9_patra import _inventory_snapshot
from zelda_i.level9_stair_run import (
    _loader_write_rows,
    _run_bomb_west_04,
    _run_bomb_west_31,
    dump_room_tiles,
    materialize_stair_room,
    take_stairs_from_source,
    walk_play_room_to_patra,
)
from zelda_i.level9_stair_suffix import run_suffix_from_live_env

BEAD = "rr-sz8.3"
TAG = "l9_stair_patra_credits_recon"
SETTLE_IDLE_FRAMES = 20
LOAD_MAX_FRAMES = 500
WALK_MAX_FRAMES = 400
PUSH_FRAMES = 40
CLEAR_MAX_FRAMES = 2500
CELLAR_MAX_FRAMES = 2800
GRID_MAX_FRAMES = 1800
ADDR_UPDATING = 0x0011
ADDR_SUBMODE = 0x0013
ADDR_UW_EXIT_TYPE = 0x005A
NORTH_PROBE_FRAMES = 400
CLEAR_PROBE_FRAMES = 1800

def _room21_patra_live(snap: Any) -> bool:
    return patra_body(snap) is not None or bool(patra_eyes(snap))


def _clear_room21(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    """Kill 0x21 Patra with the existing south-stand policy. No new phase machine."""
    cooldown = 0
    frames = 0
    for _ in range(PATRA_MAX_FRAMES):
        snap = read_snapshot(env.get_ram())
        if not in_room_21(snap):
            break
        if not _room21_patra_live(snap) and not live_combat_objects(snap):
            break
        if _room21_patra_live(snap):
            action, _reason, cooldown = patra_action(snap, cooldown=cooldown)
            _step(env, action, assist=assist, total=total)
        else:
            frame, cooldown = chase_sword_step(snap, cooldown)
            _step(env, frame.action, assist=assist, total=total)
        frames += 1
    _idle(env, 16, assist=assist, total=total)
    end = read_snapshot(env.get_ram())
    return {
        "frames": frames,
        "body_alive": patra_body(end) is not None,
        "eyes": len(patra_eyes(end)),
        "combat_left": len(live_combat_objects(end)),
        "doors": dest_report(end)["doors"],
        "screen": int(end.screen),
        "link": {"x": end.link_x, "y": end.link_y},
    }


def _probe_room21_south(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    """Controller-only south shutter push. Does not write 0x0F/0x0F on 0x21 or 0x31."""
    start = read_snapshot(env.get_ram())
    transition = None
    for _ in range(NORTH_PROBE_FRAMES + 800):
        snap = read_snapshot(env.get_ram())
        if (
            snap.screen == ROOM31
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            transition = {
                "from_room": ROOM21,
                "direction": "DOWN",
                "to_room": int(snap.screen),
                "mode": int(snap.mode),
                "objects": dest_report(snap)["objects"],
            }
            break
        if (
            snap.screen != ROOM21
            or snap.mode != PLAY_MODE
            or snap.transitioning
        ):
            _step(env, nes_action("DOWN"), assist=assist, total=total)
            continue
        frame = room21_to_31_step(snap)
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
        "entered_0x31": int(end.screen) == ROOM31,
        "transition": transition,
        "still_in_21": bool(in_room_21(end)),
        "stuck_y": int(end.link_y),
    }


def dump_room_21(*, tag: str = "l9_room21_dump") -> dict[str, Any]:
    """Live 0x21 dump + south-shutter probe. Stages 0x11, never 0x31 doors."""
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
        "room": "0x21",
        "rom_doors": {
            "0x21": {
                "north": ROOM21_ROM_NORTH,
                "south": ROOM21_ROM_SOUTH,
                "west": ROOM21_ROM_WEST,
                "east": ROOM21_ROM_EAST,
                "north_name": "open",
                "south_name": "shutter",
                "west_name": "wall",
                "east_name": "bomb",
                "secret": 0,
                "secret_name": "none",
            },
            "0x31": {
                "north": ROOM31_ROM_NORTH,
                "south": ROOM31_ROM_SOUTH,
                "west": ROOM31_ROM_WEST,
                "east": ROOM31_ROM_EAST,
                "north_name": "open",
                "south_name": "shutter",
                "west_name": "bomb",
                "east_name": "wall",
                "secret": 0,
                "secret_name": "none",
            },
            "0x11": {"south": 7, "south_name": "shutter"},
        },
        "rom_south_is_shutter": room21_rom_south_is_shutter(),
        "rom_predecessor": room21_is_rom_predecessor_of_31(),
        "loader_avoids_31": room21_loader_avoids_31(),
        "door_poke_on_31": False,
        "selected_item_fixture": B_ITEM_BOMBS,
    }

    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        total = [0]
        obs, loader, loaded = materialize_stair_room(
            env, ROOM21, total=total, door_staging=True, selected_item=B_ITEM_BOMBS
        )
        report["loader"] = {
            "label": loader.label,
            "from_room": f"0x{loader.from_room:02X}",
            "direction": loader.direction,
            "link_xy": [loader.link_x, loader.link_y],
            "door_staging_on_from_room": True,
            "from_room_is_31": loader.from_room == ROOM31,
            "from_room_is_11": loader.from_room == ROOM11,
            "note": "0x0F/0x0F is written on 0x11 so the south-shutter scroll can start; 0x31 doors are not poked.",
        }
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x21"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        snap = read_snapshot(env.get_ram())
        report["settled"] = dest_report(snap)
        report["link_start"] = {"x": snap.link_x, "y": snap.link_y}
        report["selected_item_live"] = int(env.get_ram()[ADDR_SELECTED_ITEM])
        report["bombs_live"] = int(snap.bombs)
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

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, _, loaded = materialize_stair_room(
            env, ROOM21, total=total, door_staging=True, selected_item=B_ITEM_BOMBS
        )
        if not loaded:
            report["error"] = "rematerialize failed 0x21 before south"
            return report

        south = _probe_room21_south(env, total=total, assist=assist)
        report["south_probe_uncleared"] = south
        idle = _idle(env, 16, assist=assist, total=total)
        south_png = RECORDINGS_DIR / f"{tag}_south_probe.png"
        save_rgb_png(idle, south_png)
        report["south_probe_png"] = str(south_png)
        dest_snap = read_snapshot(env.get_ram())
        if int(dest_snap.screen) == ROOM31 and dest_snap.mode == PLAY_MODE:
            report["dest_screen"] = int(dest_snap.screen)
            report["dest_mode"] = int(dest_snap.mode)
            report["lands_0x31"] = True
            dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
            save_rgb_png(idle, dest_png)
            report["dest_png"] = str(dest_png)
            report["dest_objects"] = dest_report(dest_snap)["objects"]

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, _, loaded = materialize_stair_room(
            env, ROOM21, total=total, door_staging=True, selected_item=B_ITEM_BOMBS
        )
        if loaded:
            report["patra_clear"] = _clear_room21(env, total=total, assist=assist)
            obs = _idle(env, 1, assist=assist, total=total)
            report["after_clear"] = dest_report(read_snapshot(env.get_ram()))
            clear_png = RECORDINGS_DIR / f"{tag}_after_clear.png"
            save_rgb_png(obs, clear_png)
            report["after_clear_png"] = str(clear_png)
            south_clear = _probe_room21_south(env, total=total, assist=assist)
            report["south_probe_cleared"] = south_clear
            idle = _idle(env, 1, assist=assist, total=total)
            clear_dest_png = RECORDINGS_DIR / f"{tag}_cleared_dest.png"
            save_rgb_png(idle, clear_dest_png)
            report["cleared_dest_png"] = str(clear_dest_png)
            dest_snap = read_snapshot(env.get_ram())
            report["cleared_dest_screen"] = int(dest_snap.screen)
            report["cleared_dest_mode"] = int(dest_snap.mode)
            if int(dest_snap.screen) == ROOM31 and dest_snap.mode == PLAY_MODE:
                report["dest_screen"] = int(dest_snap.screen)
                report["dest_mode"] = int(dest_snap.mode)
                report["lands_0x31"] = True
                dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
                save_rgb_png(idle, dest_png)
                report["dest_png"] = str(dest_png)
                report["dest_objects"] = dest_report(dest_snap)["objects"]
                cooldown = 0
                for _ in range(CLEAR_MAX_FRAMES):
                    snap = read_snapshot(env.get_ram())
                    if in_room_31(snap) and not live_combat_objects(snap):
                        break
                    if not in_room_31(snap):
                        break
                    frame, cooldown = chase_sword_step(snap, cooldown)
                    _step(env, frame.action, assist=assist, total=total)
                _idle(env, 16, assist=assist, total=total)
                bomb = _run_bomb_west_31(env, total=total, assist=assist)
                bomb.pop("after_bomb_obs", None)
                dest_obs = bomb.pop("obs", None)
                report["bomb_west_after_south"] = bomb
                if dest_obs is not None:
                    bomb_png = RECORDINGS_DIR / f"{tag}_bomb_west_dest.png"
                    save_rgb_png(dest_obs, bomb_png)
                    report["bomb_west_dest_png"] = str(bomb_png)
                bomb_snap = read_snapshot(env.get_ram())
                report["west_bomb_still_works"] = int(bomb_snap.screen) == ROOM30
                report["bomb_dest_screen"] = int(bomb_snap.screen)

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, loader_np, loaded_np = materialize_stair_room(
            env, ROOM21, total=total, door_staging=False, selected_item=B_ITEM_BOMBS
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

        entered = bool(
            (report.get("south_probe_uncleared") or {}).get("entered_0x31")
            or (report.get("south_probe_cleared") or {}).get("entered_0x31")
        )
        report["lands_0x31"] = bool(entered)
        report["how_south_opens"] = (
            "already_open"
            if entered and (report["settled"]["doors"].get("south"))
            else "shutter_after_clear"
            if entered
            else "sealed"
        )
        if not entered:
            last = (
                report.get("south_probe_cleared")
                or report.get("south_probe_uncleared")
                or {}
            )
            report["dest_screen"] = last.get("end_room")
            report["dest_mode"] = last.get("end_mode")
            report["next_candidate"] = (
                "0x21 south shutter stays sealed after Patra kill "
                "(RoomAllDead nonzero, doors raw 0, stand 120,189). "
                "Next clean 0x31 entry: play 0x41 north (ROM open; current "
                "0x31 loader). 0x31 east is wall; 0x31 west is live dest "
                "0x30, not pred. Do not treat 0x40 as next."
            )
        report["ok"] = bool(loaded)
        return report
    finally:
        env.close()


def dump_room_41(*, tag: str = "l9_room41_dump") -> dict[str, Any]:
    """Live 0x41 dump + north-door probe. Stages 0x51, never 0x31 doors."""
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
        "room": "0x41",
        "rom_doors": {
            "0x41": {
                "north": ROOM41_ROM_NORTH,
                "south": ROOM41_ROM_SOUTH,
                "west": ROOM41_ROM_WEST,
                "east": ROOM41_ROM_EAST,
                "north_name": "open",
                "south_name": "shutter",
                "west_name": "wall",
                "east_name": "wall",
                "secret": 0,
                "secret_name": "none",
            },
            "0x31": {
                "north": ROOM31_ROM_NORTH,
                "south": ROOM31_ROM_SOUTH,
                "west": ROOM31_ROM_WEST,
                "east": ROOM31_ROM_EAST,
                "north_name": "open",
                "south_name": "shutter",
                "west_name": "bomb",
                "east_name": "wall",
                "secret": 0,
                "secret_name": "none",
            },
            "0x51": {"north": 0, "north_name": "open"},
            "0x40": {"east": 1, "east_name": "wall", "note": "dirty; not a clean pred"},
        },
        "rom_north_is_open": room41_rom_north_is_open(),
        "rom_predecessor": room41_is_rom_predecessor_of_31(),
        "loader_avoids_31": room41_loader_avoids_31(),
        "door_poke_on_31": False,
    }

    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        total = [0]
        obs, loader, loaded = materialize_stair_room(
            env, ROOM41, total=total, door_staging=True
        )
        report["loader"] = {
            "label": loader.label,
            "from_room": f"0x{loader.from_room:02X}",
            "direction": loader.direction,
            "link_xy": [loader.link_x, loader.link_y],
            "door_staging_on_from_room": True,
            "from_room_is_31": loader.from_room == ROOM31,
            "from_room_is_51": loader.from_room == ROOM51,
            "note": "0x0F/0x0F is written on 0x51 so the south-shutter scroll can start; 0x31 doors are not poked.",
        }
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x41"
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

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, _, loaded = materialize_stair_room(
            env, ROOM41, total=total, door_staging=True
        )
        if not loaded:
            report["error"] = "rematerialize failed 0x41 before north"
            return report

        north = _probe_room41_north(env, total=total, assist=assist)
        report["north_probe_uncleared"] = north
        idle = _idle(env, 16, assist=assist, total=total)
        north_png = RECORDINGS_DIR / f"{tag}_north_probe.png"
        save_rgb_png(idle, north_png)
        report["north_probe_png"] = str(north_png)
        dest_snap = read_snapshot(env.get_ram())
        if int(dest_snap.screen) == ROOM31 and dest_snap.mode == PLAY_MODE:
            report["dest_screen"] = int(dest_snap.screen)
            report["dest_mode"] = int(dest_snap.mode)
            report["lands_0x31"] = True
            dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
            save_rgb_png(idle, dest_png)
            report["dest_png"] = str(dest_png)
            report["dest_objects"] = dest_report(dest_snap)["objects"]

        if not report.get("lands_0x31"):
            env.close()
            env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
            total = [0]
            obs, _, loaded = materialize_stair_room(
                env, ROOM41, total=total, door_staging=True
            )
            if loaded:
                cooldown = 0
                for _ in range(CLEAR_MAX_FRAMES):
                    snap = read_snapshot(env.get_ram())
                    if in_room_41(snap) and not live_combat_objects(snap):
                        break
                    if not in_room_41(snap):
                        break
                    frame, cooldown = chase_sword_step(snap, cooldown)
                    _step(env, frame.action, assist=assist, total=total)
                _idle(env, 16, assist=assist, total=total)
                report["after_clear"] = dest_report(read_snapshot(env.get_ram()))
                clear_png = RECORDINGS_DIR / f"{tag}_after_clear.png"
                save_rgb_png(_idle(env, 1, assist=assist, total=total), clear_png)
                report["after_clear_png"] = str(clear_png)
                north_clear = _probe_room41_north(env, total=total, assist=assist)
                report["north_probe_cleared"] = north_clear
                idle = _idle(env, 1, assist=assist, total=total)
                clear_dest_png = RECORDINGS_DIR / f"{tag}_cleared_dest.png"
                save_rgb_png(idle, clear_dest_png)
                report["cleared_dest_png"] = str(clear_dest_png)
                dest_snap = read_snapshot(env.get_ram())
                report["cleared_dest_screen"] = int(dest_snap.screen)
                report["cleared_dest_mode"] = int(dest_snap.mode)
                if int(dest_snap.screen) == ROOM31 and dest_snap.mode == PLAY_MODE:
                    report["dest_screen"] = int(dest_snap.screen)
                    report["dest_mode"] = int(dest_snap.mode)
                    report["lands_0x31"] = True
                    dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
                    save_rgb_png(idle, dest_png)
                    report["dest_png"] = str(dest_png)
                    report["dest_objects"] = dest_report(dest_snap)["objects"]

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, loader_np, loaded_np = materialize_stair_room(
            env, ROOM41, total=total, door_staging=False
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

        entered = bool(
            (report.get("north_probe_uncleared") or {}).get("entered_0x31")
            or (report.get("north_probe_cleared") or {}).get("entered_0x31")
            or report.get("lands_0x31")
        )
        report["lands_0x31"] = bool(entered)
        report["how_north_opens"] = (
            "already_open"
            if entered and (report["settled"]["doors"].get("north"))
            else "open_walk"
            if entered
            else "sealed"
        )
        if not entered:
            last = (
                report.get("north_probe_cleared")
                or report.get("north_probe_uncleared")
                or {}
            )
            report["dest_screen"] = last.get("end_room")
            report["dest_mode"] = last.get("end_mode")
            report["next_candidate"] = (
                "0x41 north stays sealed into 0x31 "
                f"(end_room={last.get('end_room')}, y={last.get('stuck_y')}). "
                "0x31 east is wall; 0x21 south shutter sealed after Patra. "
                "Do not treat 0x40 as next."
            )
        report["ok"] = bool(loaded)
        return report
    finally:
        env.close()


def run_room21_south_to_credits(
    *,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = "l9_play21_south_patra_credits_recon",
    trial_i: int = 0,
) -> dict[str, Any]:
    """One continuous trial: 0x21 south → 0x31 bomb-west → 0x30 stairs suffix.

    Fixture inventory + 0x11 neighbor-scroll happen before the walk.
    After materialize, no object / room / door / progression / capacity writes.
    0x31 doors are never poked. InitMode9 is not used.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    loader = stair_loader_for(ROOM21)
    writes = _loader_write_rows(loader)
    writes.append(
        {
            "name": "selected_item_bombs",
            "address": ADDR_SELECTED_ITEM,
            "address_hex": "0x0656",
            "value": B_ITEM_BOMBS,
        }
    )
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "source_room": "0x21",
        "via": "south_shutter",
        "trial": trial_i,
        "tag": tag,
        "fixture_writes": writes,
        "runtime_controller_writes": {
            "object": 0,
            "room": 0,
            "door": 0,
            "inventory": 0,
            "progression": 0,
            "capacity": 0,
        },
        "door_poke_on_31": False,
    }
    try:
        obs, used, loaded = materialize_stair_room(
            env, ROOM21, total=total, selected_item=B_ITEM_BOMBS
        )
        report["loader"] = used.label
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x21"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        report["settled"] = dest_report(read_snapshot(env.get_ram()))
        settle_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

        report["patra_clear"] = _clear_room21(env, total=total, assist=assist)

        south = _probe_room21_south(env, total=total, assist=assist)
        report["south"] = south
        snap = read_snapshot(env.get_ram())
        idle = _idle(env, 1, assist=assist, total=total)
        save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")
        report["dest_screen"] = int(snap.screen)
        if int(snap.screen) != ROOM31:
            report["error"] = (
                f"0x21 south dest 0x{snap.screen:02X} is not 0x31 "
                f"end_mode={south.get('end_mode')}"
            )
            return report

        cooldown = 0
        for _ in range(CLEAR_MAX_FRAMES):
            snap = read_snapshot(env.get_ram())
            if in_room_31(snap) and not live_combat_objects(snap):
                break
            if not in_room_31(snap):
                break
            frame, cooldown = chase_sword_step(snap, cooldown)
            _step(env, frame.action, assist=assist, total=total)
        _idle(env, 16, assist=assist, total=total)

        bomb = _run_bomb_west_31(env, total=total, assist=assist)
        bomb.pop("after_bomb_obs", None)
        dest_obs = bomb.pop("obs", None)
        report["bomb_west"] = bomb
        if dest_obs is not None:
            save_rgb_png(dest_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_bomb_dest.png")
        snap = read_snapshot(env.get_ram())
        if int(snap.screen) != ROOM30:
            report["error"] = (
                f"0x31 bomb-west dest 0x{snap.screen:02X} is not 0x30 "
                f"phase={(bomb.get('controller') or {}).get('phase')}"
            )
            return report

        dest = take_stairs_from_source(
            env, ROOM30, total=total, cellar_side="right", assist=assist
        )
        report["walk"] = dest
        snap = read_snapshot(env.get_ram())
        idle = _idle(env, 1, assist=assist, total=total)
        save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_after_walk.png")
        for _ in range(180):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and int(snap.screen) == ROOM04 and not snap.transitioning:
                break
            idle = _step(env, nes_action("UP"), assist=assist, total=total)
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE:
            idle = _idle(env, 16, assist=assist, total=total)
            snap = read_snapshot(env.get_ram())
        report["cellar_room"] = 0x67
        if int(snap.screen) != ROOM04 or snap.mode != PLAY_MODE:
            report["error"] = (
                f"0x21->0x31->0x30 stairs dest 0x{snap.screen:02X} mode {snap.mode} "
                f"is not play 0x04 (passage={dest.get('passage_entered')})"
            )
            save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_stairs_dest.png")
            return report
        save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_stairs_dest.png")

        cooldown = 0
        for _ in range(800):
            snap = read_snapshot(env.get_ram())
            if in_room_04(snap) and not live_combat_objects(snap):
                break
            if not in_room_04(snap):
                break
            frame, cooldown = chase_sword_step(snap, cooldown)
            _step(env, frame.action, assist=assist, total=total)

        bomb04 = _run_bomb_west_04(env, total=total, assist=assist)
        bomb04.pop("after_bomb_obs", None)
        bomb04.pop("obs", None)
        report["bomb_west_04"] = bomb04
        snap = read_snapshot(env.get_ram())
        if int(snap.screen) != ROOM03:
            report["error"] = (
                f"0x04 bomb-west dest 0x{snap.screen:02X} is not 0x03 "
                f"phase={(bomb04.get('controller') or {}).get('phase')}"
            )
            return report

        _idle(env, 45, assist=assist, total=total)
        _step(env, nes_action("START"), assist=assist, total=total)
        _idle(env, 40, assist=assist, total=total)
        pause_moves = 0
        for _ in range(8):
            selected = int(env.get_ram()[ADDR_SELECTED_ITEM])
            if selected == B_ITEM_ARROWS:
                break
            _step(env, nes_action("RIGHT"), assist=assist, total=total)
            _idle(env, 8, assist=assist, total=total)
            pause_moves += 1
        _step(env, nes_action("START"), assist=assist, total=total)
        _idle(env, 40, assist=assist, total=total)
        selected = int(env.get_ram()[ADDR_SELECTED_ITEM])
        report["selected_item_after_pause"] = selected
        report["pause_right_moves"] = pause_moves
        if selected != B_ITEM_ARROWS:
            report["error"] = f"pause menu left selected_item={selected}, need arrows"
            return report

        for x, y, y_first in ((208, 189, True), (32, 189, False)):
            for _ in range(WALK_MAX_FRAMES):
                snap = read_snapshot(env.get_ram())
                if snap.screen != ROOM03:
                    break
                frame = walk_to_step(snap, x, y, y_first=y_first, tol=0)
                if frame.reason == "walk_arrived":
                    break
                _step(env, frame.action, assist=assist, total=total)
        report["west_south"] = dest_report(read_snapshot(env.get_ram()))

        dest03 = walk_play_room_to_patra(
            env,
            ROOM03,
            total=total,
            cellar_side="left",
            assist=assist,
        )
        report["walk_04"] = dest03
        snap = read_snapshot(env.get_ram())
        idle_obs = _idle(env, 1, assist=None, total=total)
        save_rgb_png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_entry.png")
        if not landed_final_patra(snap):
            report["error"] = (
                f"0x21->0x31->0x30->0x04->0x03 stairs settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={dest03.get('final_patra_live')}"
            )
            return report
        if save_checkpoints:
            path = _save_checkpoint(
                env,
                "Level9Room21SouthReconFixture",
                source_state=FIXTURE_SOURCE,
                phase="play_0x21_south_into_live_patra",
                result={
                    "ok": True,
                    "source_room": ROOM21,
                    "via": "south_shutter",
                    "room": 0x52,
                    "final_patra_live": True,
                    "frames": total[0],
                },
                fixture_writes=writes,
                bead=BEAD,
            )
            report["checkpoint_path"] = str(path)

        suffix = run_suffix_from_live_env(
            env,
            assist=assist,
            total=total,
            tag=tag,
            trial_i=trial_i,
            start_state="play_0x21_south_walk",
        )
        report["suffix"] = suffix
        report["ok"] = bool(suffix.get("ok"))
        report["credits_reached"] = suffix.get("credits_reached")
        report["total_frames"] = total[0]
        if not report["ok"]:
            report["error"] = suffix.get("error") or "suffix failed"
        return report
    finally:
        env.close()
