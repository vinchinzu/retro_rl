"""Level 9 stair-source probes and room 0x13 / 0x04 / play-tile dumps."""

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
    _run_bomb_west_04,
    dump_room_tiles,
    enter_patra_via_source_cellar,
    materialize_stair_room,
    take_stairs_from_source,
)

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

def probe_sources(
    *,
    tag: str = f"{TAG}_probe",
    stop_on_patra: bool = True,
    cellar_sides: tuple[str, ...] = ("left", "right"),
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "rom_pairs": [
            [f"0x{a:02X}", f"0x{b:02X}"] for a, b in LEVEL9_STAIR_PAIRS
        ],
        "sources": [],
        "winner": None,
    }
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        for room in LEVEL9_STAIR_SOURCES:
            room_row: dict[str, Any] = {
                "source": f"0x{room:02X}",
                "paired_hypothesis": (
                    None
                    if paired_stair_dest(room) is None
                    else f"0x{paired_stair_dest(room):02X}"
                ),
                "attempts": [],
            }
            winner_here = False
            for side in cellar_sides:
                env.close()
                env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
                total = [0]
                obs, loader, loaded = materialize_stair_room(env, room, total=total)
                attempt: dict[str, Any] = {
                    "cellar_side": side,
                    "loader": loader.label,
                    "from_room": loader.from_room,
                    "loaded": loaded,
                    "frames": total[0],
                }
                if not loaded:
                    attempt["error"] = "loader did not settle"
                    attempt["final"] = compact_snapshot(read_snapshot(env.get_ram()))
                    room_row["attempts"].append(attempt)
                    print(f"0x{room:02X} {side} LOAD FAIL")
                    continue
                settle = dest_report(read_snapshot(env.get_ram()))
                attempt["settled"] = settle
                settle_png = RECORDINGS_DIR / f"{tag}_0x{room:02x}_settle.png"
                save_rgb_png(obs, settle_png)
                attempt["settle_png"] = str(settle_png)
                dest = take_stairs_from_source(
                    env, room, total=total, cellar_side=side
                )
                attempt["dest"] = dest
                attempt["frames"] = total[0]
                dest_png = RECORDINGS_DIR / f"{tag}_0x{room:02x}_{side}_dest.png"
                idle_obs = _idle(env, 1, assist=None, total=total)
                save_rgb_png(idle_obs, dest_png)
                attempt["dest_png"] = str(dest_png)
                room_row["attempts"].append(attempt)
                print(
                    f"0x{room:02X} {side} -> "
                    f"SCREEN=0x{dest['screen']:02X} NEXT=0x{dest['next_screen']:02X} "
                    f"mode={dest['mode']} patra={dest['final_patra_live']} "
                    f"eyes={dest['patra_eyes']} objs="
                    f"{[o['type_name'] for o in dest['objects'][:8]]}"
                )
                if dest.get("landed_final_patra"):
                    room_row["winner"] = True
                    report["winner"] = {
                        "source": f"0x{room:02X}",
                        "cellar_side": side,
                        "dest": dest,
                        "settle_png": str(settle_png),
                        "dest_png": str(dest_png),
                        "loader": loader.label,
                    }
                    winner_here = True
                    break
            report["sources"].append(room_row)
            if winner_here and stop_on_patra:
                report["ok"] = True
                break
    finally:
        env.close()
    if report["winner"] is None:
        report["error"] = "no stair source landed live final Patra 0x52"
    return report


def probe_cellar_dest_table(*, tag: str = f"{TAG}_dest_table") -> dict[str, Any]:
    """InitMode9 each cellar RoomId, take both mouths, record live dest.

    Dest is CheckSubroom AttrsA/B — not a NEXT_SCREEN poke.  Play-room walk-on
    stair tiles are still unfound; this table is the engine dest truth.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    rooms = list(LEVEL9_CELLAR_ROOMS) + [r for r in LEVEL9_STAIR_SOURCES if r not in LEVEL9_CELLAR_ROOMS]
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "note": (
            "cellar dest via InitMode9 + CheckSubroom mouth UP; "
            "play-room stair tile that enters the cellar is still unfound"
        ),
        "sources": [],
        "winner": None,
    }
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        for room in rooms:
            room_row: dict[str, Any] = {
                "source": f"0x{room:02X}",
                "in_cellar_array": room in LEVEL9_CELLAR_ROOMS,
                "rom_left": cellar_dest_for(room, side="left"),
                "rom_right": cellar_dest_for(room, side="right"),
                "attempts": [],
            }
            for side in ("left", "right"):
                env.close()
                env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
                total = [0]
                obs, loader, loaded = materialize_stair_room(env, room, total=total)
                attempt: dict[str, Any] = {
                    "cellar_side": side,
                    "loader": loader.label,
                    "loaded": loaded,
                    "rom_dest": cellar_dest_for(room, side=side),
                }
                if not loaded:
                    attempt["error"] = "loader did not settle"
                    attempt["final"] = compact_snapshot(read_snapshot(env.get_ram()))
                    room_row["attempts"].append(attempt)
                    print(f"0x{room:02X} {side} LOAD FAIL")
                    continue
                settle_png = RECORDINGS_DIR / f"{tag}_0x{room:02x}_settle.png"
                save_rgb_png(obs, settle_png)
                attempt["settle"] = dest_report(read_snapshot(env.get_ram()))
                attempt["settle_png"] = str(settle_png)
                dest = enter_patra_via_source_cellar(
                    env, room, total=total, side=side
                )
                dest_png = RECORDINGS_DIR / f"{tag}_0x{room:02x}_{side}_dest.png"
                idle_obs = _idle(env, 1, assist=None, total=total)
                save_rgb_png(idle_obs, dest_png)
                attempt["dest"] = dest
                attempt["dest_png"] = str(dest_png)
                attempt["frames"] = total[0]
                room_row["attempts"].append(attempt)
                print(
                    f"0x{room:02X} {side} -> "
                    f"SCREEN=0x{dest['screen']:02X} NEXT=0x{dest['next_screen']:02X} "
                    f"mode={dest['mode']} patra={dest['final_patra_live']} "
                    f"eyes={dest['patra_eyes']}"
                )
                if dest.get("landed_final_patra") and report["winner"] is None:
                    report["winner"] = {
                        "source": f"0x{room:02X}",
                        "cellar_side": side,
                        "dest": dest,
                        "dest_png": str(dest_png),
                    }
            report["sources"].append(room_row)
    finally:
        env.close()
    report["ok"] = report["winner"] is not None
    return report


def _room13_report(snap: Any) -> dict[str, Any]:
    row = dest_report(snap)
    row["cur_opened_doors"] = row["doors"]
    row["open_doorway_mask"] = row["mask"]
    return row


def _probe_room13_north(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    """Controller-only north push. Does not write 0x0F/0x0F on 0x13."""
    start = read_snapshot(env.get_ram())
    transition = None
    for _ in range(NORTH_PROBE_FRAMES):
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM13 and not snap.transitioning:
            transition = {
                "from_room": ROOM13,
                "direction": "UP",
                "to_room": int(snap.screen),
                "mode": int(snap.mode),
                "objects": dest_report(snap)["objects"],
            }
            break
        frame = room13_to_03_step(snap)
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
        "entered_0x03": int(end.screen) == ROOM03,
        "transition": transition,
        "still_in_13": bool(in_room_13(end)),
        "stuck_y": int(end.link_y),
    }


def dump_room_13(*, tag: str = "l9_room13_dump") -> dict[str, Any]:
    """Live 0x13 dump: objects, doors, how UP opens. No InitMode9."""
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
        "room": "0x13",
        "rom_doors": {
            "0x13": {
                "north": ROOM13_ROM_NORTH,
                "south": ROOM13_ROM_SOUTH,
                "west": ROOM13_ROM_WEST,
                "east": ROOM13_ROM_EAST,
                "north_name": "wall",
                "south_name": "key",
                "west_name": "key",
                "east_name": "wall",
            },
            "0x03": {
                "north": ROOM03_ROM_NORTH,
                "south": ROOM03_ROM_SOUTH,
                "west": ROOM03_ROM_WEST,
                "east": ROOM03_ROM_EAST,
                "north_name": "wall",
                "south_name": "wall",
                "west_name": "wall",
                "east_name": "bomb",
            },
        },
        "how_up_opens": "sealed_wall",
        "clean_walk": False,
        "clean_predecessor": room13_is_clean_predecessor_of_03(),
    }

    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        total = [0]
        obs, loader, loaded = materialize_stair_room(
            env, ROOM13, total=total, door_staging=True
        )
        report["loader"] = {
            "label": loader.label,
            "from_room": f"0x{loader.from_room:02X}",
            "direction": loader.direction,
            "link_xy": [loader.link_x, loader.link_y],
            "door_staging_on_from_room": True,
            "note": "0x0F/0x0F is written on 0x23 so the key-north scroll can start; 0x13 door bits after settle are the game loader's.",
        }
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x13"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        snap = read_snapshot(env.get_ram())
        report["settled"] = _room13_report(snap)
        report["link_start"] = {"x": snap.link_x, "y": snap.link_y}
        settle_png = RECORDINGS_DIR / f"{tag}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

        north = _probe_room13_north(env, total=total, assist=assist)
        report["north_probe_uncleared"] = north
        idle = _idle(env, 1, assist=assist, total=total)
        north_png = RECORDINGS_DIR / f"{tag}_north_probe.png"
        save_rgb_png(idle, north_png)
        report["north_probe_png"] = str(north_png)

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, _, loaded = materialize_stair_room(
            env, ROOM13, total=total, door_staging=True
        )
        if loaded:
            cooldown = 0
            for _ in range(CLEAR_PROBE_FRAMES):
                snap = read_snapshot(env.get_ram())
                if in_room_13(snap) and not live_combat_objects(snap):
                    break
                frame, cooldown = chase_sword_step(snap, cooldown)
                obs = _step(env, frame.action, assist=assist, total=total)
            obs = _idle(env, 16, assist=assist, total=total)
            cleared = read_snapshot(env.get_ram())
            report["after_clear"] = _room13_report(cleared)
            clear_png = RECORDINGS_DIR / f"{tag}_after_clear.png"
            save_rgb_png(obs, clear_png)
            report["after_clear_png"] = str(clear_png)
            north_clear = _probe_room13_north(env, total=total, assist=assist)
            report["north_probe_cleared"] = north_clear
            idle = _idle(env, 1, assist=assist, total=total)
            save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_north_after_clear.png")

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, loader_np, loaded_np = materialize_stair_room(
            env, ROOM13, total=total, door_staging=False
        )
        report["no_door_poke_settle"] = {
            "loaded": loaded_np,
            "loader": loader_np.label,
            "final": compact_snapshot(read_snapshot(env.get_ram())),
        }
        if loaded_np:
            no_poke_png = RECORDINGS_DIR / f"{tag}_no_door_poke_settle.png"
            save_rgb_png(obs, no_poke_png)
            report["no_door_poke_settle"]["settled"] = _room13_report(
                read_snapshot(env.get_ram())
            )
            report["no_door_poke_settle"]["png"] = str(no_poke_png)

        entered = bool(
            (report.get("north_probe_uncleared") or {}).get("entered_0x03")
            or (report.get("north_probe_cleared") or {}).get("entered_0x03")
        )
        report["clean_walk"] = bool(entered and loaded_np)
        report["how_up_opens"] = (
            "already_open"
            if entered and (report["settled"]["doors"].get("north") or loaded_np)
            else "kill_clear"
            if (report.get("north_probe_cleared") or {}).get("entered_0x03")
            else "needs_door_poke_fake_scroll"
        )
        report["ok"] = bool(loaded)
        report["disproof"] = (
            "0x13 north is ROM wall (1); 0x03 south is ROM wall (1). "
            "Controller UP after a no-0x13-door-poke settle stays in 0x13. "
            "The 0x03 loader's 0x0F/0x0F staging on 0x13 is a fake scroll, "
            "not a clean walk."
        )
        return report
    finally:
        env.close()


def dump_room_04(*, tag: str = "l9_room04_dump") -> dict[str, Any]:
    """Live 0x04 dump + bomb-west probe. Stages 0x14, never 0x03 doors."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    neighbors = room03_rom_neighbors()
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "room": "0x04",
        "rom_doors": {
            "0x04": {
                "north": ROOM04_ROM_NORTH,
                "south": ROOM04_ROM_SOUTH,
                "west": ROOM04_ROM_WEST,
                "east": ROOM04_ROM_EAST,
                "north_name": "wall",
                "south_name": "wall",
                "west_name": "bomb",
                "east_name": "wall",
            },
            "0x03": {
                "north": ROOM03_ROM_NORTH,
                "south": ROOM03_ROM_SOUTH,
                "west": ROOM03_ROM_WEST,
                "east": ROOM03_ROM_EAST,
                "north_name": "wall",
                "south_name": "wall",
                "west_name": "wall",
                "east_name": "bomb",
            },
        },
        "rom_neighbors_of_03": neighbors,
        "rom_west_is_bomb": room04_rom_west_is_bomb(),
        "rom_predecessor": room04_is_rom_predecessor_of_03(),
        "bomb_stand": list(ROOM04_BOMB_WEST_STAND),
        "selected_item_fixture": B_ITEM_BOMBS,
        "door_poke_on_03": False,
    }

    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        total = [0]
        obs, loader, loaded = materialize_stair_room(
            env, ROOM04, total=total, door_staging=True, selected_item=B_ITEM_BOMBS
        )
        report["loader"] = {
            "label": loader.label,
            "from_room": f"0x{loader.from_room:02X}",
            "direction": loader.direction,
            "link_xy": [loader.link_x, loader.link_y],
            "door_staging_on_from_room": True,
            "from_room_is_03": loader.from_room == ROOM03,
            "note": "0x0F/0x0F is written on 0x14 so the south-wall scroll can start; 0x03 doors are not poked.",
        }
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x04"
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
            env, ROOM04, total=total, door_staging=True, selected_item=B_ITEM_BOMBS
        )
        if not loaded:
            report["error"] = "rematerialize failed 0x04"
            return report
        cooldown = 0
        for _ in range(800):
            snap = read_snapshot(env.get_ram())
            if in_room_04(snap) and not live_combat_objects(snap):
                break
            if not in_room_04(snap):
                break
            frame, cooldown = chase_sword_step(snap, cooldown)
            obs = _step(env, frame.action, assist=assist, total=total)
        obs = _idle(env, 12, assist=assist, total=total)
        report["after_clear"] = dest_report(read_snapshot(env.get_ram()))

        bomb = _run_bomb_west_04(env, total=total, assist=assist)
        after_bomb_obs = bomb.pop("after_bomb_obs")
        dest_obs = bomb.pop("obs")
        report["bomb_west"] = bomb
        if after_bomb_obs is not None:
            after_png = RECORDINGS_DIR / f"{tag}_after_bomb.png"
            save_rgb_png(after_bomb_obs, after_png)
            report["after_bomb_png"] = str(after_png)
        if dest_obs is not None:
            dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
            save_rgb_png(dest_obs, dest_png)
            report["dest_png"] = str(dest_png)
        dest_snap = read_snapshot(env.get_ram())
        report["dest_screen"] = int(dest_snap.screen)
        report["lands_0x03"] = int(dest_snap.screen) == ROOM03
        if report["lands_0x03"]:
            _assign(env, ADDR_LINK_X, ROOM03_STAIR_X)
            _assign(env, ADDR_LINK_Y, ROOM03_STAIR_Y)
            _idle(env, 1, assist=None, total=total)
            stair = read_snapshot(env.get_ram())
            report["stair_tile_at_03"] = {
                "x": ROOM03_STAIR_X,
                "y": ROOM03_STAIR_Y,
                "tile": int(stair.colliding_tile),
                "still_0x72": int(stair.colliding_tile) == 0x72,
                "screen": int(stair.screen),
            }

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, loader_np, loaded_np = materialize_stair_room(
            env, ROOM04, total=total, door_staging=False, selected_item=B_ITEM_BOMBS
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

        report["ok"] = bool(loaded)
        if not report["lands_0x03"]:
            report["next_candidate"] = (
                "0x04 west did not land 0x03. ROM neighbors of 0x03: "
                "south 0x13 wall (disproved), west 0x02 wall, east 0x04 bomb, "
                "no north (row 0). Next real candidate is cellar 0x67 "
                "(right mouth dest 0x04) / play 0x30, not a fake cardinal."
            )
        return report
    finally:
        env.close()


def dump_room_30(*, tag: str = "l9_room30_dump") -> dict[str, Any]:
    """Live 0x30 dump + stairs -> cellar 0x67 right dest. Stages 0x40, not 0x04."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    pair = cellar_for_play_room(ROOM30)
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "room": "0x30",
        "rom_doors": {
            "0x30": {
                "north": ROOM30_ROM_NORTH,
                "south": ROOM30_ROM_SOUTH,
                "west": ROOM30_ROM_WEST,
                "east": ROOM30_ROM_EAST,
                "north_name": "wall",
                "south_name": "key",
                "west_name": "wall",
                "east_name": "bomb",
                "secret": ROOM30_ROM_SECRET,
                "secret_name": "block_stairs",
            },
            "0x40": {
                "north": ROOM40_ROM_NORTH,
                "north_name": "key",
            },
            "0x04": {
                "north": ROOM04_ROM_NORTH,
                "south": ROOM04_ROM_SOUTH,
                "west": ROOM04_ROM_WEST,
                "east": ROOM04_ROM_EAST,
                "west_name": "bomb",
            },
        },
        "rom_secret_block_stairs": room30_rom_secret_is_block_stairs(),
        "loader_avoids_04": room30_loader_avoids_04(),
        "cellar_pair": (
            None
            if pair is None
            else {"cellar": f"0x{pair[0]:02X}", "mouth": pair[1]}
        ),
        "hypothesized_right_dest": "0x04",
        "door_poke_on_04": False,
    }

    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        total = [0]
        obs, loader, loaded = materialize_stair_room(
            env, ROOM30, total=total, door_staging=True
        )
        report["loader"] = {
            "label": loader.label,
            "from_room": f"0x{loader.from_room:02X}",
            "direction": loader.direction,
            "link_xy": [loader.link_x, loader.link_y],
            "door_staging_on_from_room": True,
            "from_room_is_04": loader.from_room == ROOM04,
            "note": "0x0F/0x0F is written on 0x40 so the key-north scroll can start; 0x04 doors are not poked.",
        }
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x30"
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
        report["stair_hits_before_push"] = tiles["stair_hits"]

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, _, loaded = materialize_stair_room(env, ROOM30, total=total)
        if not loaded:
            report["error"] = "rematerialize failed 0x30"
            return report

        dest = take_stairs_from_source(
            env, ROOM30, total=total, cellar_side="right", assist=assist
        )
        report["walk"] = dest
        snap = read_snapshot(env.get_ram())
        idle = _idle(env, 1, assist=assist, total=total)
        after_png = RECORDINGS_DIR / f"{tag}_after_walk.png"
        save_rgb_png(idle, after_png)
        report["after_walk_png"] = str(after_png)
        cellar_mode = stair_transition_modes(snap.mode) or snap.mode == CELLAR_MODE
        if cellar_mode or int(snap.screen) == 0x67:
            cellar_png = RECORDINGS_DIR / f"{tag}_cellar.png"
            save_rgb_png(idle, cellar_png)
            report["cellar_png"] = str(cellar_png)
        dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
        save_rgb_png(idle, dest_png)
        report["dest_png"] = str(dest_png)
        report["dest_screen"] = int(snap.screen)
        report["dest_mode"] = int(snap.mode)
        report["entered_cellar_67"] = bool(
            dest.get("passage_entered") or int(snap.screen) == 0x67 or cellar_mode
        )
        for _ in range(180):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and int(snap.screen) == ROOM04 and not snap.transitioning:
                break
            idle = _step(env, nes_action("UP"), assist=assist, total=total)
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE:
            idle = _idle(env, 16, assist=assist, total=total)
            snap = read_snapshot(env.get_ram())
            dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
            save_rgb_png(idle, dest_png)
            report["dest_png"] = str(dest_png)
        report["dest_screen"] = int(snap.screen)
        report["dest_mode"] = int(snap.mode)
        report["lands_0x04"] = int(snap.screen) == ROOM04 and snap.mode == PLAY_MODE
        report["walk_log"] = dest.get("log")
        if report["lands_0x04"]:
            report["stair_tile"] = 0x72
            report["stair_xy"] = [ROOM30_STAIR_X, ROOM30_STAIR_Y]
            report["cellar_room"] = "0x67"
            report["west_still_bomb"] = room04_rom_west_is_bomb()
            report["can_reuse_compose_04"] = True
        else:
            report["next_candidate"] = (
                f"0x30 / cellar 0x67 right dest is SCREEN=0x{int(snap.screen):02X} "
                f"mode={snap.mode}, not 0x04. Honest next: stairs in 0x04 back "
                f"into 0x67, or another play room whose Attrs match cellar 0x67."
            )
        report["ok"] = bool(loaded)
        return report
    finally:
        env.close()


def dump_play_rooms(
    *,
    rooms: tuple[int, ...] = PLAY_STAIR_CANDIDATES,
    tag: str = f"{TAG}_play_tiles",
) -> dict[str, Any]:
    """Materialize play rooms and dump colliding tiles. No InitMode9."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "note": "play-room tile dump; CheckWarps walk is separate",
        "checkwarps_77": [
            {"play": f"0x{room:02X}", "mouth": side}
            for room, side in play_rooms_entering_cellar(PATRA_STAIR_SOURCE)
        ],
        "rooms": [],
    }
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        for room in rooms:
            env.close()
            env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
            total = [0]
            obs, loader, loaded = materialize_stair_room(env, room, total=total)
            row: dict[str, Any] = {
                "room": f"0x{room:02X}",
                "loader": loader.label,
                "from_room": f"0x{loader.from_room:02X}",
                "loaded": loaded,
                "cellar_pair": (
                    None
                    if cellar_for_play_room(room) is None
                    else {
                        "cellar": f"0x{cellar_for_play_room(room)[0]:02X}",
                        "mouth": cellar_for_play_room(room)[1],
                    }
                ),
            }
            if not loaded:
                row["error"] = "loader did not settle"
                row["final"] = compact_snapshot(read_snapshot(env.get_ram()))
                report["rooms"].append(row)
                print(f"0x{room:02X} LOAD FAIL")
                continue
            settle = dest_report(read_snapshot(env.get_ram()))
            row["settled"] = settle
            settle_png = RECORDINGS_DIR / f"{tag}_0x{room:02x}_settle.png"
            save_rgb_png(obs, settle_png)
            row["settle_png"] = str(settle_png)
            tiles = dump_room_tiles(env, total=total)
            row["tiles"] = {
                "stair_hits": tiles["stair_hits"],
                "mouth_hits": tiles["mouth_hits"],
                "tile_counts": tiles["tile_counts"],
                "grid_origin": tiles["grid_origin"],
                "grid_step": tiles["grid_step"],
            }
            tile_json = RECORDINGS_DIR / f"{tag}_0x{room:02x}_tiles.json"
            write_json_report(tile_json, tiles)
            row["tile_json"] = str(tile_json)
            report["rooms"].append(row)
            print(
                f"0x{room:02X} loaded stairs={len(tiles['stair_hits'])} "
                f"mouths={len(tiles['mouth_hits'])} "
                f"top={list(tiles['tile_counts'].items())[:4]}"
            )
    finally:
        env.close()
    report["ok"] = any(r.get("loaded") for r in report["rooms"])
    return report
