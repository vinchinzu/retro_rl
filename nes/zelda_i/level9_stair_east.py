"""Level 9 play 0x30 / 0x40 compose-to-credits walks and 0x40 dump."""

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

def run_room30_stairs_to_credits(
    *,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = "l9_play30_cellar67_patra_credits_recon",
    trial_i: int = 0,
) -> dict[str, Any]:
    """One continuous trial: 0x30 stairs -> cellar 0x67 right -> 0x04 suffix.

    Fixture inventory + 0x40 neighbor-scroll happen before the walk.
    After materialize, no object / room / door / progression / capacity writes.
    0x04 doors are never poked. InitMode9 is not used.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    loader = stair_loader_for(ROOM30)
    writes = _loader_write_rows(loader)
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "source_room": "0x30",
        "via": "cellar_0x67_right",
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
        "door_poke_on_04": False,
    }
    try:
        obs, used, loaded = materialize_stair_room(
            env, ROOM30, total=total, selected_item=B_ITEM_BOMBS
        )
        report["loader"] = used.label
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x30"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        report["settled"] = dest_report(read_snapshot(env.get_ram()))
        settle_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

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
        report["dest_screen"] = int(snap.screen)
        report["cellar_room"] = 0x67 if dest.get("passage_entered") or in_cellar_67(snap) else 0x67
        if int(snap.screen) != ROOM04 or snap.mode != PLAY_MODE:
            report["error"] = (
                f"0x30 stairs dest 0x{snap.screen:02X} mode {snap.mode} "
                f"is not play 0x04 (passage={dest.get('passage_entered')})"
            )
            save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")
            return report
        save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")

        cooldown = 0
        for _ in range(800):
            snap = read_snapshot(env.get_ram())
            if in_room_04(snap) and not live_combat_objects(snap):
                break
            if not in_room_04(snap):
                break
            frame, cooldown = chase_sword_step(snap, cooldown)
            _step(env, frame.action, assist=assist, total=total)

        bomb = _run_bomb_west_04(env, total=total, assist=assist)
        bomb.pop("after_bomb_obs", None)
        bomb.pop("obs", None)
        report["bomb_west"] = bomb
        snap = read_snapshot(env.get_ram())
        if int(snap.screen) != ROOM03:
            report["error"] = (
                f"bomb-west dest 0x{snap.screen:02X} is not 0x03 "
                f"phase={(bomb.get('controller') or {}).get('phase')}"
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

        # Same 0x03 recovery as --compose-04: east hole is on stair-Y;
        # seed west-south (32,189) then the accepted play-source 03 walk.
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
                f"0x30->0x04->0x03 stairs settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={dest03.get('final_patra_live')}"
            )
            return report
        if save_checkpoints:
            path = _save_checkpoint(
                env,
                "Level9Room30StairsReconFixture",
                source_state=FIXTURE_SOURCE,
                phase="play_0x30_cellar_0x67_right_into_live_patra",
                result={
                    "ok": True,
                    "source_room": ROOM30,
                    "via": "cellar_0x67_right",
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
            start_state="play_0x30_cellar_0x67_right_walk",
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


def _probe_room40_north(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    """Controller-only north key push. Does not write 0x0F/0x0F on 0x40 or 0x30."""
    start = read_snapshot(env.get_ram())
    transition = None
    for _ in range(NORTH_PROBE_FRAMES + 800):
        snap = read_snapshot(env.get_ram())
        if (
            snap.screen == ROOM30
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            transition = {
                "from_room": ROOM40,
                "direction": "UP",
                "to_room": int(snap.screen),
                "mode": int(snap.mode),
                "objects": dest_report(snap)["objects"],
            }
            break
        if (
            snap.screen != ROOM40
            or snap.mode != PLAY_MODE
            or snap.transitioning
        ):
            _step(env, nes_action("UP"), assist=assist, total=total)
            continue
        frame = room40_to_30_step(snap)
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
        "entered_0x30": int(end.screen) == ROOM30,
        "transition": transition,
        "still_in_40": bool(in_room_40(end)),
        "stuck_y": int(end.link_y),
    }


def dump_room_40(*, tag: str = "l9_room40_dump") -> dict[str, Any]:
    """Live 0x40 dump + key-north probe. Stages 0x50, never 0x30 doors."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    neighbors = room30_rom_neighbors()
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "room": "0x40",
        "rom_doors": {
            "0x40": {
                "north": ROOM40_ROM_NORTH,
                "south": ROOM40_ROM_SOUTH,
                "west": ROOM40_ROM_WEST,
                "east": ROOM40_ROM_EAST,
                "north_name": "key",
                "south_name": "key",
                "west_name": "wall",
                "east_name": "wall",
                "secret": 7,
                "secret_name": "foes_item",
            },
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
            "0x20": {"south": ROOM20_ROM_SOUTH, "south_name": "wall"},
            "0x31": {"west": ROOM31_ROM_WEST, "west_name": "bomb"},
        },
        "rom_neighbors_of_30": neighbors,
        "rom_north_is_key": room40_rom_north_is_key(),
        "rom_predecessor": room40_is_rom_predecessor_of_30(),
        "loader_avoids_30": room40_loader_avoids_30(),
        "door_poke_on_30": False,
        "cellar_67_is_successor_not_pred": True,
    }

    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        total = [0]
        obs, loader, loaded = materialize_stair_room(
            env, ROOM40, total=total, door_staging=True
        )
        report["loader"] = {
            "label": loader.label,
            "from_room": f"0x{loader.from_room:02X}",
            "direction": loader.direction,
            "link_xy": [loader.link_x, loader.link_y],
            "door_staging_on_from_room": True,
            "from_room_is_30": loader.from_room == ROOM30,
            "note": "0x0F/0x0F is written on 0x50 so the south-wall scroll can start; 0x30 doors are not poked.",
        }
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x40"
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
            env, ROOM40, total=total, door_staging=True
        )
        if not loaded:
            report["error"] = "rematerialize failed 0x40 before north"
            return report

        north = _probe_room40_north(env, total=total, assist=assist)
        report["north_probe_uncleared"] = north
        idle = _idle(env, 16, assist=assist, total=total)
        north_png = RECORDINGS_DIR / f"{tag}_north_probe.png"
        save_rgb_png(idle, north_png)
        report["north_probe_png"] = str(north_png)
        dest_snap = read_snapshot(env.get_ram())
        if int(dest_snap.screen) == ROOM30 and dest_snap.mode == PLAY_MODE:
            report["dest_screen"] = int(dest_snap.screen)
            report["dest_mode"] = int(dest_snap.mode)
            report["lands_0x30"] = True
            dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
            save_rgb_png(idle, dest_png)
            report["dest_png"] = str(dest_png)
            _assign(env, ADDR_LINK_X, ROOM30_STAIR_X)
            _assign(env, ADDR_LINK_Y, ROOM30_STAIR_Y)
            _idle(env, 1, assist=None, total=total)
            stair = read_snapshot(env.get_ram())
            report["stair_tile_at_30"] = {
                "x": ROOM30_STAIR_X,
                "y": ROOM30_STAIR_Y,
                "tile": int(stair.colliding_tile),
                "still_0x73": int(stair.colliding_tile) == 0x73,
                "screen": int(stair.screen),
            }
            dest_objs = dest_report(dest_snap)["objects"]
            report["dest_objects"] = dest_objs
            report["block_68_present"] = any(
                int(obj.get("type_id") or 0) == 0x68 for obj in dest_objs
            )
            report["block_stairs_still_works"] = bool(report["block_68_present"])

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, _, loaded = materialize_stair_room(
            env, ROOM40, total=total, door_staging=True
        )
        if loaded:
            cooldown = 0
            for _ in range(CLEAR_PROBE_FRAMES):
                snap = read_snapshot(env.get_ram())
                if in_room_40(snap) and not live_combat_objects(snap):
                    break
                if not in_room_40(snap):
                    break
                frame, cooldown = chase_sword_step(snap, cooldown)
                obs = _step(env, frame.action, assist=assist, total=total)
            obs = _idle(env, 16, assist=assist, total=total)
            cleared = read_snapshot(env.get_ram())
            report["after_clear"] = dest_report(cleared)
            clear_png = RECORDINGS_DIR / f"{tag}_after_clear.png"
            save_rgb_png(obs, clear_png)
            report["after_clear_png"] = str(clear_png)
            north_clear = _probe_room40_north(env, total=total, assist=assist)
            report["north_probe_cleared"] = north_clear
            idle = _idle(env, 1, assist=assist, total=total)
            clear_dest_png = RECORDINGS_DIR / f"{tag}_cleared_dest.png"
            save_rgb_png(idle, clear_dest_png)
            report["cleared_dest_png"] = str(clear_dest_png)
            dest_snap = read_snapshot(env.get_ram())
            report["cleared_dest_screen"] = int(dest_snap.screen)
            report["cleared_dest_mode"] = int(dest_snap.mode)
            if int(dest_snap.screen) == ROOM30 and dest_snap.mode == PLAY_MODE:
                report["dest_screen"] = int(dest_snap.screen)
                report["dest_mode"] = int(dest_snap.mode)
                report["lands_0x30"] = True
                dest_png = RECORDINGS_DIR / f"{tag}_dest.png"
                save_rgb_png(idle, dest_png)
                report["dest_png"] = str(dest_png)
                _assign(env, ADDR_LINK_X, ROOM30_STAIR_X)
                _assign(env, ADDR_LINK_Y, ROOM30_STAIR_Y)
                _idle(env, 1, assist=None, total=total)
                stair = read_snapshot(env.get_ram())
                report["stair_tile_at_30"] = {
                    "x": ROOM30_STAIR_X,
                    "y": ROOM30_STAIR_Y,
                    "tile": int(stair.colliding_tile),
                    "still_0x73": int(stair.colliding_tile) == 0x73,
                    "screen": int(stair.screen),
                }

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, loader_np, loaded_np = materialize_stair_room(
            env, ROOM40, total=total, door_staging=False
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
            (report.get("north_probe_uncleared") or {}).get("entered_0x30")
            or (report.get("north_probe_cleared") or {}).get("entered_0x30")
        )
        report["clean_walk"] = bool(entered)
        report["how_up_opens"] = (
            "already_open"
            if entered and (report["settled"]["doors"].get("north"))
            else "key"
            if entered
            else "sealed_wall"
        )
        if not entered:
            report["next_candidate"] = (
                "0x40 north did not land 0x30. ROM neighbors of 0x30: "
                "north 0x20 wall, west 0x2F shutter/wall, east 0x31 bomb, "
                "south 0x40 key. Next real candidate is 0x31 west bomb."
            )
        report["ok"] = bool(loaded)
        return report
    finally:
        env.close()


def run_room40_key_north_to_credits(
    *,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = "l9_play40_keynorth_patra_credits_recon",
    trial_i: int = 0,
) -> dict[str, Any]:
    """One continuous trial: 0x40 key-north -> 0x30 stairs suffix.

    Fixture inventory + 0x50 neighbor-scroll happen before the walk.
    After materialize, no object / room / door / progression / capacity writes.
    0x30 doors are never poked. InitMode9 is not used.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    loader = stair_loader_for(ROOM40)
    writes = _loader_write_rows(loader)
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "source_room": "0x40",
        "via": "key_north",
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
        "door_poke_on_30": False,
    }
    try:
        obs, used, loaded = materialize_stair_room(
            env, ROOM40, total=total, selected_item=B_ITEM_BOMBS
        )
        report["loader"] = used.label
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x40"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        report["settled"] = dest_report(read_snapshot(env.get_ram()))
        settle_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

        # Live: door-column UP from the south alcove opens the key door.
        # Kill-clear first leaves Link in the plus and misses 0x30.
        north = _probe_room40_north(env, total=total, assist=assist)
        report["north_walk"] = north
        snap = read_snapshot(env.get_ram())
        idle = _idle(env, 16, assist=assist, total=total)
        save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")
        report["dest_screen"] = int(snap.screen)
        if int(snap.screen) != ROOM30 or snap.mode != PLAY_MODE:
            report["error"] = (
                f"0x40 key-north dest 0x{snap.screen:02X} mode {snap.mode} "
                f"is not play 0x30 (entered={north.get('entered_0x30')})"
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
        report["cellar_dest_screen"] = int(snap.screen)
        report["cellar_room"] = 0x67
        if int(snap.screen) != ROOM04 or snap.mode != PLAY_MODE:
            report["error"] = (
                f"0x40->0x30 stairs dest 0x{snap.screen:02X} mode {snap.mode} "
                f"is not play 0x04 (passage={dest.get('passage_entered')})"
            )
            save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_cellar_dest.png")
            return report
        save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_cellar_dest.png")

        cooldown = 0
        for _ in range(800):
            snap = read_snapshot(env.get_ram())
            if in_room_04(snap) and not live_combat_objects(snap):
                break
            if not in_room_04(snap):
                break
            frame, cooldown = chase_sword_step(snap, cooldown)
            _step(env, frame.action, assist=assist, total=total)

        bomb = _run_bomb_west_04(env, total=total, assist=assist)
        bomb.pop("after_bomb_obs", None)
        bomb.pop("obs", None)
        report["bomb_west"] = bomb
        snap = read_snapshot(env.get_ram())
        if int(snap.screen) != ROOM03:
            report["error"] = (
                f"bomb-west dest 0x{snap.screen:02X} is not 0x03 "
                f"phase={(bomb.get('controller') or {}).get('phase')}"
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
            room03_chase_mode="blocking",
        )
        report["walk_04"] = dest03
        snap = read_snapshot(env.get_ram())
        idle_obs = _idle(env, 1, assist=None, total=total)
        save_rgb_png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_entry.png")
        if not landed_final_patra(snap):
            report["error"] = (
                f"0x40->0x30->0x04->0x03 stairs settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={dest03.get('final_patra_live')}"
            )
            return report
        if save_checkpoints:
            path = _save_checkpoint(
                env,
                "Level9Room40KeyNorthReconFixture",
                source_state=FIXTURE_SOURCE,
                phase="play_0x40_key_north_into_live_patra",
                result={
                    "ok": True,
                    "source_room": ROOM40,
                    "via": "key_north",
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
            start_state="play_0x40_key_north_walk",
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
