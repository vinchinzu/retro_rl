"""Materialize L9 stair sources and find the live drop into Patra 0x52.

Backwards-development tool.  Fixture inventory + room-loader setup stays
route-ineligible.  After a continuous run starts there are no object / room /
door / progression / capacity writes.

Examples::

    uv run python nes/zelda_i/scripts/run_level9_stairs.py --probe
    uv run python nes/zelda_i/scripts/run_level9_stairs.py \
      --build-fixture --infinite-life --save-state --trials 2 \
      --tag l9_stairXX_patra_credits_recon
"""

from __future__ import annotations

import argparse
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


def _loader_write_rows(loader: StairLoader) -> list[dict[str, Any]]:
    rows = [
        {
            "name": name,
            "address": address,
            "address_hex": f"0x{address:04X}",
            "value": value,
        }
        for name, address, value in FULL_LOADOUT
    ]
    rows.extend(
        [
            {
                "name": "loader_level",
                "address": ADDR_LEVEL,
                "address_hex": "0x0010",
                "value": LEVEL9,
            },
            {
                "name": "loader_mode",
                "address": ADDR_MODE,
                "address_hex": "0x0012",
                "value": PLAY_MODE,
            },
            {
                "name": "loader_current_room",
                "address": ADDR_SCREEN,
                "address_hex": "0x00EB",
                "value": loader.from_room,
            },
            {
                "name": "loader_next_room",
                "address": ADDR_NEXT_SCREEN,
                "address_hex": "0x00EC",
                "value": loader.room,
            },
            {
                "name": "loader_link_position",
                "addresses": [ADDR_LINK_X, ADDR_LINK_Y],
                "address_hex": ["0x0070", "0x0084"],
                "values": [loader.link_x, loader.link_y],
            },
            {
                "name": "loader_door_staging",
                "addresses": [ADDR_CUR_OPENED_DOORS, ADDR_OPEN_DOORWAY_MASK],
                "address_hex": ["0x00EE", "0x033F"],
                "values": [0x0F, 0x0F],
            },
            {
                "name": "loader_hold_direction",
                "value": loader.direction,
                "from_room": loader.from_room,
                "to_room": loader.room,
                "label": loader.label,
            },
        ]
    )
    return rows


def _apply_loader(
    env: Any,
    loader: StairLoader,
    *,
    door_staging: bool = True,
    selected_item: int | None = None,
) -> None:
    for _, address, value in FULL_LOADOUT:
        _assign(env, address, value)
    pairs = [
        (ADDR_LEVEL, LEVEL9),
        (ADDR_MODE, PLAY_MODE),
        (ADDR_SCREEN, loader.from_room),
        (ADDR_NEXT_SCREEN, loader.room),
        (ADDR_LINK_X, loader.link_x),
        (ADDR_LINK_Y, loader.link_y),
    ]
    if door_staging:
        pairs.extend(
            (
                (ADDR_CUR_OPENED_DOORS, 0x0F),
                (ADDR_OPEN_DOORWAY_MASK, 0x0F),
            )
        )
    if selected_item is not None:
        pairs.append((ADDR_SELECTED_ITEM, int(selected_item) & 0xFF))
    for address, value in pairs:
        _assign(env, address, value)


def _hold_until_room(
    env: Any,
    loader: StairLoader,
    *,
    total: list[int],
    max_frames: int = LOAD_MAX_FRAMES,
):
    obs = None
    for _ in range(max_frames):
        obs = _step(env, nes_action(loader.direction), assist=None, total=total)
        if in_stair_source(read_snapshot(env.get_ram()), loader.room):
            return obs, True
    return obs, False


def materialize_stair_room(
    env: Any,
    room: int,
    *,
    total: list[int],
    door_staging: bool = True,
    selected_item: int | None = None,
) -> tuple[Any, StairLoader, bool]:
    loader = stair_loader_for(room)
    reset_obs(env)
    _apply_loader(
        env, loader, door_staging=door_staging, selected_item=selected_item
    )
    obs, loaded = _hold_until_room(env, loader, total=total)
    if loaded:
        obs = _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=total)
    return obs, loader, loaded


def _left_source(snap: Any, source: int) -> bool:
    return (
        snap.screen != int(source)
        and not snap.transitioning
        and snap.mode not in (6, 7)
    )


def _exit_cellar(env: Any, *, total: list[int], side: str, max_frames: int = CELLAR_MAX_FRAMES):
    obs = None
    start = read_snapshot(env.get_ram())
    start_room = int(start.screen)
    start_xy = (int(start.link_x), int(start.link_y))
    placed = False
    for i in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if (
            snap.mode == PLAY_MODE
            and snap.level == LEVEL9
            and snap.screen != start_room
            and not snap.transitioning
        ):
            return obs, snap
        if snap.mode == PLAY_MODE and snap.screen == start_room and not stair_transition_modes(start.mode):
            # Dropped back to the same play room without a dest change.
            if i > 30:
                return obs, snap
        # Mode 9 can flip before LayoutCellar places Link in the stairwell.
        if not placed:
            moved = abs(int(snap.link_x) - start_xy[0]) > 8 or abs(
                int(snap.link_y) - start_xy[1]
            ) > 8
            if snap.mode == CELLAR_MODE and not snap.transitioning and (moved or i > 90):
                placed = True
            else:
                obs = _step(env, nes_action("UP"), assist=None, total=total)
                continue
        frame = cellar_exit_step(snap, side=side)
        obs = _step(env, frame.action, assist=None, total=total)
    return obs, read_snapshot(env.get_ram())


def _walk_target(env: Any, total: list[int], x: int, y: int, frames: int = WALK_MAX_FRAMES):
    obs = None
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or _left_source(snap, snap.screen):
            return obs, snap
        frame = walk_to_step(snap, x, y, y_first=True)
        if frame.reason == "walk_arrived":
            return obs, snap
        obs = _step(env, frame.action, assist=None, total=total)
    return obs, read_snapshot(env.get_ram())


def take_stairs_from_source(
    env: Any,
    source: int,
    *,
    total: list[int],
    cellar_side: str = "left",
    assist: Any = None,
    chase_types: tuple[int, ...] | None = None,
    clear_frames: int | None = None,
    room03_chase_mode: str = "early_clear",
    chase_y_min: int | None = None,
) -> dict[str, Any]:
    """Controller-only: kill-clear / push / walk onto stairs / exit cellar.

    ``chase_types`` filters sword targets (default: all live_combat_objects).
    ``clear_frames`` overrides the pre-push chase budget (0 skips it).
    ``room03_chase_mode``: ``early_clear`` keeps the accepted 0x03 900f chase;
    ``blocking`` swords only Like-Likes that grab or sit on the push tile.
    """
    log: list[str] = []
    obs = None

    def current() -> Any:
        return read_snapshot(env.get_ram())

    def note(label: str) -> None:
        snap = current()
        block = next((o for o in snap.objects if o.type_id == 0x68), None)
        by = f" block=({block.x},{block.y})" if block is not None else ""
        log.append(
            f"{label}: room=0x{snap.screen:02x} mode={snap.mode} "
            f"xy=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x}{by}"
        )

    def saw_stairs(snap: Any) -> bool:
        # 0x03 colliding_tile 0x70-0x73 is sticky around the diamond; mode wins.
        if int(source) == 0x03:
            return stair_transition_modes(snap.mode) or in_patra_cellar(snap)
        if int(source) == 0x30:
            return stair_transition_modes(snap.mode) or in_cellar_67(snap)
        return (
            stair_transition_modes(snap.mode)
            or on_warp_tile(snap)
            or in_patra_cellar(snap)
        )

    # 1. Visible stairs first (generic rooms). 0x03 / 0x30 need the block push.
    if int(source) not in (0x03, 0x30):
        for x, y in STAIR_STANDS[:7]:
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            for _ in range(WALK_MAX_FRAMES):
                snap = current()
                if saw_stairs(snap) or _left_source(snap, source):
                    break
                frame = take_stairs_step(snap, source=source, target=(x, y), push=False)
                obs = _step(env, frame.action, assist=assist, total=total)
                if frame.reason in {"on_stair_tile", "stand_on_stairs"}:
                    obs = _idle(env, 20, assist=assist, total=total)
                    break
            else:
                continue
            break
    note("after_visible_stairs")

    # 2. Kill-clear if the room has combat objects.
    cooldown = 0
    clear_budget = CLEAR_MAX_FRAMES if clear_frames is None else int(clear_frames)
    for _ in range(clear_budget):
        snap = current()
        if saw_stairs(snap) or _left_source(snap, source):
            break
        combat = live_combat_objects(snap)
        if chase_types is not None:
            combat = tuple(obj for obj in combat if obj.type_id in chase_types)
        if in_stair_source(snap, source) and not combat:
            break
        if chase_y_min is not None and int(snap.link_y) < int(chase_y_min):
            frame = walk_to_step(snap, 120, 189, y_first=True)
        else:
            frame, cooldown = chase_sword_step(snap, cooldown, types=chase_types)
        obs = _step(env, frame.action, assist=assist, total=total)
    if clear_budget:
        _idle(env, 12, assist=assist, total=total)
    note("after_clear")

    # 2b. Play room 0x03: push west 0x68 UP, walk the vacated slot, stand (128,141).
    if int(source) == 0x03:
        cooldown = 0
        for i in range(5000):
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            likes = tuple(
                obj for obj in live_combat_objects(snap) if obj.type_id == 0x17
            )
            grabbed = any(
                abs(int(obj.x) - snap.link_x) <= 8
                and abs(int(obj.y) - snap.link_y) <= 8
                for obj in likes
            )
            combat = live_combat_objects(snap)
            if chase_types is not None:
                combat = tuple(obj for obj in combat if obj.type_id in chase_types)
            if room03_chase_mode == "grabbed":
                should_chase = grabbed
                chase_filter = (0x17,)
            elif room03_chase_mode == "blocking":
                should_chase = (
                    room03_like_like_blocks_push(snap)
                    and not room03_west_block_pushed(snap)
                )
                chase_filter = (0x17,)
            else:
                should_chase = (
                    (grabbed or (combat and i < 900))
                    and not room03_west_block_pushed(snap)
                )
                chase_filter = chase_types
            if should_chase:
                if chase_y_min is not None and int(snap.link_y) < int(chase_y_min):
                    frame = walk_to_step(snap, 120, 189, y_first=True)
                else:
                    frame, cooldown = chase_sword_step(
                        snap, cooldown, types=chase_filter
                    )
            else:
                frame = room03_stairs_step(snap)
            obs = _step(env, frame.action, assist=assist, total=total)
            if getattr(frame, "reason", "") == "stand_on_03_stairs":
                obs = _idle(env, 8, assist=assist, total=total)
                if saw_stairs(current()):
                    break
        note("after_room03_stairs")

    if int(source) == 0x30:
        for i in range(4000):
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            frame = room30_stairs_step(snap)
            obs = _step(env, frame.action, assist=assist, total=total)
            if getattr(frame, "reason", "") == "stand_on_30_stairs":
                obs = _idle(env, 12, assist=assist, total=total)
                if saw_stairs(current()):
                    break
        note("after_room30_stairs")

    # 2. Push typical left-blocks, then stand on known stair tiles.
    # 0x03 / 0x30 have their own push; leftover LEFT/grid nudges the 0x68.
    skip_generic = int(source) in (0x03, 0x30)
    for x, y in BLOCK_PUSH_STANDS:
        if skip_generic:
            break
        snap = current()
        if saw_stairs(snap) or _left_source(snap, source):
            break
        obs, snap = _walk_target(env, total, x, y)
        if saw_stairs(snap) or _left_source(snap, source):
            break
        for _ in range(PUSH_FRAMES):
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            obs = _step(env, nes_action("LEFT"), assist=None, total=total)
        else:
            continue
        break
    note("after_block_push")

    for x, y in STAIR_STANDS:
        if skip_generic:
            break
        snap = current()
        if saw_stairs(snap) or _left_source(snap, source):
            break
        for _ in range(WALK_MAX_FRAMES):
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            frame = take_stairs_step(snap, source=source, target=(x, y), push=False)
            obs = _step(env, frame.action, assist=None, total=total)
            if frame.reason in {"on_stair_tile", "stand_on_stairs"}:
                obs = _idle(env, 20, assist=None, total=total)
                break
        else:
            continue
        break
    note("after_stair_stands")

    # 3. Coarse grid walk if stairs still hidden.
    snap = current()
    if (not skip_generic) and in_stair_source(snap, source) and not saw_stairs(snap):
        for y in (0x60, 0x80, 0x90, 0xA0, 0xB0):
            for x in (0x40, 0x60, 0x78, 0x90, 0xB0, 0xD0):
                snap = current()
                if saw_stairs(snap) or _left_source(snap, source):
                    break
                obs, snap = _walk_target(env, total, x, y, frames=220)
                if saw_stairs(snap) or _left_source(snap, source):
                    break
            else:
                continue
            break
        note("after_grid")

    # 4. If we entered a passage / cellar, walk out the requested mouth.
    snap = current()
    passage_entered = bool(stair_transition_modes(snap.mode) or snap.mode == CELLAR_MODE)
    if passage_entered or snap.mode in (CELLAR_MODE, ITEM_CELLAR_MODE, 10, 16):
        obs, snap = _exit_cellar(env, total=total, side=cellar_side)
        note(f"after_cellar_{cellar_side}")

    # Finish mode-10/16 scroll into play. If still in the cellar, keep
    # the requested mouth (blind UP from the pit climbs the wrong side).
    for _ in range(400):
        snap = current()
        if snap.mode == PLAY_MODE and not snap.transitioning:
            break
        if snap.mode == CELLAR_MODE and not snap.transitioning:
            frame = cellar_exit_step(snap, side=cellar_side)
            obs = _step(env, frame.action, assist=assist, total=total)
        else:
            obs = _step(env, nes_action("UP"), assist=assist, total=total)
    if snap.mode == PLAY_MODE:
        obs = _idle(env, 24, assist=None, total=total)
        snap = current()

    result = dest_report(snap)
    result["source"] = int(source)
    result["paired_hypothesis"] = paired_stair_dest(source)
    result["passage_entered"] = passage_entered or stair_transition_modes(current().mode)
    result["log"] = log
    result["ok_left_source"] = bool(_left_source(snap, source) or snap.screen != source)
    return result


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



def _cellar_write_rows(source: int, side: str) -> list[dict[str, Any]]:
    mouth_x, mouth_y = cellar_mouth_xy(side=side)
    return [
        {
            "name": "init_mode9_cellar",
            "address": ADDR_MODE,
            "address_hex": "0x0012",
            "value": 9,
            "note": "lets the engine run InitMode9 (fade + LayoutCellar)",
        },
        {
            "name": "init_submode",
            "address": ADDR_SUBMODE,
            "address_hex": "0x0013",
            "value": 0,
        },
        {
            "name": "init_is_updating_mode",
            "address": ADDR_UPDATING,
            "address_hex": "0x0011",
            "value": 0,
        },
        {
            "name": "underground_exit_type",
            "address": ADDR_UW_EXIT_TYPE,
            "address_hex": "0x005A",
            "value": 0,
        },
        {
            "name": "cellar_mouth_stand",
            "addresses": [ADDR_LINK_X, ADDR_LINK_Y],
            "address_hex": ["0x0070", "0x0084"],
            "values": [mouth_x, mouth_y],
            "side": side,
            "checksubroom_dest": cellar_dest_for(source, side=side),
        },
    ]


def enter_patra_via_source_cellar(
    env: Any,
    source: int,
    *,
    total: list[int],
    side: str = "left",
) -> dict[str, Any]:
    """Fixture-build: InitMode9, stand on a mouth, let CheckSubroom pick dest.

    Does not write NEXT_SCREEN / SCREEN to 0x52.  Live dest is AttrsA/B.
    """
    _assign(env, ADDR_MODE, CELLAR_MODE)
    _assign(env, ADDR_SUBMODE, 0)
    _assign(env, ADDR_UPDATING, 0)
    _assign(env, ADDR_UW_EXIT_TYPE, 0)
    for i in range(400):
        _step(env, nes_idle_action(), assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        if snap.mode == CELLAR_MODE and int(env.get_ram()[ADDR_UPDATING]) != 0 and i > 20:
            break
    mouth_x, mouth_y = cellar_mouth_xy(side=side)
    _assign(env, ADDR_LINK_X, mouth_x)
    _assign(env, ADDR_LINK_Y, mouth_y)
    _idle(env, 4, assist=None, total=total)
    expected = cellar_dest_for(source, side=side)
    for _ in range(80):
        snap = read_snapshot(env.get_ram())
        if expected is not None and snap.screen == expected:
            break
        _step(env, nes_action("UP"), assist=None, total=total)
    for _ in range(400):
        snap = read_snapshot(env.get_ram())
        if (
            snap.mode == PLAY_MODE
            and expected is not None
            and snap.screen == expected
            and not snap.transitioning
        ):
            _idle(env, 24, assist=None, total=total)
            break
        _step(env, nes_idle_action(), assist=None, total=total)
    snap = read_snapshot(env.get_ram())
    result = dest_report(snap)
    result["source"] = int(source)
    result["cellar_side"] = side
    result["expected_dest"] = expected
    result["passage_entered"] = True
    return result



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


def build_winning_fixture(
    *,
    source: int,
    cellar_side: str = "left",
    tag: str = TAG,
    fixture_name: str | None = None,
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    name = fixture_name or f"Level9Stair{source:02X}PatraEnteredReconFixture"
    loader = stair_loader_for(source)
    writes = _loader_write_rows(loader)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "source_state": FIXTURE_SOURCE,
        "stair_source": source,
        "cellar_side": cellar_side,
        "checkpoint": name,
        "fixture_writes": writes,
    }
    try:
        obs, used, loaded = materialize_stair_room(env, source, total=total)
        report["loader"] = used.label
        if not loaded:
            report["error"] = f"loader did not settle 0x{source:02X}"
            return report
        writes.extend(_cellar_write_rows(source, cellar_side))
        report["fixture_writes"] = writes
        if cellar_dest_for(source, side=cellar_side) == 0x52 or is_patra_cellar_source(source):
            dest = enter_patra_via_source_cellar(
                env, source, total=total, side=cellar_side
            )
        else:
            dest = take_stairs_from_source(
                env, source, total=total, cellar_side=cellar_side
            )
        snap = read_snapshot(env.get_ram())
        report["dest"] = dest
        png = RECORDINGS_DIR / f"{tag}_entered.png"
        save_rgb_png(obs if obs is not None else env.render(), png)
        idle_obs = _idle(env, 1, assist=None, total=total)
        save_rgb_png(idle_obs, png)
        report["screenshot"] = str(png)
        report["ok"] = bool(landed_final_patra(snap))
        if report["ok"]:
            path = _save_checkpoint(
                env,
                name,
                source_state=FIXTURE_SOURCE,
                phase=f"cellar_checksubroom_0x{source:02x}_{cellar_side}_into_live_patra",
                result={
                    "ok": True,
                    "stair_source": source,
                    "room": 0x52,
                    "final_patra_live": True,
                    "patra_eye_count": len(patra_eyes(snap)),
                    "frames": total[0],
                },
                fixture_writes=writes,
                bead=BEAD,
            )
            report["checkpoint_path"] = str(path)
        else:
            report["error"] = (
                f"stairs from 0x{source:02X} settled "
                f"room 0x{snap.screen:02X} mode {snap.mode} "
                f"patra={final_patra_live(snap)} eyes={len(patra_eyes(snap))}"
            )
        return report
    finally:
        env.close()


def run_suffix_from_fixture(
    *,
    start_state: str,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = TAG,
    trial_i: int = 0,
) -> dict[str, Any]:
    """Reuse the proven Patra → Ganon → Zelda → credits suffix."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "start_state": start_state,
        "trial": trial_i,
        "tag": tag,
        "runtime_controller_writes": {
            "object": 0,
            "room": 0,
            "door": 0,
            "inventory": 0,
            "progression": 0,
            "capacity": 0,
        },
        "checkpoints": [],
    }

    def checkpoint(name: str, phase: str) -> None:
        if not save_checkpoints:
            return
        path = _save_checkpoint(
            env,
            name,
            source_state=start_state,
            phase=phase,
            result=_checkpoint_result(env, total),
            fixture_writes=[],
            bead=BEAD,
        )
        report["checkpoints"].append(str(path))

    try:
        obs, _ = reset_obs(env)
        start = read_snapshot(env.get_ram())
        report["start"] = compact_snapshot(start)
        report["start_inventory"] = _inventory_snapshot(env.get_ram())
        if not landed_final_patra(start):
            report["error"] = (
                "expected live final Patra with eight eyes and closed north door"
            )
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_start.png")

        patra_fight = FinalPatraFightController().run(
            env, assist=assist, total=total
        )
        report["patra_fight"] = patra_fight
        if not patra_fight["ok"] or not final_patra_north_door_earned(
            read_snapshot(env.get_ram())
        ):
            report["error"] = "final Patra controller timed out before north door"
            return report
        obs = _idle(env, 45, assist=assist, total=total)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_cleared.png")
        checkpoint(
            "Level9StairPatraClearedReconFixture",
            "final_patra_north_door_earned",
        )

        obs, entered = _enter_ganon(env, assist=assist, total=total)
        report["ganon_entered"] = entered
        if not entered:
            report["error"] = "failed to enter live Ganon room 0x42"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_start.png")

        ganon_fight = GanonFightController().run(env, assist=assist, total=total)
        report["ganon_fight"] = ganon_fight
        report["runtime_controller_writes"]["inventory"] = int(
            ganon_fight["selected_item_writes"]
        )
        if (
            not ganon_fight["ok"]
            or not ganon_defeated(env.get_ram())
            or ganon_fight["selected_item_writes"] != 0
        ):
            report["error"] = "Ganon suffix failed or wrote B-item selection"
            return report
        obs = _idle(env, 1, assist=assist, total=total)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_arrow_kill.png")

        obs, power = _collect_power_triforce(env, assist=assist, total=total)
        report["power_triforce_collected"] = power
        if not power:
            report["error"] = "Ganon died but north door did not open"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_defeated.png")

        obs, zelda_room = _enter_zelda(env, assist=assist, total=total)
        report["zelda_room_entered"] = zelda_room
        if not zelda_room:
            report["error"] = "failed to enter live Zelda room 0x32"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_zelda_room.png")

        obs, rescued = _rescue_zelda(env, assist=assist, total=total)
        report["zelda_rescued"] = rescued
        if not rescued:
            report["error"] = "failed to clear guard fires and trigger Zelda ending"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ending_start.png")

        credits_frame = None
        credits_capture_frame = None
        final_frame = None
        for _ in range(12000):
            snap = read_snapshot(env.get_ram())
            if credits_frame is None and credits_rolling(snap):
                credits_frame = total[0]
                obs = _idle(env, 240, assist=assist, total=total)
                credits_capture_frame = total[0]
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_credits.png")
            if final_ending_screen(snap):
                final_frame = total[0]
                obs = _idle(env, 90, assist=assist, total=total)
                save_rgb_png(
                    obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_final_screen.png"
                )
                break
            obs = _step(env, nes_idle_action(), assist=assist, total=total)

        assist_report = assist.report() if assist is not None else {"enabled": False}
        report["credits_frame"] = credits_frame
        report["credits_capture_frame"] = credits_capture_frame
        report["final_screen_frame"] = final_frame
        report["credits_reached"] = credits_frame is not None
        report["final_screen_reached"] = final_frame is not None
        report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
        report["total_frames"] = total[0]
        report["assist"] = assist_report
        report["continuous_session"] = True
        report["state_loads_after_start"] = 0
        report["ok"] = bool(
            credits_frame is not None
            and final_frame is not None
            and assist_report.get("progression_writes", 0) == 0
            and assist_report.get("capacity_writes", 0) == 0
            and not any(report["runtime_controller_writes"].values())
        )
        return report
    finally:
        env.close()



def dump_room_tiles(env: Any, *, total: list[int]) -> dict[str, Any]:
    """Fixture-only colliding-tile scan. Pokes Link XY; not route-eligible."""
    from collections import Counter

    hits: list[dict[str, int]] = []
    mouths: list[dict[str, int]] = []
    counts: Counter[int] = Counter()
    grid: list[list[int]] = []
    for y in range(0x4D, 0xDE, 8):
        row: list[int] = []
        for x in range(0x20, 0xE1, 8):
            _assign(env, ADDR_LINK_X, x)
            _assign(env, ADDR_LINK_Y, y)
            _step(env, nes_idle_action(), assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            tile = int(snap.colliding_tile)
            counts[tile] += 1
            row.append(tile)
            rec = {"x": int(snap.link_x), "y": int(snap.link_y), "tile": tile}
            if STAIR_TILE_LO <= tile <= STAIR_TILE_HI:
                hits.append(rec)
            if tile == BLACK_MOUTH_TILE:
                mouths.append(rec)
        grid.append(row)
    return {
        "stair_hits": hits,
        "mouth_hits": mouths,
        "tile_counts": {f"0x{t:02X}": n for t, n in counts.most_common(12)},
        "grid_origin": [0x20, 0x4D],
        "grid_step": 8,
        "grid": grid,
    }



NORTH_PROBE_FRAMES = 400
CLEAR_PROBE_FRAMES = 1800


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


def _run_bomb_west_04(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    """Controller-only 0x04 west bomb via BombWallController. No door poke."""
    obs = None
    for _ in range(WALK_MAX_FRAMES + 400):
        snap = read_snapshot(env.get_ram())
        if not in_room_04(snap):
            break
        frame = room04_bomb_west_approach_step(snap)
        if (
            abs(int(snap.link_x) - ROOM04_BOMB_WEST_STAND[0]) <= 4
            and abs(int(snap.link_y) - ROOM04_BOMB_WEST_STAND[1]) <= 4
        ):
            break
        obs = _step(env, frame.action, assist=assist, total=total)
    ctrl = make_room04_bomb_west_controller()
    after_bomb_obs = None
    for _ in range(ctrl.max_frames):
        snap = read_snapshot(env.get_ram())
        frame = ctrl.step(snap)
        obs = _step(env, frame.action, assist=assist, total=total)
        if after_bomb_obs is None and ctrl.phase.name == "PUSH":
            after_bomb_obs = obs
        if ctrl.success or ctrl.phase.name in {"DONE", "FAILED"}:
            break
    snap = read_snapshot(env.get_ram())
    return {
        "controller": ctrl.report(),
        "dest": dest_report(snap),
        "entered_0x03": int(snap.screen) == ROOM03,
        "still_in_04": bool(in_room_04(snap)),
        "obs": obs,
        "after_bomb_obs": after_bomb_obs,
    }


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


def run_room04_bomb_west_to_credits(
    *,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = "l9_play04_bombwest_patra_credits_recon",
    trial_i: int = 0,
) -> dict[str, Any]:
    """One continuous trial: 0x04 bomb-west → 0x03 stairs → Patra → credits.

    Fixture inventory + 0x14 neighbor-scroll happen before the walk.
    After materialize, no object / room / door / progression / capacity writes.
    0x03 doors are never poked. InitMode9 is not used.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    loader = stair_loader_for(ROOM04)
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
        "source_room": "0x04",
        "via": "bomb_west",
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
        "door_poke_on_03": False,
    }
    try:
        obs, used, loaded = materialize_stair_room(
            env, ROOM04, total=total, selected_item=B_ITEM_BOMBS
        )
        report["loader"] = used.label
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x04"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        report["settled"] = dest_report(read_snapshot(env.get_ram()))
        settle_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

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
        after_bomb_obs = bomb.pop("after_bomb_obs")
        dest_obs = bomb.pop("obs")
        report["bomb_west"] = bomb
        if after_bomb_obs is not None:
            save_rgb_png(after_bomb_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_after_bomb.png")
        if dest_obs is not None:
            save_rgb_png(dest_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")
        snap = read_snapshot(env.get_ram())
        report["dest_screen"] = int(snap.screen)
        if int(snap.screen) != ROOM03:
            report["error"] = (
                f"bomb-west dest 0x{snap.screen:02X} is not 0x03 "
                f"phase={bomb['controller'].get('phase')}"
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

        # East hole sits on stair-Y but the diamond blocks a straight LEFT.
        # Accepted 0x03 push recovers down the west wall; seed that stand
        # (32,189) before the same take_stairs_from_source as play-source 03.
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

        dest = walk_play_room_to_patra(
            env,
            ROOM03,
            total=total,
            cellar_side="left",
            assist=assist,
        )
        report["walk"] = dest
        snap = read_snapshot(env.get_ram())
        idle_obs = _idle(env, 1, assist=None, total=total)
        save_rgb_png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_after_walk.png")
        if not landed_final_patra(snap):
            report["error"] = (
                f"0x04→0x03 stairs settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={dest.get('final_patra_live')}"
            )
            report["final"] = dest
            return report
        save_rgb_png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_entry.png")
        if save_checkpoints:
            path = _save_checkpoint(
                env,
                "Level9Room04BombWestReconFixture",
                source_state=FIXTURE_SOURCE,
                phase="play_0x04_bomb_west_into_live_patra",
                result={
                    "ok": True,
                    "source_room": ROOM04,
                    "via": "bomb_west",
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
            start_state="play_0x04_bomb_west_walk",
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



def _run_bomb_west_31(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    """Controller-only 0x31 west bomb via BombWallController. No door poke."""
    obs = None
    for _ in range(WALK_MAX_FRAMES + 400):
        snap = read_snapshot(env.get_ram())
        if not in_room_31(snap):
            break
        frame = room31_bomb_west_approach_step(snap)
        if (
            abs(int(snap.link_x) - ROOM31_BOMB_WEST_STAND[0]) <= 4
            and abs(int(snap.link_y) - ROOM31_BOMB_WEST_STAND[1]) <= 4
        ):
            break
        obs = _step(env, frame.action, assist=assist, total=total)
    ctrl = make_room31_bomb_west_controller()
    after_bomb_obs = None
    for _ in range(ctrl.max_frames):
        snap = read_snapshot(env.get_ram())
        frame = ctrl.step(snap)
        obs = _step(env, frame.action, assist=assist, total=total)
        if after_bomb_obs is None and ctrl.phase.name == "PUSH":
            after_bomb_obs = obs
        if ctrl.success or ctrl.phase.name in {"DONE", "FAILED"}:
            break
    snap = read_snapshot(env.get_ram())
    return {
        "controller": ctrl.report(),
        "dest": dest_report(snap),
        "entered_0x30": int(snap.screen) == ROOM30,
        "still_in_31": bool(in_room_31(snap)),
        "obs": obs,
        "after_bomb_obs": after_bomb_obs,
    }


def dump_room_31(*, tag: str = "l9_room31_dump") -> dict[str, Any]:
    """Live 0x31 dump + bomb-west probe. Stages 0x41, never 0x30 doors."""
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
        "room": "0x31",
        "rom_doors": {
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
            "0x41": {"north": 0, "north_name": "open"},
        },
        "rom_neighbors_of_30": neighbors,
        "rom_west_is_bomb": room31_rom_west_is_bomb(),
        "rom_predecessor": room31_is_rom_predecessor_of_30(),
        "loader_avoids_30": room31_loader_avoids_30(),
        "bomb_stand": list(ROOM31_BOMB_WEST_STAND),
        "selected_item_fixture": B_ITEM_BOMBS,
        "door_poke_on_30": False,
    }

    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        total = [0]
        obs, loader, loaded = materialize_stair_room(
            env, ROOM31, total=total, door_staging=True, selected_item=B_ITEM_BOMBS
        )
        report["loader"] = {
            "label": loader.label,
            "from_room": f"0x{loader.from_room:02X}",
            "direction": loader.direction,
            "link_xy": [loader.link_x, loader.link_y],
            "door_staging_on_from_room": True,
            "from_room_is_30": loader.from_room == ROOM30,
            "from_room_is_41": loader.from_room == ROOM41,
            "note": "0x0F/0x0F is written on 0x41 so the south-shutter scroll can start; 0x30 doors are not poked.",
        }
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x31"
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
            env, ROOM31, total=total, door_staging=True, selected_item=B_ITEM_BOMBS
        )
        if not loaded:
            report["error"] = "rematerialize failed 0x31"
            return report
        cooldown = 0
        for _ in range(CLEAR_MAX_FRAMES):
            snap = read_snapshot(env.get_ram())
            if in_room_31(snap) and not live_combat_objects(snap):
                break
            if not in_room_31(snap):
                break
            frame, cooldown = chase_sword_step(snap, cooldown)
            obs = _step(env, frame.action, assist=assist, total=total)
        obs = _idle(env, 16, assist=assist, total=total)
        report["after_clear"] = dest_report(read_snapshot(env.get_ram()))

        bomb = _run_bomb_west_31(env, total=total, assist=assist)
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
        report["dest_mode"] = int(dest_snap.mode)
        report["lands_0x30"] = int(dest_snap.screen) == ROOM30 and dest_snap.mode == PLAY_MODE
        report["dest_objects"] = dest_report(dest_snap)["objects"]
        if report["lands_0x30"]:
            has_block = any(
                int(obj.get("type_id") or 0) == 0x68
                for obj in (report["dest_objects"] or [])
            )
            report["block_0x68_present"] = has_block
            report["rom_secret_still_block_stairs"] = room30_rom_secret_is_block_stairs()
            dest = take_stairs_from_source(
                env, ROOM30, total=total, cellar_side="right", assist=assist
            )
            report["stairs_probe"] = dest
            snap = read_snapshot(env.get_ram())
            for _ in range(180):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE and int(snap.screen) == ROOM04 and not snap.transitioning:
                    break
                _step(env, nes_action("UP"), assist=assist, total=total)
            snap = read_snapshot(env.get_ram())
            report["stairs_still_work"] = (
                has_block
                and room30_rom_secret_is_block_stairs()
                and (
                    int(snap.screen) == ROOM04
                    or int(snap.screen) == 0x67
                    or bool(dest.get("passage_entered"))
                )
            )
            report["stairs_dest_screen"] = int(snap.screen)
            report["stairs_dest_mode"] = int(snap.mode)
            stairs_png = RECORDINGS_DIR / f"{tag}_stairs_dest.png"
            idle = _idle(env, 1, assist=assist, total=total)
            save_rgb_png(idle, stairs_png)
            report["stairs_dest_png"] = str(stairs_png)

        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, loader_np, loaded_np = materialize_stair_room(
            env, ROOM31, total=total, door_staging=False, selected_item=B_ITEM_BOMBS
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
        if not report["lands_0x30"]:
            report["next_candidate"] = (
                "0x31 west did not land 0x30. ROM neighbors of 0x30 still "
                "untested as clean walks: north 0x20 wall (ROM-disproved), "
                "west 0x2F shutter/wall, south 0x40 key (door-staged, not clean). "
                "Do not treat 0x40 as clean."
            )
        return report
    finally:
        env.close()


def run_room31_bomb_west_to_credits(
    *,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = "l9_play31_bombwest_patra_credits_recon",
    trial_i: int = 0,
    from_41: bool = False,
) -> dict[str, Any]:
    """One continuous trial: [0x41 north →] 0x31 bomb-west → 0x30 stairs suffix.

    Fixture inventory + neighbor-scroll happen before the walk.
    After materialize, no object / room / door / progression / capacity writes.
    0x31/0x30 doors are never poked. InitMode9 is not used.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    start_room = ROOM41 if from_41 else ROOM31
    loader = stair_loader_for(start_room)
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
        "source_room": "0x41" if from_41 else "0x31",
        "via": "north_then_bomb_west" if from_41 else "bomb_west",
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
        "door_poke_on_31": False,
    }
    try:
        obs, used, loaded = materialize_stair_room(
            env, start_room, total=total, selected_item=B_ITEM_BOMBS
        )
        report["loader"] = used.label
        report["loaded"] = loaded
        if not loaded:
            report["error"] = f"loader did not settle 0x{start_room:02X}"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        report["settled"] = dest_report(read_snapshot(env.get_ram()))
        settle_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

        if from_41:
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
            north = _probe_room41_north(env, total=total, assist=assist)
            report["north"] = north
            snap = read_snapshot(env.get_ram())
            idle = _idle(env, 1, assist=assist, total=total)
            save_rgb_png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")
            report["dest_screen"] = int(snap.screen)
            if int(snap.screen) != ROOM31:
                report["error"] = (
                    f"0x41 north dest 0x{snap.screen:02X} is not 0x31 "
                    f"end_mode={north.get('end_mode')}"
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
        after_bomb_obs = bomb.pop("after_bomb_obs")
        dest_obs = bomb.pop("obs")
        report["bomb_west"] = bomb
        if after_bomb_obs is not None:
            save_rgb_png(
                after_bomb_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_after_bomb.png"
            )
        if dest_obs is not None:
            save_rgb_png(dest_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")
        snap = read_snapshot(env.get_ram())
        report["dest_screen"] = int(snap.screen)
        if int(snap.screen) != ROOM30:
            report["error"] = (
                f"bomb-west dest 0x{snap.screen:02X} is not 0x30 "
                f"phase={(bomb.get('controller') or {}).get('phase')}"
            )
            return report

        dest = take_stairs_from_source(
            env,
            ROOM30,
            total=total,
            cellar_side="right",
            assist=assist,
            chase_y_min=180 if from_41 else None,
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
                f"0x31->0x30 stairs dest 0x{snap.screen:02X} mode {snap.mode} "
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
                f"0x31->0x30->0x04->0x03 stairs settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={dest03.get('final_patra_live')}"
            )
            return report
        if save_checkpoints:
            path = _save_checkpoint(
                env,
                (
                    "Level9Room41NorthReconFixture"
                    if from_41
                    else "Level9Room31BombWestReconFixture"
                ),
                source_state=FIXTURE_SOURCE,
                phase=(
                    "play_0x41_north_into_live_patra"
                    if from_41
                    else "play_0x31_bomb_west_into_live_patra"
                ),
                result={
                    "ok": True,
                    "source_room": ROOM41 if from_41 else ROOM31,
                    "via": "north_then_bomb_west" if from_41 else "bomb_west",
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
            start_state=(
                "play_0x41_north_walk" if from_41 else "play_0x31_bomb_west_walk"
            ),
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


def _probe_room41_north(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    """Controller-only north open-door push. Does not write 0x0F/0x0F on 0x41 or 0x31."""
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
                "from_room": ROOM41,
                "direction": "UP",
                "to_room": int(snap.screen),
                "mode": int(snap.mode),
                "objects": dest_report(snap)["objects"],
            }
            break
        if (
            snap.screen != ROOM41
            or snap.mode != PLAY_MODE
            or snap.transitioning
        ):
            _step(env, nes_action("UP"), assist=assist, total=total)
            continue
        frame = room41_to_31_step(snap)
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
        "still_in_41": bool(in_room_41(end)),
        "stuck_y": int(end.link_y),
    }


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


def walk_play_room_to_patra(
    env: Any,
    source: int,
    *,
    total: list[int],
    cellar_side: str = "left",
    assist: Any = None,
    chase_types: tuple[int, ...] | None = None,
    clear_frames: int | None = None,
    room03_chase_mode: str = "early_clear",
    chase_y_min: int | None = None,
) -> dict[str, Any]:
    """Controller-only: play room stairs -> cellar 0x77 -> left mouth -> 0x52.

    No InitMode9. Dest comes from CheckWarps / CheckSubroom.
    """
    dest = take_stairs_from_source(
        env,
        source,
        total=total,
        cellar_side=cellar_side,
        assist=assist,
        chase_types=chase_types,
        clear_frames=clear_frames,
        room03_chase_mode=room03_chase_mode,
        chase_y_min=chase_y_min,
    )
    snap = read_snapshot(env.get_ram())
    dest["entered_cellar_77"] = bool(in_patra_cellar(snap) or dest.get("passage_entered"))
    dest["cellar_for_source"] = (
        None
        if cellar_for_play_room(source) is None
        else {
            "cellar": f"0x{cellar_for_play_room(source)[0]:02X}",
            "mouth": cellar_for_play_room(source)[1],
        }
    )
    return dest


def run_play_source_to_credits(
    *,
    source: int,
    cellar_side: str = "left",
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = TAG,
    trial_i: int = 0,
) -> dict[str, Any]:
    """One continuous trial: play source -> cellar 0x77 -> live Patra -> credits.

    Fixture inventory + neighbor-scroll settle happen before the walk.
    After materialize, no object / room / door / progression / capacity writes.
    InitMode9 is not used.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    loader = stair_loader_for(source)
    writes = _loader_write_rows(loader)
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "source_room": f"0x{source:02X}",
        "cellar_side": cellar_side,
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
    }
    try:
        obs, used, loaded = materialize_stair_room(env, source, total=total)
        report["loader"] = used.label
        report["loaded"] = loaded
        if not loaded:
            report["error"] = f"loader did not settle 0x{source:02X}"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        settle = dest_report(read_snapshot(env.get_ram()))
        report["settled"] = settle
        settle_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

        tiles = dump_room_tiles(env, total=total)
        tile_json = RECORDINGS_DIR / f"{tag}_t{trial_i}_0x{source:02x}_tiles.json"
        write_json_report(tile_json, tiles)
        report["tile_json"] = str(tile_json)
        report["stair_hits"] = tiles["stair_hits"]
        report["mouth_hits"] = tiles["mouth_hits"]
        if tiles["stair_hits"]:
            hit = tiles["stair_hits"][0]
            report["stair_tile"] = hit["tile"]
            report["stair_xy"] = [hit["x"], hit["y"]]
        elif tiles["mouth_hits"]:
            hit = tiles["mouth_hits"][0]
            report["stair_tile"] = hit["tile"]
            report["stair_xy"] = [hit["x"], hit["y"]]

        # Rematerialize so the tile-scan pokes are not on the acceptance path.
        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, used, loaded = materialize_stair_room(env, source, total=total)
        if not loaded:
            report["error"] = f"rematerialize failed 0x{source:02X}"
            return report
        save_rgb_png(obs, settle_png)

        dest = walk_play_room_to_patra(
            env, source, total=total, cellar_side=cellar_side, assist=assist
        )
        report["walk"] = dest
        snap = read_snapshot(env.get_ram())
        walk_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_after_walk.png"
        idle_obs = _idle(env, 1, assist=None, total=total)
        save_rgb_png(idle_obs, walk_png)
        report["after_walk_png"] = str(walk_png)
        if in_patra_cellar(snap) or snap.mode == CELLAR_MODE:
            cellar_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_cellar.png"
            save_rgb_png(idle_obs, cellar_png)
            report["cellar_png"] = str(cellar_png)
            report["entered_cellar_77"] = snap.screen == PATRA_STAIR_SOURCE
        if landed_final_patra(snap):
            patra_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_entry.png"
            save_rgb_png(idle_obs, patra_png)
            report["patra_entry_png"] = str(patra_png)
            report["final_patra_live"] = True
        else:
            report["error"] = (
                f"walk from 0x{source:02X} settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={dest.get('final_patra_live')} "
                f"eyes={dest.get('patra_eyes')}"
            )
            report["final"] = dest
            return report

        if save_checkpoints:
            path = _save_checkpoint(
                env,
                f"Level9Room{source:02X}StairsReconFixture",
                source_state=FIXTURE_SOURCE,
                phase=f"play_0x{source:02x}_stairs_into_live_patra",
                result={
                    "ok": True,
                    "source_room": source,
                    "room": 0x52,
                    "final_patra_live": True,
                    "patra_eye_count": len(patra_eyes(snap)),
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
            start_state=f"play_0x{source:02x}_walk",
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


def run_suffix_from_live_env(
    env: Any,
    *,
    assist: Any,
    total: list[int],
    tag: str,
    trial_i: int,
    start_state: str,
) -> dict[str, Any]:
    """Patra -> Ganon -> Zelda -> credits on an already-live env."""
    report: dict[str, Any] = {
        "ok": False,
        "start_state": start_state,
        "trial": trial_i,
        "runtime_controller_writes": {
            "object": 0,
            "room": 0,
            "door": 0,
            "inventory": 0,
            "progression": 0,
            "capacity": 0,
        },
    }
    start = read_snapshot(env.get_ram())
    report["start"] = compact_snapshot(start)
    if not landed_final_patra(start):
        report["error"] = "expected live final Patra with eight eyes and closed north door"
        return report
    save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_start.png")

    patra_fight = FinalPatraFightController().run(env, assist=assist, total=total)
    report["patra_fight"] = patra_fight
    if not patra_fight["ok"] or not final_patra_north_door_earned(
        read_snapshot(env.get_ram())
    ):
        report["error"] = "final Patra controller timed out before north door"
        return report
    obs = _idle(env, 45, assist=assist, total=total)
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_cleared.png")

    obs, entered = _enter_ganon(env, assist=assist, total=total)
    report["ganon_entered"] = entered
    if not entered:
        report["error"] = "failed to enter live Ganon room 0x42"
        return report
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_start.png")

    ganon_fight = GanonFightController().run(env, assist=assist, total=total)
    report["ganon_fight"] = ganon_fight
    report["runtime_controller_writes"]["inventory"] = int(
        ganon_fight["selected_item_writes"]
    )
    if (
        not ganon_fight["ok"]
        or not ganon_defeated(env.get_ram())
        or ganon_fight["selected_item_writes"] != 0
    ):
        report["error"] = "Ganon suffix failed or wrote B-item selection"
        return report
    obs = _idle(env, 1, assist=assist, total=total)
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_arrow_kill.png")

    obs, power = _collect_power_triforce(env, assist=assist, total=total)
    report["power_triforce_collected"] = power
    if not power:
        report["error"] = "Ganon died but north door did not open"
        return report
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_defeated.png")

    obs, zelda_room = _enter_zelda(env, assist=assist, total=total)
    report["zelda_room_entered"] = zelda_room
    if not zelda_room:
        report["error"] = "failed to enter live Zelda room 0x32"
        return report
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_zelda_room.png")

    obs, rescued = _rescue_zelda(env, assist=assist, total=total)
    report["zelda_rescued"] = rescued
    if not rescued:
        report["error"] = "failed to clear guard fires and trigger Zelda ending"
        return report
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ending_start.png")

    credits_frame = None
    credits_capture_frame = None
    final_frame = None
    for _ in range(12000):
        snap = read_snapshot(env.get_ram())
        if credits_frame is None and credits_rolling(snap):
            credits_frame = total[0]
            obs = _idle(env, 240, assist=assist, total=total)
            credits_capture_frame = total[0]
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_credits.png")
        if final_ending_screen(snap):
            final_frame = total[0]
            obs = _idle(env, 90, assist=assist, total=total)
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_final_screen.png")
            break
        obs = _step(env, nes_idle_action(), assist=assist, total=total)

    assist_report = assist.report() if assist is not None else {"enabled": False}
    report["credits_frame"] = credits_frame
    report["credits_capture_frame"] = credits_capture_frame
    report["final_screen_frame"] = final_frame
    report["credits_reached"] = credits_frame is not None
    report["final_screen_reached"] = final_frame is not None
    report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
    report["total_frames"] = total[0]
    report["assist"] = assist_report
    report["continuous_session"] = True
    report["state_loads_after_start"] = 0
    report["ok"] = bool(
        credits_frame is not None
        and final_frame is not None
        and assist_report.get("progression_writes", 0) == 0
        and assist_report.get("capacity_writes", 0) == 0
        and not any(report["runtime_controller_writes"].values())
    )
    return report


def _trial_summary(report: dict[str, Any]) -> dict[str, Any]:
    patra = report.get("patra_fight") or {}
    ganon = report.get("ganon_fight") or {}
    return {
        "trial": report.get("trial"),
        "ok": report.get("ok"),
        "patra_north_door": patra.get("north_door_earned"),
        "patra_frames": patra.get("frames"),
        "ganon_defeated": ganon.get("last_boss_defeated"),
        "credits_frame": report.get("credits_frame"),
        "final_screen_frame": report.get("final_screen_frame"),
        "error": report.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--dest-table", action="store_true")
    parser.add_argument("--dump-play", action="store_true")
    parser.add_argument("--dump-13", action="store_true")
    parser.add_argument("--dump-04", action="store_true")
    parser.add_argument("--compose-04", action="store_true")
    parser.add_argument("--dump-30", action="store_true")
    parser.add_argument("--compose-30", action="store_true")
    parser.add_argument("--dump-40", action="store_true")
    parser.add_argument("--compose-40", action="store_true")
    parser.add_argument("--dump-31", action="store_true")
    parser.add_argument("--compose-31", action="store_true")
    parser.add_argument("--dump-21", action="store_true")
    parser.add_argument("--compose-21", action="store_true")
    parser.add_argument("--dump-41", action="store_true")
    parser.add_argument("--compose-41", action="store_true")
    parser.add_argument("--play-source", default="", help="hex play room, e.g. 03")
    parser.add_argument("--build-fixture", action="store_true")
    parser.add_argument("--source", default="", help="hex stair source, e.g. 60")
    parser.add_argument("--cellar-side", default="left", choices=("left", "right"))
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args()

    if args.probe:
        probed = probe_sources(tag=f"{args.tag}_probe")
        out = RECORDINGS_DIR / f"{args.tag}_probe.json"
        write_json_report(out, probed)
        print("PROBE winner", probed.get("winner"))
        print("REPORT", out)
        return 0 if probed.get("ok") else 1

    if args.dest_table:
        table = probe_cellar_dest_table(tag=f"{args.tag}_dest_table")
        out = RECORDINGS_DIR / f"{args.tag}_dest_table.json"
        write_json_report(out, table)
        print("DEST_TABLE winner", table.get("winner"))
        print("REPORT", out)
        return 0 if table.get("ok") else 1

    if args.dump_13:
        dumped = dump_room_13(tag=args.tag if args.tag != TAG else "l9_room13_dump")
        out = RECORDINGS_DIR / f"{args.tag if args.tag != TAG else 'l9_room13_dump'}.json"
        write_json_report(out, dumped)
        print("DUMP_13", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "clean_walk": dumped.get("clean_walk"),
            "how_up_opens": dumped.get("how_up_opens"),
            "north_uncleared": dumped.get("north_probe_uncleared"),
            "north_cleared": dumped.get("north_probe_cleared"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.dump_04:
        dumped = dump_room_04(tag=args.tag if args.tag != TAG else "l9_room04_dump")
        out = RECORDINGS_DIR / f"{args.tag if args.tag != TAG else 'l9_room04_dump'}.json"
        write_json_report(out, dumped)
        print("DUMP_04", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x03": dumped.get("lands_0x03"),
            "dest_screen": dumped.get("dest_screen"),
            "bomb_west": (dumped.get("bomb_west") or {}).get("controller"),
            "stair_tile_at_03": dumped.get("stair_tile_at_03"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_04:
        trials: list[dict[str, Any]] = []
        tag = args.tag if args.tag != TAG else "l9_play04_bombwest_patra_credits_recon"
        for trial_i in range(max(1, args.trials)):
            result = run_room04_bomb_west_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE04_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk") or {}).get("landed_final_patra"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x04_bomb_west_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x04",
            "via": "bomb_west",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = RECORDINGS_DIR / f"{tag}.json"
        write_json_report(out, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_30:
        dumped = dump_room_30(tag=args.tag if args.tag != TAG else "l9_room30_dump")
        out = RECORDINGS_DIR / f"{args.tag if args.tag != TAG else 'l9_room30_dump'}.json"
        write_json_report(out, dumped)
        print("DUMP_30", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "stair_hits": dumped.get("stair_hits"),
            "entered_cellar_67": dumped.get("entered_cellar_67"),
            "dest_screen": dumped.get("dest_screen"),
            "lands_0x04": dumped.get("lands_0x04"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_30:
        trials: list[dict[str, Any]] = []
        tag = args.tag if args.tag != TAG else "l9_play30_cellar67_patra_credits_recon"
        for trial_i in range(max(1, args.trials)):
            result = run_room30_stairs_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE30_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "cellar": result.get("cellar_room"),
                    "patra": (result.get("suffix_from_04") or result.get("walk") or {}).get("landed_final_patra")
                    if False else (result.get("walk_04") or {}).get("landed_final_patra") or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x30_cellar_0x67_right_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x30",
            "via": "cellar_0x67_right",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = RECORDINGS_DIR / f"{tag}.json"
        write_json_report(out, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_40:
        dumped = dump_room_40(tag=args.tag if args.tag != TAG else "l9_room40_dump")
        out = RECORDINGS_DIR / f"{args.tag if args.tag != TAG else 'l9_room40_dump'}.json"
        write_json_report(out, dumped)
        print("DUMP_40", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x30": dumped.get("lands_0x30"),
            "dest_screen": dumped.get("dest_screen"),
            "how_up_opens": dumped.get("how_up_opens"),
            "stair_tile_at_30": dumped.get("stair_tile_at_30"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_40:
        trials: list[dict[str, Any]] = []
        tag = args.tag if args.tag != TAG else "l9_play40_keynorth_patra_credits_recon"
        for trial_i in range(max(1, args.trials)):
            result = run_room40_key_north_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE40_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk_04") or {}).get("landed_final_patra")
                    or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x40_key_north_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x40",
            "via": "key_north",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = RECORDINGS_DIR / f"{tag}.json"
        write_json_report(out, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_31:
        dumped = dump_room_31(tag=args.tag if args.tag != TAG else "l9_room31_dump")
        out = RECORDINGS_DIR / f"{args.tag if args.tag != TAG else 'l9_room31_dump'}.json"
        write_json_report(out, dumped)
        print("DUMP_31", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x30": dumped.get("lands_0x30"),
            "dest_screen": dumped.get("dest_screen"),
            "bomb_west": (dumped.get("bomb_west") or {}).get("controller"),
            "stairs_still_work": dumped.get("stairs_still_work"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_31:
        trials: list[dict[str, Any]] = []
        tag = args.tag if args.tag != TAG else "l9_play31_bombwest_patra_credits_recon"
        for trial_i in range(max(1, args.trials)):
            result = run_room31_bomb_west_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE31_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk_04") or {}).get("landed_final_patra")
                    or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x31_bomb_west_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x31",
            "via": "bomb_west",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = RECORDINGS_DIR / f"{tag}.json"
        write_json_report(out, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_21:
        dumped = dump_room_21(tag=args.tag if args.tag != TAG else "l9_room21_dump")
        out = RECORDINGS_DIR / f"{args.tag if args.tag != TAG else 'l9_room21_dump'}.json"
        write_json_report(out, dumped)
        print("DUMP_21", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x31": dumped.get("lands_0x31"),
            "dest_screen": dumped.get("dest_screen"),
            "how_south_opens": dumped.get("how_south_opens"),
            "west_bomb_still_works": dumped.get("west_bomb_still_works"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_21:
        trials: list[dict[str, Any]] = []
        tag = args.tag if args.tag != TAG else "l9_play21_south_patra_credits_recon"
        for trial_i in range(max(1, args.trials)):
            result = run_room21_south_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE21_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk_04") or {}).get("landed_final_patra")
                    or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x21_south_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x21",
            "via": "south_shutter",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = RECORDINGS_DIR / f"{tag}.json"
        write_json_report(out, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.compose_41:
        trials: list[dict[str, Any]] = []
        tag = args.tag if args.tag != TAG else "l9_play41_north_patra_credits_recon"
        for trial_i in range(max(1, args.trials)):
            result = run_room31_bomb_west_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
                from_41=True,
            )
            trials.append(result)
            print(
                "COMPOSE41_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk_04") or {}).get("landed_final_patra")
                    or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x41_north_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x41",
            "via": "north_then_bomb_west",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = RECORDINGS_DIR / f"{tag}.json"
        write_json_report(out, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_41:
        dumped = dump_room_41(tag=args.tag if args.tag != TAG else "l9_room41_dump")
        out = RECORDINGS_DIR / f"{args.tag if args.tag != TAG else 'l9_room41_dump'}.json"
        write_json_report(out, dumped)
        print("DUMP_41", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x31": dumped.get("lands_0x31"),
            "dest_screen": dumped.get("dest_screen"),
            "how_north_opens": dumped.get("how_north_opens"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.dump_play:
        dumped = dump_play_rooms(tag=f"{args.tag}_play_tiles")
        out = RECORDINGS_DIR / f"{args.tag}_play_tiles.json"
        write_json_report(out, dumped)
        print("DUMP_PLAY rooms", len(dumped.get("rooms") or []))
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.play_source:
        play = int(args.play_source, 16)
        trials: list[dict[str, Any]] = []
        for trial_i in range(max(1, args.trials)):
            result = run_play_source_to_credits(
                source=play,
                cellar_side=args.cellar_side,
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=args.tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "PLAY_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "source": result.get("source_room"),
                    "stair_tile": result.get("stair_tile"),
                    "stair_xy": result.get("stair_xy"),
                    "patra": (result.get("walk") or {}).get("landed_final_patra"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_room_stairs_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": f"0x{play:02X}",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = RECORDINGS_DIR / f"{args.tag}.json"
        write_json_report(out, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    source = int(args.source, 16) if args.source else 0
    fixture_name = (
        f"Level9Stair{source:02X}PatraEnteredReconFixture" if source else ""
    )
    built = None
    if args.build_fixture:
        if not source:
            print("ERROR --source is required with --build-fixture")
            return 2
        built = build_winning_fixture(
            source=source,
            cellar_side=args.cellar_side,
            tag=args.tag,
            fixture_name=fixture_name,
        )
        print("FIXTURE", {"ok": built.get("ok"), "path": built.get("checkpoint_path")})
        if not built.get("ok"):
            write_json_report(RECORDINGS_DIR / f"{args.tag}.json", {"fixture": built, "ok": False})
            return 1

    start_state = args.from_state or fixture_name
    if not start_state:
        print("ERROR provide --from-state or --build-fixture --source")
        return 2

    trials: list[dict[str, Any]] = []
    for trial_i in range(max(1, args.trials)):
        result = run_suffix_from_fixture(
            start_state=start_state,
            infinite_life=args.infinite_life,
            save_checkpoints=args.save_state,
            tag=args.tag,
            trial_i=trial_i,
        )
        trials.append(result)
        print("TRIAL", _trial_summary(result))

    report = {
        "bead": BEAD,
        "segment": "stair_source_to_final_screen",
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "fixture": built,
        "ok": all(trial.get("ok") for trial in trials),
        "trials": trials,
    }
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, report)
    print("REPORT", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
