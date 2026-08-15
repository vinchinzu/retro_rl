"""Level 9 stair-run env-step primitives (loader, take-stairs, cellar)."""

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
