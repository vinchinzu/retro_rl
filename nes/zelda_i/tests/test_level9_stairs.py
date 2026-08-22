
from __future__ import annotations

import numpy as np

from zelda_i.dungeon_trace import action_button_names
from zelda_i.level9_ganon import LEVEL9, ROOM_BEFORE_GANON
from zelda_i.level9_patra import OBJ_PATRA, OBJ_PATRA_EYE, PATRA_EYE_COUNT
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES
from zelda_i.level9_stairs import (
    BLACK_MOUTH_TILE,
    BLOCK_STAIRS_X,
    BLOCK_STAIRS_Y,
    CELLAR_LEFT_X,
    CELLAR_MODE,
    ROOM03_STAIR_X,
    ROOM03_STAIR_Y,
    LEVEL9_CELLAR_ROOMS,
    LEVEL9_STAIR_LIST_DC,
    LEVEL9_STAIR_LIST_INES,
    LEVEL9_STAIR_PAIRS,
    PATRA_STAIR_SOURCE,
    PLAY_STAIR_CANDIDATES,
    STAIRS_ENTER_MODE,
    cellar_dest_for,
    cellar_exit_step,
    cellar_for_play_room,
    cellar_mouth_xy,
    chase_sword_step,
    in_patra_cellar,
    in_room_13,
    is_patra_cellar_source,
    dest_report,
    in_stair_source,
    landed_final_patra,
    on_warp_tile,
    paired_stair_dest,
    play_rooms_entering_cellar,
    room03_invuln_on_push_column,
    room03_like_like_blocks_push,
    room03_stairs_step,
    room03_west_block_pushed,
    room13_is_clean_predecessor_of_03,
    room13_to_03_step,
    ROOM03_ROM_EAST,
    ROOM03_ROM_NORTH,
    ROOM03_ROM_SOUTH,
    ROOM03_ROM_WEST,
    ROOM04,
    ROOM04_BOMB_WEST_APPROACH,
    ROOM04_BOMB_WEST_STAND,
    room04_bomb_west_approach_step,
    ROOM04_ROM_EAST,
    ROOM04_ROM_NORTH,
    ROOM04_ROM_SOUTH,
    ROOM04_ROM_WEST,
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
    ROOM02,
    ROOM02_ROM_EAST,
    ROOM13,
    ROOM13_ROM_EAST,
    ROOM13_ROM_NORTH,
    ROOM13_ROM_SOUTH,
    ROOM13_ROM_WEST,
    in_room_04,
    in_room_30,
    make_room04_bomb_west_controller,
    room30_loader_avoids_04,
    room30_rom_secret_is_block_stairs,
    room30_block_secret_open,
    room30_stairs_step,
    in_cellar_67,
    CELLAR_67,
    ROOM30_PUSH_X,
    ROOM30_PUSH_Y,
    pause_select_next_b_item_script,
    rom_door_name,
    room03_rom_neighbors,
    room04_is_rom_predecessor_of_03,
    room04_rom_west_is_bomb,
    room30_rom_neighbors,
    room40_is_rom_predecessor_of_30,
    room40_loader_avoids_30,
    room40_rom_north_is_key,
    room40_to_30_step,
    in_room_40,
    ROOM20,
    ROOM2F,
    ROOM11,
    ROOM21,
    ROOM21_ROM_EAST,
    ROOM21_ROM_NORTH,
    ROOM21_ROM_SOUTH,
    ROOM21_ROM_WEST,
    ROOM21_SOUTH_Y,
    ROOM21_WEST_X,
    ROOM31,
    ROOM31_BOMB_WEST_APPROACH,
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
    ROOM41_SOUTH_Y,
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
    in_room_21,
    in_room_31,
    in_room_41,
    in_room_51,
    make_room31_bomb_west_controller,
    room21_is_rom_predecessor_of_31,
    room21_loader_avoids_31,
    room21_rom_south_is_shutter,
    room21_to_31_step,
    room41_is_rom_predecessor_of_31,
    room41_loader_avoids_31,
    room41_rom_north_is_open,
    room41_to_31_step,
    room31_bomb_west_approach_step,
    room31_is_rom_predecessor_of_30,
    room31_loader_avoids_30,
    room31_rom_west_is_bomb,
    ROOM40_ROM_SOUTH,
    ROOM40_ROM_WEST,
    ROOM40_ROM_EAST,
    stair_loader_for,
    take_stairs_step,
    walk_to_step,
)
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _snap(
    *,
    screen: int = 0x60,
    mode: int = PLAY_MODE,
    link_x: int = 120,
    link_y: int = 141,
    patra: bool = False,
    eyes: int = 0,
    doors: int = 0,
    tile: int = 0x26,
):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = LEVEL9
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = link_x
    ram[ADDR_LINK_Y] = link_y
    ram[ADDR_CUR_OPENED_DOORS] = doors
    ram[ADDR_OPEN_DOORWAY_MASK] = 0
    ram[0x049E] = tile
    if patra:
        ram[ADDR_OBJ_TYPE + 1] = OBJ_PATRA
        ram[ADDR_OBJ_HP + 1] = 0xB0
        ram[ADDR_LINK_X + 1] = 120
        ram[ADDR_LINK_Y + 1] = 125
        for index in range(eyes):
            slot = index + 2
            ram[ADDR_OBJ_TYPE + slot] = OBJ_PATRA_EYE
            ram[ADDR_OBJ_HP + slot] = 0x60
            ram[ADDR_LINK_X + slot] = 80 + index * 8
            ram[ADDR_LINK_Y + slot] = 125
    return read_snapshot(ram)


def test_rom_stair_list_and_pairs() -> None:
    assert LEVEL9_STAIR_SOURCES == (0x60, 0x70, 0x72, 0x75, 0x67, 0x77, 0x00, 0x4F)
    assert 0x52 not in LEVEL9_STAIR_SOURCES
    assert LEVEL9_STAIR_LIST_INES - LEVEL9_STAIR_LIST_DC == 0x10
    assert paired_stair_dest(0x60) == 0x70
    assert paired_stair_dest(0x70) == 0x60
    assert paired_stair_dest(0x72) == 0x75
    assert paired_stair_dest(0x00) == 0x4F
    assert paired_stair_dest(0x52) is None
    assert {room for pair in LEVEL9_STAIR_PAIRS for room in pair} == set(
        LEVEL9_STAIR_SOURCES
    )


def test_preferred_loader_uses_open_or_false_door() -> None:
    loader = stair_loader_for(0x60)
    assert loader.from_room == 0x50
    assert loader.direction == "DOWN"
    assert loader.link_x == 0x78
    bottom = stair_loader_for(0x72)
    assert bottom.from_room == 0x62
    assert bottom.direction == "DOWN"
    south = stair_loader_for(0x67)
    assert south.from_room == 0x77
    assert south.direction == "UP"


def test_in_stair_source_and_walk_nav() -> None:
    snap = _snap(screen=0x60, link_x=200, link_y=180)
    assert in_stair_source(snap, 0x60)
    assert not in_stair_source(snap, 0x70)
    step = walk_to_step(snap, 120, 96, y_first=True)
    assert action_button_names(step.action) == ["UP"]
    arrived = walk_to_step(_snap(link_x=120, link_y=96), 120, 96)
    assert arrived.reason == "walk_arrived"


def test_take_stairs_and_cellar_exit_nav() -> None:
    push = take_stairs_step(
        _snap(screen=0x60, link_x=BLOCK_STAIRS_X, link_y=BLOCK_STAIRS_Y),
        source=0x60,
        target=(BLOCK_STAIRS_X, BLOCK_STAIRS_Y),
        push=True,
    )
    # Standing on the stand with push requests LEFT; on-tile idle wins if tile is stairs.
    assert push.reason in {"push_block_left", "on_stair_tile", "stand_on_stairs"}

    on_tile = take_stairs_step(
        _snap(screen=0x60, tile=0x70),
        source=0x60,
        target=(BLOCK_STAIRS_X, BLOCK_STAIRS_Y),
    )
    assert on_tile.reason == "on_stair_tile"

    cellar = cellar_exit_step(
        _snap(mode=CELLAR_MODE, screen=0x60, link_x=0x20, link_y=0x80),
        side="left",
    )
    assert action_button_names(cellar.action) == ["RIGHT"]
    assert "align_x_left" in cellar.reason
    at_mouth = cellar_exit_step(
        _snap(mode=CELLAR_MODE, screen=0x60, link_x=CELLAR_LEFT_X, link_y=0x3D),
        side="left",
    )
    assert action_button_names(at_mouth.action) == ["UP"]


def test_landed_final_patra_requires_body_eyes_closed_north() -> None:
    live = _snap(
        screen=ROOM_BEFORE_GANON,
        patra=True,
        eyes=PATRA_EYE_COUNT,
        doors=0,
    )
    assert landed_final_patra(live)
    assert dest_report(live)["landed_final_patra"] is True
    open_north = _snap(
        screen=ROOM_BEFORE_GANON,
        patra=True,
        eyes=PATRA_EYE_COUNT,
        doors=0x08,
    )
    assert not landed_final_patra(open_north)
    no_eyes = _snap(screen=ROOM_BEFORE_GANON, patra=True, eyes=0)
    assert not landed_final_patra(no_eyes)


def test_cellar_dest_table_and_patra_source() -> None:
    assert LEVEL9_CELLAR_ROOMS == (0x60, 0x70, 0x72, 0x75, 0x67, 0x77)
    assert cellar_dest_for(0x77, side="left") == 0x52
    assert cellar_dest_for(0x77, side="right") == 0x03
    assert cellar_dest_for(0x67, side="left") == 0x30
    assert cellar_dest_for(0x00, side="left") is None
    assert is_patra_cellar_source(0x77)
    assert not is_patra_cellar_source(0x67)
    assert PATRA_STAIR_SOURCE == 0x77
    assert cellar_mouth_xy(side="left") == (0x50, 0x3D)
    assert cellar_mouth_xy(side="right") == (0xB0, 0x3D)


def test_play_room_checkwarps_reverse_and_0x03_loader() -> None:
    assert play_rooms_entering_cellar(0x77) == ((0x52, "left"), (0x03, "right"))
    assert cellar_for_play_room(0x03) == (0x77, "right")
    assert cellar_for_play_room(0x52) == (0x77, "left")
    assert cellar_for_play_room(0x04) == (0x67, "right")
    assert cellar_for_play_room(0x13) is None
    assert 0x03 in PLAY_STAIR_CANDIDATES
    loader = stair_loader_for(0x03)
    assert loader.from_room == 0x13
    assert loader.direction == "UP"
    assert loader.link_x == 0x78
    assert loader.link_y == 0x58


def test_warp_tile_and_patra_cellar_predicates() -> None:
    stairs = _snap(tile=0x70)
    mouth = _snap(tile=BLACK_MOUTH_TILE)
    floor = _snap(tile=0x26)
    assert on_warp_tile(stairs)
    assert on_warp_tile(mouth)
    assert not on_warp_tile(floor)
    cellar = _snap(screen=0x77, mode=CELLAR_MODE)
    entering = _snap(screen=0x77, mode=STAIRS_ENTER_MODE)
    play = _snap(screen=0x03, mode=5)
    assert in_patra_cellar(cellar)
    assert in_patra_cellar(entering)
    assert not in_patra_cellar(play)


def _snap_room03(*, link_x: int, link_y: int, block_y: int = 144, tile: int = 0x26, doors: int = 0):
    snap_ram_base = _snap(screen=0x03, link_x=link_x, link_y=link_y, tile=tile)
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = LEVEL9
    ram[ADDR_SCREEN] = 0x03
    ram[ADDR_LINK_X] = link_x
    ram[ADDR_LINK_Y] = link_y
    ram[ADDR_CUR_OPENED_DOORS] = doors
    ram[0x049E] = tile
    ram[ADDR_OBJ_TYPE + 11] = 0x68
    ram[ADDR_LINK_X + 11] = 96
    ram[ADDR_LINK_Y + 11] = block_y
    ram[ADDR_OBJ_HP + 11] = 176
    return read_snapshot(ram)


def test_room03_push_walk_and_exact_stand() -> None:
    south = _snap_room03(link_x=120, link_y=189)
    assert not room03_west_block_pushed(south)
    step = room03_stairs_step(south)
    assert action_button_names(step.action) == ["LEFT"]

    east = _snap_room03(link_x=208, link_y=141)
    step = room03_stairs_step(east)
    assert action_button_names(step.action) == ["DOWN"]

    at_push = _snap_room03(link_x=96, link_y=170)
    step = room03_stairs_step(at_push)
    assert step.reason == "push_03_west_block_up"
    assert action_button_names(step.action) == ["UP"]

    near_push = _snap_room03(link_x=96, link_y=173)
    step = room03_stairs_step(near_push)
    assert action_button_names(step.action) == ["UP"]
    assert step.reason != "walk_arrived"

    overshoot = _snap_room03(link_x=96, link_y=149)
    step = room03_stairs_step(overshoot)
    assert action_button_names(step.action) == ["UP"]
    north_of_block = _snap_room03(link_x=96, link_y=101)
    step = room03_stairs_step(north_of_block)
    assert action_button_names(step.action) == ["LEFT"]
    # y=152 is the hold-against-block window, not an overshoot.
    hold = _snap_room03(link_x=96, link_y=152)
    step = room03_stairs_step(hold)
    assert action_button_names(step.action) == ["UP"]

    pushed = _snap_room03(link_x=96, link_y=159, block_y=128)
    assert room03_west_block_pushed(pushed)
    step = room03_stairs_step(pushed)
    assert action_button_names(step.action) == ["UP"]

    slot = _snap_room03(link_x=96, link_y=133, block_y=128)
    step = room03_stairs_step(slot)
    assert action_button_names(step.action) == ["RIGHT"]

    near = _snap_room03(link_x=128, link_y=139, block_y=128)
    step = room03_stairs_step(near)
    assert action_button_names(step.action) == ["DOWN"]
    assert walk_to_step(near, ROOM03_STAIR_X, ROOM03_STAIR_Y, tol=0).reason != "walk_arrived"

    stand = _snap_room03(link_x=ROOM03_STAIR_X, link_y=ROOM03_STAIR_Y, block_y=128)
    step = room03_stairs_step(stand)
    assert step.reason == "stand_on_03_stairs"

    # Closed east: keep the SE-corner detour (accepted play-source 03).
    east_closed = _snap_room03(link_x=176, link_y=109)
    step = room03_stairs_step(east_closed)
    assert action_button_names(step.action) == ["DOWN"]
    # Open east hole from 0x04: x-first to the east corridor, not into x=208.
    east_open = _snap_room03(link_x=160, link_y=109, doors=0x01)
    step = room03_stairs_step(east_open)
    assert action_button_names(step.action) == ["RIGHT"]
    corridor = _snap_room03(link_x=176, link_y=109, doors=0x01)
    step = room03_stairs_step(corridor)
    assert action_button_names(step.action) == ["DOWN"]
    mid = _snap_room03(link_x=144, link_y=165, doors=0x01)
    step = room03_stairs_step(mid)
    assert action_button_names(step.action) == ["RIGHT"]
    north_plus = _snap_room03(link_x=103, link_y=101, doors=0x01)
    step = room03_stairs_step(north_plus)
    assert action_button_names(step.action) == ["LEFT"]
    # Push column at y=165 still UPs (do not south-detour off the stand).
    on_col = _snap_room03(link_x=96, link_y=165)
    step = room03_stairs_step(on_col)
    assert action_button_names(step.action) == ["UP"]


def _snap_room03_like(*, link_x: int, link_y: int, like_x: int, like_y: int):
    snap = _snap_room03(link_x=link_x, link_y=link_y)
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = LEVEL9
    ram[ADDR_SCREEN] = 0x03
    ram[ADDR_LINK_X] = link_x
    ram[ADDR_LINK_Y] = link_y
    ram[ADDR_OBJ_TYPE + 5] = 0x17
    ram[ADDR_LINK_X + 5] = like_x
    ram[ADDR_LINK_Y + 5] = like_y
    ram[ADDR_OBJ_HP + 5] = 144
    ram[ADDR_OBJ_TYPE + 11] = 0x68
    ram[ADDR_LINK_X + 11] = 96
    ram[ADDR_LINK_Y + 11] = 144
    ram[ADDR_OBJ_HP + 11] = 176
    return read_snapshot(ram)


def test_room03_like_like_overlap_slashes_not_walk() -> None:
    grabbed = _snap_room03_like(link_x=96, link_y=152, like_x=96, like_y=152)
    assert room03_like_like_blocks_push(grabbed)
    frame, _cd = chase_sword_step(grabbed, 0, types=(0x17,))
    assert "A" in action_button_names(frame.action)
    assert frame.reason == "chase_overlap_slash"
    west = _snap_room03_like(link_x=122, link_y=189, like_x=80, like_y=142)
    assert room03_like_like_blocks_push(west)
    far = _snap_room03_like(link_x=122, link_y=189, like_x=160, like_y=109)
    assert not room03_like_like_blocks_push(far)


def test_cellar_exit_drops_to_corridor_from_right_mouth() -> None:
    right = _snap(mode=CELLAR_MODE, screen=0x77, link_x=192, link_y=93)
    step = cellar_exit_step(right, side="left")
    assert action_button_names(step.action) == ["DOWN"]
    assert "corridor" in step.reason
    # Live compose-30: 0x03 stairs spawned on the right well at y=165.
    low = _snap(mode=CELLAR_MODE, screen=0x77, link_x=192, link_y=165)
    step = cellar_exit_step(low, side="left")
    assert action_button_names(step.action) == ["DOWN"]
    floor = _snap(mode=CELLAR_MODE, screen=0x77, link_x=192, link_y=189)
    step = cellar_exit_step(floor, side="left")
    assert action_button_names(step.action) == ["LEFT"]
    left = _snap(mode=CELLAR_MODE, screen=0x77, link_x=CELLAR_LEFT_X, link_y=0x3D)
    step = cellar_exit_step(left, side="left")
    assert action_button_names(step.action) == ["UP"]
    pit = _snap(mode=CELLAR_MODE, screen=0x77, link_x=80, link_y=189)
    step = cellar_exit_step(pit, side="left")
    assert action_button_names(step.action) == ["LEFT"]
    on_col = _snap(mode=CELLAR_MODE, screen=0x77, link_x=0x30, link_y=189)
    step = cellar_exit_step(on_col, side="left")
    assert action_button_names(step.action) == ["UP"]
    mid = _snap(mode=CELLAR_MODE, screen=0x77, link_x=0x30, link_y=93)
    step = cellar_exit_step(mid, side="left")
    assert action_button_names(step.action) == ["UP"]


def test_room13_rom_and_live_disprove_clean_up_to_03() -> None:
    assert ROOM13 == 0x13
    assert ROOM13_ROM_NORTH == 1
    assert ROOM13_ROM_SOUTH == 5
    assert ROOM13_ROM_WEST == 5
    assert ROOM13_ROM_EAST == 1
    assert ROOM03_ROM_NORTH == 1
    assert ROOM03_ROM_SOUTH == 1
    assert ROOM03_ROM_WEST == 1
    assert ROOM03_ROM_EAST == 4
    assert room13_is_clean_predecessor_of_03() is False
    loader = stair_loader_for(ROOM13)
    assert loader.from_room == 0x23
    assert loader.direction == "UP"
    assert loader.link_x == 0x78
    assert loader.link_y == 0x58


def test_room13_nav_recenters_then_pushes_north() -> None:
    align = room13_to_03_step(_snap(screen=0x13, link_x=0x78 - 8, link_y=0xBD))
    assert action_button_names(align.action) == ["RIGHT"]
    assert align.reason == "room13_align_x"
    push = room13_to_03_step(_snap(screen=0x13, link_x=0x78, link_y=0xBD))
    assert action_button_names(push.action) == ["UP"]
    assert push.reason == "room13_push_north"
    arrived = room13_to_03_step(_snap(screen=0x03, link_x=0x78, link_y=0xBD))
    assert arrived.reason == "room03_arrived"
    assert in_room_13(_snap(screen=0x13))
    assert not in_room_13(_snap(screen=0x03))


def test_room04_rom_bomb_west_and_factory() -> None:
    assert ROOM04 == 0x04
    assert ROOM04_ROM_NORTH == 1
    assert ROOM04_ROM_SOUTH == 1
    assert ROOM04_ROM_WEST == 4
    assert ROOM04_ROM_EAST == 1
    assert ROOM02 == 0x02
    assert ROOM02_ROM_EAST == 1
    assert room04_rom_west_is_bomb() is True
    assert room04_is_rom_predecessor_of_03() is True
    assert rom_door_name(4) == "bomb"
    assert rom_door_name(1) == "wall"
    loader = stair_loader_for(ROOM04)
    assert loader.from_room == 0x14
    assert loader.direction == "UP"
    assert loader.room == ROOM04
    # Loader stages 0x14, not 0x03.
    assert loader.from_room != 0x03
    ctrl = make_room04_bomb_west_controller()
    assert ctrl.level == 9
    assert ctrl.from_room == ROOM04
    assert ctrl.to_room == 0x03
    assert ctrl.face == "LEFT"
    assert ctrl.stand == ROOM04_BOMB_WEST_STAND
    assert ctrl.stand == (48, 141)
    assert ROOM04_BOMB_WEST_APPROACH == (48, 189)
    assert in_room_04(_snap(screen=0x04))
    assert not in_room_04(_snap(screen=0x03))
    mid = room04_bomb_west_approach_step(_snap(screen=0x04, link_x=136, link_y=181))
    assert action_button_names(mid.action) == ["DOWN"]
    south = room04_bomb_west_approach_step(_snap(screen=0x04, link_x=120, link_y=189))
    assert action_button_names(south.action) == ["LEFT"]
    at_approach = room04_bomb_west_approach_step(_snap(screen=0x04, link_x=48, link_y=189))
    assert action_button_names(at_approach.action) == ["UP"]


def test_room03_rom_neighbors_table() -> None:
    rows = {row["dir"]: row for row in room03_rom_neighbors()}
    assert set(rows) == {"self", "south", "west", "east"}
    assert rows["self"]["room"] == 0x03
    assert rows["self"]["e_name"] == "bomb"
    assert rows["south"]["room"] == 0x13
    assert rows["south"]["n_name"] == "wall"
    assert rows["west"]["room"] == 0x02
    assert rows["west"]["e_name"] == "wall"
    assert rows["east"]["room"] == 0x04
    assert rows["east"]["w_name"] == "bomb"
    script = pause_select_next_b_item_script()
    assert script[0].reason == "pause_open"
    assert any(frame.reason == "pause_next_item" for frame in script)
    assert script[-1].reason == "pause_resume"


def test_room30_rom_block_stairs_and_loader() -> None:
    assert ROOM30 == 0x30
    assert ROOM30_ROM_NORTH == 1
    assert ROOM30_ROM_SOUTH == 5
    assert ROOM30_ROM_WEST == 1
    assert ROOM30_ROM_EAST == 4
    assert ROOM30_ROM_SECRET == 5
    assert room30_rom_secret_is_block_stairs() is True
    assert ROOM40 == 0x40
    assert ROOM40_ROM_NORTH == 5
    assert cellar_for_play_room(0x30) == (0x67, "left")
    assert cellar_dest_for(0x67, side="right") == 0x04
    assert play_rooms_entering_cellar(0x67) == ((0x30, "left"), (0x04, "right"))
    assert 0x30 in PLAY_STAIR_CANDIDATES
    loader = stair_loader_for(ROOM30)
    assert loader.from_room == ROOM40
    assert loader.direction == "UP"
    assert loader.room == ROOM30
    assert loader.from_room != 0x04
    assert room30_loader_avoids_04() is True
    assert in_room_30(_snap(screen=0x30))
    assert not in_room_30(_snap(screen=0x04))
    assert ROOM30_STAIR_X == BLOCK_STAIRS_X
    assert ROOM30_STAIR_Y == BLOCK_STAIRS_Y


def _snap_room30(*, link_x: int, link_y: int, block_x: int = 96, block_y: int = 144, mode: int = 5):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = LEVEL9
    ram[ADDR_SCREEN] = 0x30
    ram[ADDR_LINK_X] = link_x
    ram[ADDR_LINK_Y] = link_y
    ram[ADDR_OBJ_TYPE + 11] = 0x68
    ram[ADDR_LINK_X + 11] = block_x
    ram[ADDR_LINK_Y + 11] = block_y
    ram[ADDR_OBJ_HP + 11] = 176
    return read_snapshot(ram)


def test_room30_push_then_exact_stair_stand() -> None:
    south = _snap_room30(link_x=120, link_y=205)
    assert not room30_block_secret_open(south)
    step = room30_stairs_step(south)
    assert action_button_names(step.action) == ["LEFT"]
    at_push = _snap_room30(link_x=ROOM30_PUSH_X, link_y=ROOM30_PUSH_Y)
    step = room30_stairs_step(at_push)
    assert step.reason == "push_30_west_block_up"
    assert action_button_names(step.action) == ["UP"]
    opened = _snap_room30(link_x=96, link_y=160, block_x=208, block_y=96)
    assert room30_block_secret_open(opened)
    step = room30_stairs_step(opened)
    assert action_button_names(step.action) == ["DOWN"]
    east = _snap_room30(link_x=208, link_y=189, block_x=208, block_y=96)
    step = room30_stairs_step(east)
    assert action_button_names(step.action) == ["UP"]
    near = _snap_room30(link_x=206, link_y=93, block_x=208, block_y=96)
    step = room30_stairs_step(near)
    # tol=0: 206,93 is not arrived
    assert step.reason != "stand_on_30_stairs"
    stand = _snap_room30(link_x=ROOM30_STAIR_X, link_y=ROOM30_STAIR_Y, block_x=208, block_y=96)
    step = room30_stairs_step(stand)
    assert step.reason == "stand_on_30_stairs"
    cellar = _snap(screen=CELLAR_67, mode=CELLAR_MODE)
    assert in_cellar_67(cellar)
    assert not in_cellar_67(_snap(screen=0x30, mode=5))


def test_room30_rom_neighbors_and_40_pred() -> None:
    rows = {row["dir"]: row for row in room30_rom_neighbors()}
    assert set(rows) == {"self", "north", "south", "west", "east", "cellar_successor"}
    assert rows["self"]["room"] == 0x30
    assert rows["self"]["s_name"] == "key"
    assert rows["self"]["e_name"] == "bomb"
    assert rows["north"]["room"] == ROOM20
    assert rows["north"]["s_name"] == "wall"
    assert rows["south"]["room"] == ROOM40
    assert rows["south"]["n_name"] == "key"
    assert rows["west"]["room"] == ROOM2F
    assert rows["west"]["e_name"] == "shutter"
    assert rows["east"]["room"] == ROOM31
    assert rows["east"]["w_name"] == "bomb"
    assert rows["cellar_successor"]["room"] == 0x67
    assert ROOM40_ROM_SOUTH == 5
    assert ROOM40_ROM_WEST == 1
    assert ROOM40_ROM_EAST == 1
    assert room40_rom_north_is_key() is True
    assert room40_is_rom_predecessor_of_30() is True
    loader = stair_loader_for(ROOM40)
    assert loader.from_room == 0x50
    assert loader.direction == "UP"
    assert loader.room == ROOM40
    assert loader.from_room != ROOM30
    assert room40_loader_avoids_30() is True
    assert in_room_40(_snap(screen=0x40))
    assert not in_room_40(_snap(screen=0x30))
    spawn = room40_to_30_step(_snap(screen=0x40, link_x=120, link_y=205))
    assert action_button_names(spawn.action) == ["UP"]
    assert spawn.reason == "room40_push_north"
    off = room40_to_30_step(_snap(screen=0x40, link_x=96, link_y=125))
    assert action_button_names(off.action) == ["DOWN"]
    south = room40_to_30_step(_snap(screen=0x40, link_x=96, link_y=189))
    assert action_button_names(south.action) == ["RIGHT"]
    door = room40_to_30_step(_snap(screen=0x40, link_x=120, link_y=109))
    assert action_button_names(door.action) == ["UP"]
    arrived = room40_to_30_step(_snap(screen=0x30, link_x=120, link_y=205))
    assert arrived.reason == "room30_arrived"
    scroll = room40_to_30_step(_snap(screen=0x30, link_x=120, link_y=221, mode=4))
    assert action_button_names(scroll.action) == ["UP"]
    assert scroll.reason == "room30_scroll"
    # 0x20 default would be south-from-0x30; preferred loader stages 0x10.
    loader20 = stair_loader_for(ROOM20)
    assert loader20.from_room != ROOM30


def test_room31_rom_bomb_west_and_factory() -> None:
    assert ROOM31 == 0x31
    assert ROOM31_ROM_NORTH == 0
    assert ROOM31_ROM_SOUTH == 7
    assert ROOM31_ROM_WEST == 4
    assert ROOM31_ROM_EAST == 1
    assert ROOM30_ROM_EAST == 4
    assert room31_rom_west_is_bomb() is True
    assert room31_is_rom_predecessor_of_30() is True
    loader = stair_loader_for(ROOM31)
    assert loader.from_room == ROOM41
    assert loader.direction == "UP"
    assert loader.room == ROOM31
    assert loader.from_room != ROOM30
    assert room31_loader_avoids_30() is True
    ctrl = make_room31_bomb_west_controller()
    assert ctrl.level == 9
    assert ctrl.from_room == ROOM31
    assert ctrl.to_room == ROOM30
    assert ctrl.face == "LEFT"
    assert ctrl.stand == ROOM31_BOMB_WEST_STAND
    assert ctrl.stand == (48, 141)
    assert ROOM31_BOMB_WEST_APPROACH == (48, 189)
    assert in_room_31(_snap(screen=0x31))
    assert not in_room_31(_snap(screen=0x30))
    mid = room31_bomb_west_approach_step(_snap(screen=0x31, link_x=136, link_y=181))
    assert action_button_names(mid.action) == ["DOWN"]
    south = room31_bomb_west_approach_step(_snap(screen=0x31, link_x=120, link_y=189))
    assert action_button_names(south.action) == ["LEFT"]
    at_approach = room31_bomb_west_approach_step(_snap(screen=0x31, link_x=48, link_y=189))
    assert action_button_names(at_approach.action) == ["UP"]


def test_room21_rom_south_shutter_and_loader() -> None:
    assert ROOM21 == 0x21
    assert ROOM21_ROM_NORTH == 0
    assert ROOM21_ROM_SOUTH == 7
    assert ROOM21_ROM_WEST == 1
    assert ROOM21_ROM_EAST == 4
    assert ROOM31_ROM_NORTH == 0
    assert room21_rom_south_is_shutter() is True
    assert room21_is_rom_predecessor_of_31() is True
    loader = stair_loader_for(ROOM21)
    assert loader.from_room == ROOM11
    assert loader.direction == "DOWN"
    assert loader.room == ROOM21
    assert loader.from_room != ROOM31
    assert room21_loader_avoids_31() is True
    assert in_room_21(_snap(screen=0x21))
    assert not in_room_21(_snap(screen=0x31))
    # North doorway on the plus: slide west first.
    north = room21_to_31_step(_snap(screen=0x21, link_x=120, link_y=69))
    assert action_button_names(north.action) == ["LEFT"]
    assert north.reason == "room21_off_plus"
    west = room21_to_31_step(_snap(screen=0x21, link_x=48, link_y=69))
    assert action_button_names(west.action) == ["DOWN"]
    band = room21_to_31_step(_snap(screen=0x21, link_x=48, link_y=189))
    assert action_button_names(band.action) == ["RIGHT"]
    assert band.reason == "room21_align_x"
    door = room21_to_31_step(_snap(screen=0x21, link_x=120, link_y=189))
    assert action_button_names(door.action) == ["DOWN"]
    assert door.reason == "room21_push_south"
    arrived = room21_to_31_step(_snap(screen=0x31, link_x=120, link_y=69))
    assert arrived.reason == "room31_arrived"
    scroll = room21_to_31_step(_snap(screen=0x31, link_x=120, link_y=61, mode=4))
    assert action_button_names(scroll.action) == ["DOWN"]
    assert scroll.reason == "room31_scroll"
    assert ROOM21_SOUTH_Y == 189
    assert ROOM21_WEST_X == 48


def test_room41_rom_north_open_and_loader() -> None:
    assert ROOM41 == 0x41
    assert ROOM41_ROM_NORTH == 0
    assert ROOM41_ROM_SOUTH == 7
    assert ROOM41_ROM_WEST == 1
    assert ROOM41_ROM_EAST == 1
    assert ROOM31_ROM_SOUTH == 7
    assert room41_rom_north_is_open() is True
    assert room41_is_rom_predecessor_of_31() is True
    loader = stair_loader_for(ROOM41)
    assert loader.from_room == ROOM51
    assert loader.direction == "UP"
    assert loader.room == ROOM41
    assert loader.from_room != ROOM31
    assert room41_loader_avoids_31() is True
    assert in_room_41(_snap(screen=0x41))
    assert not in_room_41(_snap(screen=0x31))
    spawn = room41_to_31_step(_snap(screen=0x41, link_x=120, link_y=205))
    assert action_button_names(spawn.action) == ["UP"]
    assert spawn.reason == "room41_push_north"
    off = room41_to_31_step(_snap(screen=0x41, link_x=96, link_y=125))
    assert action_button_names(off.action) == ["DOWN"]
    south = room41_to_31_step(_snap(screen=0x41, link_x=96, link_y=189))
    assert action_button_names(south.action) == ["RIGHT"]
    door = room41_to_31_step(_snap(screen=0x41, link_x=120, link_y=109))
    assert action_button_names(door.action) == ["UP"]
    arrived = room41_to_31_step(_snap(screen=0x31, link_x=120, link_y=205))
    assert arrived.reason == "room31_arrived"
    scroll = room41_to_31_step(_snap(screen=0x31, link_x=120, link_y=221, mode=4))
    assert action_button_names(scroll.action) == ["UP"]
    assert scroll.reason == "room31_scroll"
    assert ROOM41_SOUTH_Y == 189
    # 0x21 south shutter sealed after Patra is not a clean 0x31 pred.
    assert room21_rom_south_is_shutter() is True


def test_room51_rom_north_open_and_loader() -> None:
    from zelda_i.level9_room51 import (
        ROOM51_SOUTH_Y,
        room51_is_rom_predecessor_of_41,
        room51_loader_avoids_41,
        room51_rom_north_is_open,
        room51_to_41_step,
    )

    assert ROOM51 == 0x51
    assert ROOM51_ROM_NORTH == 0
    assert ROOM51_ROM_SOUTH == 0
    assert ROOM51_ROM_WEST == 7
    assert ROOM51_ROM_EAST == 1
    assert ROOM51_ROM_SECRET == 1
    assert ROOM41_ROM_SOUTH == 7
    assert ROOM61 == 0x61
    assert ROOM61_ROM_NORTH == 0
    assert ROOM61_ROM_SOUTH == 0
    assert ROOM61_ROM_WEST == 1
    assert ROOM61_ROM_EAST == 0
    assert ROOM50 == 0x50
    assert ROOM50_ROM_NORTH == 5
    assert ROOM50_ROM_EAST == 7
    assert room51_rom_north_is_open() is True
    assert room51_is_rom_predecessor_of_41() is True
    loader = stair_loader_for(ROOM51)
    assert loader.from_room == ROOM61
    assert loader.direction == "UP"
    assert loader.room == ROOM51
    assert loader.from_room != ROOM41
    assert room51_loader_avoids_41() is True
    assert in_room_51(_snap(screen=0x51))
    assert not in_room_51(_snap(screen=0x41))
    spawn = room51_to_41_step(_snap(screen=0x51, link_x=120, link_y=205))
    assert action_button_names(spawn.action) == ["UP"]
    assert spawn.reason == "room51_center_to_thread"
    pocket = room51_to_41_step(_snap(screen=0x51, link_x=136, link_y=108))
    assert action_button_names(pocket.action) == ["DOWN"]
    assert pocket.reason == "room51_drop_to_thread"
    slide = room51_to_41_step(_snap(screen=0x51, link_x=120, link_y=133))
    assert action_button_names(slide.action) == ["RIGHT"]
    climb = room51_to_41_step(_snap(screen=0x51, link_x=144, link_y=133))
    assert action_button_names(climb.action) == ["UP"]
    assert climb.reason == "room51_climb_thread"
    door = room51_to_41_step(_snap(screen=0x51, link_x=120, link_y=93))
    assert action_button_names(door.action) == ["UP"]
    assert door.reason == "room51_push_north"
    arrived = room51_to_41_step(_snap(screen=0x41, link_x=120, link_y=205))
    assert arrived.reason == "room41_arrived"
    scroll = room51_to_41_step(_snap(screen=0x41, link_x=120, link_y=221, mode=4))
    assert action_button_names(scroll.action) == ["UP"]
    assert scroll.reason == "room41_scroll"
    assert ROOM51_SOUTH_Y == 189
    # Dirty 0x40 is 0x50's north-key neighbor, not this predecessor chain.
    assert ROOM50_ROM_NORTH == 5


def test_run_level9_stairs_is_thin_cli() -> None:
    from pathlib import Path

    from zelda_i.scripts import run_level9_stairs as cli

    script = Path(cli.__file__)
    assert script.read_text().count("\n") < 700
    assert hasattr(cli, "main")
    assert cli.materialize_stair_room.__module__ == "zelda_i.level9_stair_run"
    assert cli.take_stairs_from_source.__module__ == "zelda_i.level9_stair_run"
    assert cli.probe_sources.__module__ == "zelda_i.level9_stair_probe"
    assert cli.build_winning_fixture.__module__ == "zelda_i.level9_stair_suffix"


def test_run_level9_stairs_reexports_probe_helpers() -> None:
    from zelda_i.scripts import run_level9_stairs as cli

    for name in (
        "materialize_stair_room",
        "_apply_loader",
        "_hold_until_room",
        "_exit_cellar",
        "_walk_target",
        "dump_room_tiles",
        "take_stairs_from_source",
    ):
        assert callable(getattr(cli, name)), name


def test_run_level9_stairs_cli_flags() -> None:
    from zelda_i.scripts.run_level9_stairs import TAG, main

    # Flags used by docs / probes; add_common_args supplies shared ones.
    import argparse

    parser = argparse.ArgumentParser()
    from zelda_i.runner import add_common_args

    add_common_args(parser, default_state="", default_tag=TAG, default_trials=1)
    args = parser.parse_args([])
    assert args.tag == TAG
    assert args.trials == 1
    assert args.infinite_life is True
    assert args.save_state is False
    assert callable(main)


def test_level9_stair_library_files_under_1k() -> None:
    from pathlib import Path

    import zelda_i

    root = Path(zelda_i.__file__).resolve().parent
    for name in (
        "level9_stair_run.py",
        "level9_stair_suffix.py",
        "level9_stair_probe.py",
        "level9_stair_east.py",
        "level9_stair_west.py",
        "level9_stair_north.py",
        "level9_room51.py",
    ):
        lines = (root / name).read_text().count("\n")
        assert lines < 1000, f"{name} is {lines} LOC"
