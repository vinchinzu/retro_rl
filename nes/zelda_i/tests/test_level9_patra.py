from __future__ import annotations

import numpy as np

from zelda_i.dungeon_trace import action_button_names
from zelda_i.level9_ganon import LEVEL9, ROOM_BEFORE_GANON
from zelda_i.level9_patra import (
    NORTH_DOOR,
    OBJ_PATRA,
    OBJ_PATRA_EYE,
    PATRA_ATTACK_COOLDOWN,
    PATRA_BODY_HP_START,
    PATRA_EYE_COUNT,
    PATRA_EYE_HP_START,
    final_patra_live,
    final_patra_north_door_earned,
    patra_action,
    patra_eyes,
)
from zelda_i.level9_path import (
    NORTH_DOOR_X,
    final_patra_to_ganon_step,
)
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.scripts.run_level9_ganon import _fixture_write_rows


def _patra_snapshot(
    *,
    link_x: int = 128,
    link_y: int = 142,
    body_x: int = 128,
    body_y: int = 112,
    eyes: int = PATRA_EYE_COUNT,
    body: bool = True,
    north_door: bool = False,
):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = LEVEL9
    ram[ADDR_SCREEN] = ROOM_BEFORE_GANON
    ram[ADDR_LINK_X] = link_x
    ram[ADDR_LINK_Y] = link_y
    ram[ADDR_HEALTH] = 0xFF
    if body:
        ram[ADDR_OBJ_TYPE + 1] = OBJ_PATRA
        ram[ADDR_LINK_X + 1] = body_x
        ram[ADDR_LINK_Y + 1] = body_y
        ram[ADDR_OBJ_HP + 1] = PATRA_BODY_HP_START
    for index in range(eyes):
        slot = index + 2
        ram[ADDR_OBJ_TYPE + slot] = OBJ_PATRA_EYE
        ram[ADDR_LINK_X + slot] = body_x + index
        ram[ADDR_LINK_Y + slot] = body_y
        ram[ADDR_OBJ_HP + slot] = PATRA_EYE_HP_START
    if north_door:
        ram[ADDR_CUR_OPENED_DOORS] = NORTH_DOOR
    return read_snapshot(ram)


def test_live_final_patra_anchors_and_liveness() -> None:
    snap = _patra_snapshot()
    assert OBJ_PATRA == 0x47
    assert OBJ_PATRA_EYE == 0x25
    assert PATRA_BODY_HP_START == 0xB0
    assert PATRA_EYE_HP_START == 0x60
    assert final_patra_live(snap)
    assert len(patra_eyes(snap)) == 8
    assert not final_patra_north_door_earned(snap)


def test_patra_action_walks_to_south_stand_then_pulses_up() -> None:
    approach, approach_reason, approach_cooldown = patra_action(
        _patra_snapshot(link_x=96, link_y=142), cooldown=0
    )
    assert action_button_names(approach) == ["RIGHT"]
    assert approach_reason == "align_south_x"
    assert approach_cooldown == 0

    attack, attack_reason, attack_cooldown = patra_action(_patra_snapshot(), cooldown=0)
    assert set(action_button_names(attack)) == {"A", "UP"}
    assert attack_reason == "sword_pulse_up"
    assert attack_cooldown == PATRA_ATTACK_COOLDOWN


def test_patra_clear_requires_natural_north_door_bit() -> None:
    dead_closed = _patra_snapshot(body=False, eyes=0, north_door=False)
    assert not final_patra_live(dead_closed)
    assert not final_patra_north_door_earned(dead_closed)

    dead_open = _patra_snapshot(body=False, eyes=0, north_door=True)
    assert final_patra_north_door_earned(dead_open)


def test_final_patra_door_recenters_before_north_push() -> None:
    align = final_patra_to_ganon_step(
        _patra_snapshot(link_x=NORTH_DOOR_X - 8, body=False, eyes=0)
    )
    assert action_button_names(align.action) == ["RIGHT"]
    assert align.reason == "ganon_align_x"

    push = final_patra_to_ganon_step(
        _patra_snapshot(link_x=NORTH_DOOR_X, body=False, eyes=0)
    )
    assert action_button_names(push.action) == ["UP"]
    assert push.reason == "ganon_push_north"


def test_live_patra_fixture_excludes_skip_and_door_writes() -> None:
    live_names = {row["name"] for row in _fixture_write_rows(clear_final_patra=False)}
    assert "loader_door_staging" in live_names
    assert "clear_final_patra_object_slots" not in live_names
    assert "clear_room_object_count" not in live_names
    assert "mark_room_all_dead" not in live_names
    assert "open_north_door" not in live_names

    skipped_names = {row["name"] for row in _fixture_write_rows()}
    assert "clear_final_patra_object_slots" in skipped_names
    assert "open_north_door" in skipped_names
