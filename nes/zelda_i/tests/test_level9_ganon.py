from __future__ import annotations

import numpy as np

from zelda_i.dungeon_trace import action_button_names
from zelda_i.level9_ganon import (
    ADDR_GANON_OBJ_PHASE_BASE,
    ADDR_GANON_SCENE_PHASE,
    ADDR_LAST_BOSS_DEFEATED,
    B_ITEM_ARROWS,
    B_ITEM_BOMBS,
    ENDING_SUBMODE_CREDITS,
    ENDING_SUBMODE_FINAL_SCREEN,
    GANON_HP_START,
    MODE_ENDING,
    OBJ_GANON,
    OBJ_ZELDA,
    ROOM_BEFORE_GANON,
    ROOM_GANON,
    ROOM_ZELDA,
    credits_rolling,
    final_ending_screen,
    ganon_action,
    ganon_is_brown,
    in_ganon_fight,
    in_room_before_ganon,
    in_zelda_room,
)
from zelda_i.level9_overworld import (
    ROOM_LEVEL9_ENTRY,
    level9_ending_stop,
    level9_entry_stop,
    level9_ganon_planning_notes,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_IS_UPDATING_MODE,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_SUBMODE,
    PLAY_MODE,
    read_snapshot,
)


def _snapshot(
    *,
    mode: int = PLAY_MODE,
    level: int = 9,
    room: int = ROOM_GANON,
    obj_type: int = OBJ_GANON,
    obj_state: int = 0,
    obj_x: int = 120,
    obj_y: int = 100,
    link_x: int = 120,
    link_y: int = 122,
    submode: int = 0,
    updating: int = 1,
):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_SUBMODE] = submode
    ram[ADDR_IS_UPDATING_MODE] = updating
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = link_x
    ram[ADDR_LINK_Y] = link_y
    ram[ADDR_HEALTH] = 0xFF
    ram[ADDR_OBJ_TYPE + 1] = obj_type
    ram[ADDR_LINK_X + 1] = obj_x
    ram[ADDR_LINK_Y + 1] = obj_y
    ram[0x00AC + 1] = obj_state
    ram[ADDR_OBJ_HP + 1] = GANON_HP_START
    return read_snapshot(ram)


def test_live_level9_room_and_ram_anchors() -> None:
    assert ROOM_LEVEL9_ENTRY == 0x76
    assert ROOM_BEFORE_GANON == 0x52
    assert ROOM_GANON == 0x42
    assert ROOM_ZELDA == 0x32
    assert OBJ_GANON == 0x3E
    assert OBJ_ZELDA == 0x37
    assert ADDR_GANON_OBJ_PHASE_BASE == 0x042C
    assert ADDR_GANON_SCENE_PHASE == 0x0445
    assert ADDR_LAST_BOSS_DEFEATED == 0x0672
    assert B_ITEM_BOMBS == 1
    assert B_ITEM_ARROWS == 2


def test_room_predicates_use_live_ids() -> None:
    assert in_ganon_fight(_snapshot())
    assert in_room_before_ganon(
        _snapshot(room=ROOM_BEFORE_GANON, obj_type=0)
    )
    assert in_zelda_room(_snapshot(room=ROOM_ZELDA, obj_type=OBJ_ZELDA))
    assert level9_entry_stop(_snapshot(room=ROOM_LEVEL9_ENTRY, obj_type=0))


def test_ganon_brown_is_any_nonzero_object_state() -> None:
    assert not ganon_is_brown(_snapshot(obj_state=0))
    # The engine seeds FF then commonly exposes FE after the first decrement.
    assert ganon_is_brown(_snapshot(obj_state=0xFE))


def test_ganon_action_pulses_sword_then_aligned_silver_arrow() -> None:
    sword, sword_reason, sword_cooldown = ganon_action(
        _snapshot(obj_state=0), cooldown=0
    )
    assert sword_reason == "sword_pulse"
    assert set(action_button_names(sword)) == {"A", "UP"}
    assert sword_cooldown == 12

    arrow, arrow_reason, arrow_cooldown = ganon_action(
        _snapshot(obj_state=0xFE), cooldown=0
    )
    assert arrow_reason == "silver_arrow"
    assert set(action_button_names(arrow)) == {"B", "UP"}
    assert arrow_cooldown == 16


def test_ending_stop_distinguishes_init_submodes_from_update_credits() -> None:
    init_credits_number = _snapshot(
        mode=MODE_ENDING,
        submode=ENDING_SUBMODE_CREDITS,
        updating=0,
    )
    assert not credits_rolling(init_credits_number)
    assert not level9_ending_stop(init_credits_number)

    rolling = _snapshot(
        mode=MODE_ENDING,
        submode=ENDING_SUBMODE_CREDITS,
        updating=1,
    )
    assert credits_rolling(rolling)
    assert level9_ending_stop(rolling)

    final = _snapshot(
        mode=MODE_ENDING,
        submode=ENDING_SUBMODE_FINAL_SCREEN,
        updating=1,
    )
    assert final_ending_screen(final)
    assert level9_ending_stop(final)


def test_ganon_report_is_live_verified() -> None:
    report = level9_ganon_planning_notes()
    assert report["live_verified"] is True
    assert report["object_type_id"] == OBJ_GANON
