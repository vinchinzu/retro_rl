"""Unit tests for the White Sword detour controller and its Level 9 wiring."""

from __future__ import annotations

from zelda_i.level9.dungeon import (
    FULL_TRIFORCE,
    MAGICAL_SWORD,
    MEASURED_POST_L8_HANDOFF,
    WHITE_SWORD,
    level9_live_patra_stop,
)
from zelda_i.level9.hops import level9_entry_chapter
from zelda_i.overworld.white_sword import (
    MIN_HEART_CONTAINERS,
    SCREEN_WHITE_SWORD_CAVE,
    WhiteSwordDetourController,
    WhiteSwordPhase,
    make_white_sword_detour_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

CAVE_MODE = 11


def _snap(
    *,
    level: int = 0,
    screen: int = 0x05,
    mode: int = PLAY_MODE,
    sword: int = 1,
    containers: int = 10,
    link_x: int = 240,
    link_y: int = 141,
) -> ZeldaSnapshot:
    # health high nibble is containers - 1; the assist only refills the low one.
    health = ((containers - 1) << 4) | (containers - 1)
    return ZeldaSnapshot(
        mode=mode, level=level, screen=screen, next_screen=screen,
        link_x=link_x, link_y=link_y, facing=0x08, sword=sword, bombs=14,
        rupees=0, keys=0, health=health, triforce=FULL_TRIFORCE, compass=0,
        dialog_timer=0, colliding_tile=0, room_item_id=0, room_all_dead=0,
        room_obj_count=0, cur_opened_doors=0, open_doorway_mask=0, objects=(),
    )


def test_detour_starts_on_the_level9_approach_screen() -> None:
    ctl = make_white_sword_detour_controller()
    ctl.step(_snap())
    assert not ctl.failed
    assert ctl.phase is WhiteSwordPhase.OUT_LEGS


def test_detour_refuses_off_contract_start() -> None:
    ctl = make_white_sword_detour_controller()
    ctl.step(_snap(screen=0x77))
    assert ctl.failed
    assert ctl.notes == ["white_sword_predecessor_contract_miss"]


def test_detour_refuses_below_the_heart_container_gate() -> None:
    """The Old Man gates on containers; infinite-life fill cannot open him."""
    ctl = make_white_sword_detour_controller()
    ctl.step(_snap(containers=MIN_HEART_CONTAINERS - 1))
    assert ctl.failed
    assert ctl.notes == ["white_sword_heart_container_gate"]


def test_detour_at_the_gate_is_allowed() -> None:
    ctl = make_white_sword_detour_controller()
    ctl.step(_snap(containers=MIN_HEART_CONTAINERS))
    assert not ctl.failed


def test_detour_short_circuits_when_the_sword_is_already_held() -> None:
    ctl = make_white_sword_detour_controller()
    ctl.step(_snap(sword=WHITE_SWORD))
    assert ctl.success
    assert ctl.phase is WhiteSwordPhase.DONE


def test_detour_writes_nothing() -> None:
    ctl = make_white_sword_detour_controller()
    ctl.step(_snap())
    rep = ctl.report()
    assert rep["controller_memory_writes"] == 0
    assert rep["progression_writes"] == 0
    assert rep["inventory_writes"] == 0
    assert rep["evidence"] == "live"


def test_take_sword_waits_out_the_old_man_before_walking() -> None:
    ctl = WhiteSwordDetourController(phase=WhiteSwordPhase.TAKE_SWORD)
    ctl.start_checked = True
    snap = _snap(screen=SCREEN_WHITE_SWORD_CAVE, mode=CAVE_MODE, link_x=32, link_y=213)
    assert not any(ctl.step(snap).action)  # stand through the dialog


def test_take_sword_exits_once_the_sword_is_in_hand() -> None:
    ctl = WhiteSwordDetourController(phase=WhiteSwordPhase.TAKE_SWORD)
    ctl.start_checked = True
    snap = _snap(screen=SCREEN_WHITE_SWORD_CAVE, mode=CAVE_MODE,
                 sword=WHITE_SWORD, link_x=120, link_y=157)
    assert ctl.step(snap).reason == "sword_taken"
    assert ctl.phase is WhiteSwordPhase.EXIT_CAVE


def test_entry_chapter_runs_the_detour_before_the_rock_bomb() -> None:
    stages = level9_entry_chapter(handoff=MEASURED_POST_L8_HANDOFF)
    names = [name for name, _, _ in stages]
    assert names == [
        "level9_post_l8_overworld",
        "bomb_restock_l8",
        "exit_bomb_restock_l8",
        "level9_post_l8_to_rock",
        "level9_white_sword",
        "level9_spectacle_rock_bomb",
    ]


def _live_patra_snap(sword: int) -> ZeldaSnapshot:
    from zelda_i.level9.dungeon import LEVEL9, ROOM_FINAL_PATRA, SILVER_ARROWS
    from zelda_i.level9.patra import OBJ_PATRA, OBJ_PATRA_EYE, PATRA_EYE_COUNT
    from zelda_i.ram import ZeldaObject

    eyes = tuple(
        ZeldaObject(slot=i, type_id=OBJ_PATRA_EYE, hp=0x60, x=100 + i * 4, y=100,
                    facing=0, state=0)
        for i in range(1, PATRA_EYE_COUNT + 1)
    )
    body = ZeldaObject(slot=10, type_id=OBJ_PATRA, hp=0xB0, x=120, y=100,
                       facing=0, state=0)
    return ZeldaSnapshot(
        mode=PLAY_MODE, level=LEVEL9, screen=ROOM_FINAL_PATRA,
        next_screen=ROOM_FINAL_PATRA, link_x=120, link_y=157, facing=0x08,
        sword=sword, bombs=6, rupees=0, keys=0, health=0x99,
        triforce=FULL_TRIFORCE, compass=0, dialog_timer=0, colliding_tile=0,
        room_item_id=0, room_all_dead=0, room_obj_count=9, cur_opened_doors=0,
        open_doorway_mask=0, objects=eyes + (body,), bow=1,
        arrows=SILVER_ARROWS,
    )


def test_patra_stop_accepts_the_white_sword() -> None:
    """The Magical Sword needs 12 containers; the natural run arrives with 10."""
    assert WHITE_SWORD < MAGICAL_SWORD
    assert level9_live_patra_stop(_live_patra_snap(WHITE_SWORD))


def test_patra_stop_still_rejects_the_wooden_sword() -> None:
    assert not level9_live_patra_stop(_live_patra_snap(1))
