"""Unit tests for NaturalPatraJoinController, chapter wiring, and stop predicates."""

from __future__ import annotations

from zelda_i.level9.dungeon import (
    FULL_TRIFORCE,
    LEVEL9,
    MAGICAL_SWORD,
    PATRA_EYE_COUNT,
    ROOM_FINAL_PATRA,
    SILVER_ARROWS,
    level9_credits_stop,
    level9_live_patra_stop,
)
from zelda_i.level9.hops import (
    Level9NaturalRouteSelection,
    level9_patra_chapter,
)
from zelda_i.level9.natural_path import (
    NaturalPatraJoinController,
    NaturalRouteUnavailableController,
    PatraJoinPhase,
    make_natural_patra_join_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot


def _make_snap(
    *,
    level: int = LEVEL9,
    screen: int = 0x10,
    mode: int = PLAY_MODE,
    submode: int = 0,
    is_updating_mode: int = 1,
    triforce: int = FULL_TRIFORCE,
    bombs: int = 12,
    bow: int = 1,
    arrows: int = SILVER_ARROWS,
    sword: int = MAGICAL_SWORD,
    link_x: int = 120,
    link_y: int = 189,
    facing: int = 0x08,
    cur_opened_doors: int = 0,
    objects: tuple[ZeldaObject, ...] = (),
) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=mode,
        level=level,
        screen=screen,
        next_screen=screen,
        link_x=link_x,
        link_y=link_y,
        facing=facing,
        sword=sword,
        bombs=bombs,
        rupees=0,
        keys=0,
        health=0x4F,
        triforce=triforce,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=len(objects),
        cur_opened_doors=cur_opened_doors,
        open_doorway_mask=0,
        objects=objects,
        submode=submode,
        is_updating_mode=is_updating_mode,
        bow=bow,
        arrows=arrows,
    )


def test_natural_patra_join_controller_init():
    ctrl = make_natural_patra_join_controller()
    assert isinstance(ctrl, NaturalPatraJoinController)
    assert ctrl.phase == PatraJoinPhase.SOUTH_10
    assert ctrl.max_frames == 24000
    assert ctrl.frames == 0
    assert ctrl.phase_frames == 0
    assert not ctrl.success
    assert not ctrl.failed
    report = ctrl.report()
    assert report["phase"] == "SOUTH_10"
    assert report["fixture_loaded"] is False
    assert report["route_eligible"] is False
    assert report["controller_memory_writes"] == 0
    assert report["progression_writes"] == 0
    assert report["capacity_writes"] == 0
    assert report["inventory_writes"] == 0


def test_natural_patra_join_fail_closed_contracts():
    ctrl = make_natural_patra_join_controller()
    bad_tf = _make_snap(triforce=0x7F)
    ctrl.step(bad_tf)
    assert ctrl.failed
    assert "contract_miss" in ctrl.notes[-1]

    ctrl2 = make_natural_patra_join_controller()
    bad_bombs = _make_snap(bombs=0)
    ctrl2.step(bad_bombs)
    assert ctrl2.failed

    ctrl3 = make_natural_patra_join_controller()
    bad_screen = _make_snap(screen=0x05)
    ctrl3.step(bad_screen)
    assert ctrl3.failed

    ctrl4 = make_natural_patra_join_controller()
    death = _make_snap(mode=17)
    ctrl4.step(death)
    assert ctrl4.failed
    assert "link_death" in ctrl4.notes[-1]


def test_patra_chapter_wiring():
    # When suffix join room is selected, returns make_natural_patra_join_controller
    stages = level9_patra_chapter()
    assert len(stages) == 1
    name, ctrl, max_f = stages[0]
    assert name == "level9_natural_patra_join"
    assert isinstance(ctrl, NaturalPatraJoinController)
    assert max_f == 24000

    # When suffix join room is None, returns fail-closed unavailable controller
    empty_route = Level9NaturalRouteSelection(suffix_join_room=None)
    unavail_stages = level9_patra_chapter(empty_route)
    assert len(unavail_stages) == 1
    u_name, u_ctrl, _ = unavail_stages[0]
    assert u_name == "level9_natural_patra_join"
    assert isinstance(u_ctrl, NaturalRouteUnavailableController)


def test_level9_live_patra_stop_predicate():
    eyes = tuple(
        ZeldaObject(slot=i, type_id=0x25, hp=16, x=100 + i * 4, y=100, facing=0, state=0)
        for i in range(1, PATRA_EYE_COUNT + 1)
    )
    patra_body = ZeldaObject(slot=10, type_id=0x47, hp=100, x=120, y=100, facing=0, state=0)
    good = _make_snap(
        screen=ROOM_FINAL_PATRA,
        cur_opened_doors=0,
        objects=eyes + (patra_body,),
    )
    assert level9_live_patra_stop(good)

    # Missing eyes
    assert not level9_live_patra_stop(_make_snap(
        screen=ROOM_FINAL_PATRA,
        objects=(patra_body,),
    ))

    # North door already open
    assert not level9_live_patra_stop(_make_snap(
        screen=ROOM_FINAL_PATRA,
        cur_opened_doors=0x08,
        objects=eyes + (patra_body,),
    ))

    # Wrong screen
    assert not level9_live_patra_stop(_make_snap(
        screen=0x42,
        objects=eyes + (patra_body,),
    ))


def test_level9_credits_stop_predicate():
    rolling = _make_snap(mode=0x13, submode=3, is_updating_mode=1)
    assert level9_credits_stop(rolling, deaths=0)
    assert not level9_credits_stop(rolling, deaths=1)
    not_ending = _make_snap(mode=5, submode=0, is_updating_mode=1)
    assert not level9_credits_stop(not_ending, deaths=0)

