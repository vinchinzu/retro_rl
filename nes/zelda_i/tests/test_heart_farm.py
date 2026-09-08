"""Unit tests for overworld heart-farm restock. No emulator."""

from __future__ import annotations

from retro_harness.nes import nes_action

from zelda_i.dungeon.ids import RUPEE_DROP_OBJECT_TYPE
from zelda_i.overworld.heart_farm import HeartFarmController, HeartFarmPhase
from zelda_i.overworld.locations import farm_at
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

FARM_SCREEN = 0x4A
NEIGHBOR_SCREEN = 0x49
NO_RESTOCK_SCREEN = 0x77


def _snap(**kwargs) -> ZeldaSnapshot:
    fields = dict(
        mode=PLAY_MODE,
        level=0,
        screen=FARM_SCREEN,
        next_screen=FARM_SCREEN,
        link_x=120,
        link_y=149,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=0x22,  # 2 filled, below min_filled=3
        triforce=0,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )
    fields.update(kwargs)
    return ZeldaSnapshot(**fields)


def _farm(**overrides) -> HeartFarmController:
    kwargs = dict(
        min_filled=3,
        farm_screen=FARM_SCREEN,
        restock_neighbor_screen=NEIGHBOR_SCREEN,
        restock_direction="LEFT",
        empty_wait_frames=5,
    )
    kwargs.update(overrides)
    return HeartFarmController(**kwargs)


def test_constructor_without_restock_kwargs_still_works() -> None:
    farm = HeartFarmController(min_filled=3, farm_screen=NO_RESTOCK_SCREEN)
    assert farm.min_filled == 3
    assert farm.farm_screen == NO_RESTOCK_SCREEN
    assert farm.restock_neighbor_screen is None
    assert farm.restock_direction is None
    assert farm.phase is HeartFarmPhase.FARM


def test_farm_at_4a_fills_restock_from_catalog() -> None:
    spot = farm_at(0x4A)
    assert spot is not None
    assert spot.restock_neighbor == 0x49
    assert spot.restock_direction == "LEFT"
    farm = HeartFarmController(farm_screen=0x4A)
    assert farm.restock_neighbor_screen == 0x49
    assert farm.restock_direction == "LEFT"


def test_no_restock_leave_screen_still_fails() -> None:
    farm = HeartFarmController(min_filled=3, farm_screen=NO_RESTOCK_SCREEN)
    act = farm.step(_snap(screen=0x78, health=0x22))
    assert act.reason == "left_farm_screen"
    assert farm.phase is HeartFarmPhase.FAILED
    assert not farm.success


def test_empty_farm_screen_eventually_leaves_toward_neighbor() -> None:
    farm = _farm()
    snap = _snap(screen=FARM_SCREEN, health=0x22)
    for _ in range(4):
        act = farm.step(snap)
        assert act.reason == "farm_wait"
        assert farm.phase is HeartFarmPhase.FARM
    act = farm.step(snap)
    assert act.reason == "farm_leave"
    assert list(act.action) == list(nes_action("LEFT"))
    assert farm.phase is HeartFarmPhase.FARM


def test_neighbor_screen_emits_opposite_farm_respawn() -> None:
    farm = _farm()
    act = farm.step(_snap(screen=NEIGHBOR_SCREEN, health=0x22))
    assert act.reason == "farm_respawn"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert farm.phase is HeartFarmPhase.FARM


def test_hearts_recovered_on_farm_screen_is_done() -> None:
    farm = _farm()
    act = farm.step(_snap(screen=FARM_SCREEN, health=0x33))
    assert farm.success
    assert farm.phase is HeartFarmPhase.DONE
    assert act.reason == "farm_done"


def test_recovered_on_neighbor_stays_farm_then_done_on_return() -> None:
    farm = _farm()
    act = farm.step(_snap(screen=NEIGHBOR_SCREEN, health=0x33))
    assert farm.phase is HeartFarmPhase.FARM
    assert act.reason == "farm_respawn"
    assert list(act.action) == list(nes_action("RIGHT"))
    act = farm.step(_snap(screen=FARM_SCREEN, health=0x33))
    assert farm.phase is HeartFarmPhase.DONE
    assert farm.success


def test_unexpected_screen_fails_even_with_restock() -> None:
    farm = _farm()
    act = farm.step(_snap(screen=0x12, health=0x22))
    assert act.reason == "left_farm_screen"
    assert farm.phase is HeartFarmPhase.FAILED


def test_transition_holds_restock_direction_leaving_farm_screen() -> None:
    farm = _farm()
    snap = _snap(mode=6, screen=FARM_SCREEN, next_screen=NEIGHBOR_SCREEN, health=0x22)
    act = farm.step(snap)
    assert act.reason == "farm_scroll"
    assert list(act.action) == list(nes_action("LEFT"))
    assert farm.phase is HeartFarmPhase.FARM


def test_transition_holds_opposite_direction_returning_to_farm_screen() -> None:
    farm = _farm()
    snap = _snap(mode=6, screen=NEIGHBOR_SCREEN, next_screen=FARM_SCREEN, health=0x22)
    act = farm.step(snap)
    assert act.reason == "farm_scroll"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert farm.phase is HeartFarmPhase.FARM


def test_chases_live_prey_on_farm_screen() -> None:
    prey = ZeldaObject(slot=3, type_id=0x11, x=180, y=149, facing=0, hp=1, state=0)
    farm = _farm()
    act = farm.step(_snap(link_x=120, link_y=149, objects=(prey,)))
    assert "farm_chase" in act.reason
    assert farm.phase is HeartFarmPhase.FARM


def test_walks_onto_rupee_drop_when_no_prey() -> None:
    drop = ZeldaObject(
        slot=2, type_id=RUPEE_DROP_OBJECT_TYPE, x=180, y=149, facing=0, hp=0, state=0
    )
    farm = _farm()
    act = farm.step(_snap(link_x=120, link_y=149, objects=(drop,)))
    assert "farm_rupee" in act.reason
    assert list(act.action) == list(nes_action("RIGHT"))


def test_link_death_fails() -> None:
    farm = _farm()
    act = farm.step(_snap(mode=17, health=0x22))
    assert farm.phase is HeartFarmPhase.FAILED
    assert farm.notes[-1] == "link_death"
    assert act.reason == "link_death"


def test_leaving_overworld_fails() -> None:
    farm = _farm()
    act = farm.step(_snap(level=1, health=0x22))
    assert farm.phase is HeartFarmPhase.FAILED
    assert act.reason == "left_overworld"
