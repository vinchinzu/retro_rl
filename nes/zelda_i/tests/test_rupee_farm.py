"""Unit tests for the generic leftover-safe rupee farm. No emulator."""

from __future__ import annotations

from retro_harness.nes import nes_action

from zelda_i.overworld.rupee_farm import RupeeFarmController, RupeeFarmPhase
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

FARM_SCREEN = 0x4A
NEIGHBOR_SCREEN = 0x49


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
        health=0xFF,
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


def _farm(**overrides) -> RupeeFarmController:
    kwargs = dict(
        target_rupees=80,
        farm_screen=FARM_SCREEN,
        restock_neighbor_screen=NEIGHBOR_SCREEN,
        restock_direction="LEFT",
    )
    kwargs.update(overrides)
    return RupeeFarmController(**kwargs)


def test_never_writes_rupees_only_reads_snapshot() -> None:
    """The controller has no RAM-writing surface: step() only reads snap."""
    farm = _farm()
    snap = _snap(rupees=10)
    before = snap.rupees
    farm.step(snap)
    assert snap.rupees == before  # frozen snapshot; farm cannot have mutated it


def test_already_satisfied_requires_target_and_leftover_screen() -> None:
    farm = _farm()
    assert not farm.already_satisfied(_snap(rupees=79, screen=FARM_SCREEN))
    assert farm.already_satisfied(_snap(rupees=80, screen=FARM_SCREEN))
    assert not farm.already_satisfied(_snap(rupees=80, screen=NEIGHBOR_SCREEN))


def test_chases_nearest_live_prey_object() -> None:
    farm = _farm(swing_period=0)
    prey = ZeldaObject(slot=3, type_id=0x11, x=180, y=149, facing=0, hp=1, state=0)
    snap = _snap(link_x=120, link_y=149, rupees=0, objects=(prey,))
    act = farm.step(snap)
    assert list(act.action) == list(nes_action("RIGHT"))
    assert "farm_chase" in act.reason
    assert farm.phase is RupeeFarmPhase.FARM


def test_ignores_dead_and_sentinel_and_out_of_bounds_slots() -> None:
    farm = _farm()
    objects = (
        ZeldaObject(slot=1, type_id=0x00, x=180, y=149, facing=0, hp=1, state=0),
        ZeldaObject(slot=2, type_id=0xFF, x=180, y=149, facing=0, hp=1, state=0),
        ZeldaObject(slot=3, type_id=0x11, x=180, y=149, facing=0, hp=0, state=0),  # dead
        ZeldaObject(slot=4, type_id=0x11, x=5, y=149, facing=0, hp=1, state=0),  # off-bounds x
        ZeldaObject(slot=0, type_id=0x11, x=180, y=149, facing=0, hp=1, state=0),  # link slot
    )
    snap = _snap(link_x=120, link_y=149, rupees=0, objects=objects)
    act = farm.step(snap)
    # Nothing farmable: empty-frame wait policy kicks in (not a chase).
    assert "farm_wait" in act.reason


def test_prefers_rupee_drop_over_live_prey() -> None:
    farm = _farm(swing_period=0)
    prey = ZeldaObject(slot=1, type_id=0x11, x=60, y=149, facing=0, hp=1, state=0)
    drop = ZeldaObject(slot=2, type_id=0x60, x=180, y=149, facing=0, hp=1, state=0)
    snap = _snap(link_x=120, link_y=149, rupees=0, objects=(prey, drop))
    act = farm.step(snap)
    assert "farm_rupee" in act.reason
    assert list(act.action) == list(nes_action("RIGHT"))  # toward the drop, not the prey


def test_empty_screen_waits_then_leaves_toward_restock_neighbor() -> None:
    farm = _farm(empty_wait_frames=5, swing_period=0)
    snap = _snap(rupees=0)
    for _ in range(4):
        act = farm.step(snap)
        assert act.reason == "farm_wait"
    act = farm.step(snap)
    assert act.reason == "farm_leave"
    assert list(act.action) == list(nes_action("LEFT"))


def test_respawn_toggle_walks_back_from_neighbor_screen() -> None:
    farm = _farm(swing_period=0)
    snap = _snap(screen=NEIGHBOR_SCREEN, rupees=0)
    act = farm.step(snap)
    assert act.reason == "farm_respawn"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_transition_holds_restock_direction_leaving_farm_screen() -> None:
    farm = _farm()
    snap = _snap(mode=6, screen=FARM_SCREEN, next_screen=NEIGHBOR_SCREEN, rupees=0)
    act = farm.step(snap)
    assert act.reason == "farm_scroll"
    assert list(act.action) == list(nes_action("LEFT"))


def test_transition_holds_opposite_direction_returning_to_farm_screen() -> None:
    farm = _farm()
    snap = _snap(mode=6, screen=NEIGHBOR_SCREEN, next_screen=FARM_SCREEN, rupees=0)
    act = farm.step(snap)
    assert act.reason == "farm_scroll"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_finishes_on_farm_screen_when_leftover_is_farm_screen() -> None:
    farm = _farm(leftover_screen=FARM_SCREEN)
    snap = _snap(screen=FARM_SCREEN, rupees=80)
    act = farm.step(snap)
    assert farm.success
    assert farm.phase is RupeeFarmPhase.DONE
    assert farm.report()["glance"] == {
        "screen": FARM_SCREEN,
        "x": snap.link_x,
        "y": snap.link_y,
        "rupees": 80,
        "mode": PLAY_MODE,
    }


def test_returns_to_leftover_screen_before_finishing_when_different() -> None:
    farm = _farm(leftover_screen=NEIGHBOR_SCREEN, swing_period=0)
    # Rupees already hit target while standing on farm_screen: must not
    # finish here — walk back to the neighbor (leftover) screen first.
    on_farm = _snap(screen=FARM_SCREEN, rupees=80)
    act = farm.step(on_farm)
    assert not farm.success
    assert farm.phase is RupeeFarmPhase.RETURN
    assert act.reason == "farm_return"
    assert list(act.action) == list(nes_action("LEFT"))

    on_neighbor = _snap(screen=NEIGHBOR_SCREEN, rupees=80)
    act = farm.step(on_neighbor)
    assert farm.success
    assert farm.phase is RupeeFarmPhase.DONE


def test_unreachable_leftover_screen_fails_closed_not_stranded() -> None:
    farm = _farm(leftover_screen=0x12)  # not farm_screen nor restock_neighbor_screen
    snap = _snap(screen=FARM_SCREEN, rupees=80)
    act = farm.step(snap)
    assert not farm.success
    assert farm.phase is RupeeFarmPhase.FAILED
    assert "farm_return_unreachable" in farm.notes[-1]
    glance = farm.report()["glance"]
    assert set(glance) == {"screen", "x", "y", "rupees", "mode"}


def test_timeout_fails_closed_with_ram_glance_not_a_poke() -> None:
    farm = _farm(max_frames=3)
    snap = _snap(rupees=5)
    for _ in range(3):
        act = farm.step(snap)
    assert farm.phase is RupeeFarmPhase.FAILED
    assert not farm.success
    assert "farm_timeout" in farm.notes[-1]
    glance = farm.report()["glance"]
    assert glance == {
        "screen": FARM_SCREEN,
        "x": snap.link_x,
        "y": snap.link_y,
        "rupees": 5,
        "mode": PLAY_MODE,
    }


def test_link_death_fails_closed() -> None:
    farm = _farm()
    act = farm.step(_snap(mode=17, rupees=0))
    assert farm.phase is RupeeFarmPhase.FAILED
    assert farm.notes[-1] == "link_death"


def test_leaving_overworld_fails_closed() -> None:
    farm = _farm()
    act = farm.step(_snap(level=1, rupees=0))
    assert farm.phase is RupeeFarmPhase.FAILED
    assert "farm_left_level_1" in farm.notes[-1]


def test_left_farm_screen_unexpectedly_fails_closed() -> None:
    farm = _farm()
    act = farm.step(_snap(screen=0x12, rupees=0))
    assert farm.phase is RupeeFarmPhase.FAILED
    assert "farm_left_12" in farm.notes[-1]


def test_leftover_screen_defaults_to_farm_screen() -> None:
    farm = _farm()
    assert farm.leftover_screen == FARM_SCREEN
