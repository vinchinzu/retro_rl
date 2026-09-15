"""Unit tests for the generic leftover-safe rupee farm. No emulator."""

from __future__ import annotations

from retro_harness.nes import nes_action

from zelda_i.dungeon.ids import RUPEE_DROP_STATE
from zelda_i.overworld.heart_farm import RESTOCK_LANE_Y as LANE_Y
from zelda_i.overworld.rupee_farm import (
    MAX_LEAVE_STALLS,
    RupeeFarmController,
    RupeeFarmPhase,
)
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
    drop = ZeldaObject(
        slot=2, type_id=0x60, x=180, y=149, facing=0, hp=1, state=RUPEE_DROP_STATE
    )
    snap = _snap(link_x=120, link_y=149, rupees=0, objects=(prey, drop))
    act = farm.step(snap)
    assert "farm_rupee" in act.reason
    assert list(act.action) == list(nes_action("RIGHT"))  # toward the drop, not the prey


def test_empty_screen_waits_then_leaves_toward_restock_neighbor() -> None:
    """On the lane, the leave is the plain restock direction."""
    farm = _farm(empty_wait_frames=5, swing_period=0)
    snap = _snap(rupees=0, link_y=LANE_Y)
    for _ in range(4):
        act = farm.step(snap)
        assert act.reason == "farm_wait"
    act = farm.step(snap)
    assert act.reason == "farm_leave"
    assert list(act.action) == list(nes_action("LEFT"))


def test_leave_walks_to_the_open_lane_before_pushing_the_edge() -> None:
    """Off the lane, the leave aims at the lane cell, not blindly at the edge."""
    farm = _farm(empty_wait_frames=1, swing_period=0)
    act = farm.step(_snap(rupees=0, link_x=120, link_y=LANE_Y + 8))
    assert act.reason == "farm_leave"
    assert list(act.action) == list(nes_action("UP"))


def test_leave_at_the_edge_cell_holds_the_direction_for_the_scroll() -> None:
    farm = _farm(empty_wait_frames=1, swing_period=0)
    act = farm.step(_snap(rupees=0, link_x=6, link_y=LANE_Y))
    assert act.reason == "farm_leave_push"
    assert list(act.action) == list(nes_action("LEFT"))


def test_blocked_leave_routes_around_instead_of_pressing_into_the_bush() -> None:
    """A rock between Link and the edge must not cost the whole max_frames."""
    farm = _farm(empty_wait_frames=1, swing_period=0)
    snap = _snap(rupees=0, link_x=120, link_y=LANE_Y)
    first = farm.step(snap)
    assert list(first.action) == list(nes_action("LEFT"))
    # Link does not move: the walker grades a miss, blocks the cell ahead and
    # replans onto another cardinal rather than holding LEFT forever.
    actions = [farm.step(snap) for _ in range(3)]
    assert farm._occ.misses >= 1
    assert (119, LANE_Y) in farm._occ.walker.grid.blocked
    assert any(list(a.action) != list(nes_action("LEFT")) for a in actions)
    assert all(a.reason.startswith("farm_leave") for a in actions)


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
    on_farm = _snap(screen=FARM_SCREEN, rupees=80, link_y=LANE_Y)
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


def test_chase_nearby_prey_slashes_only_on_hitbox_not_blind_cadence() -> None:
    farm = _farm()
    nearby = ZeldaObject(slot=1, type_id=0x07, x=135, y=149, facing=0, hp=1, state=0)
    snap = _snap(link_x=120, link_y=149, rupees=0, objects=(nearby,))
    act = farm.step(snap)
    assert "farm_chase" in act.reason
    assert act.reason == "farm_chase_slash"
    assert list(act.action) == list(nes_action("RIGHT", "A"))

    farm.reset()
    far = ZeldaObject(slot=1, type_id=0x07, x=200, y=149, facing=0, hp=1, state=0)
    snap = _snap(link_x=120, link_y=149, rupees=0, objects=(far,))
    act = farm.step(snap)
    assert act.reason == "farm_chase"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_chases_enemies_in_upper_corridor_below_y120() -> None:
    """Screen enemies on 0x78 (y=90..109) and 0x4A (y=93..110) are hunted."""
    farm = _farm(swing_period=0)
    octorok = ZeldaObject(slot=1, type_id=0x07, x=120, y=95, facing=0, hp=1, state=0)
    snap = _snap(link_x=120, link_y=149, rupees=0, objects=(octorok,))
    act = farm.step(snap)
    assert "farm_chase" in act.reason
    assert list(act.action) == list(nes_action("UP"))


def test_ignores_slot11_cave_door_trigger() -> None:
    """Slot 11 type 0x64 hp 240 is the cave mouth trigger, never prey."""
    farm = _farm()
    cave_trigger = ZeldaObject(
        slot=11, type_id=0x64, x=176, y=77, facing=0, hp=240, state=0
    )
    snap = _snap(link_x=120, link_y=149, rupees=0, objects=(cave_trigger,))
    act = farm.step(snap)
    assert "farm_wait" in act.reason  # cave trigger ignored; wait/leave kicks in


def test_farm_leave_persists_across_multiple_frames() -> None:
    """farm_leave must not reset empty_frames to 0 after a single frame."""
    farm = _farm(empty_wait_frames=5, swing_period=0)
    for _ in range(4):
        act = farm.step(_snap(rupees=0, link_y=LANE_Y))
        assert act.reason == "farm_wait"
    # Frame 5 onward: leaving mode holds; Link walking the lane keeps LEFT.
    x = 120
    for _ in range(3):
        act = farm.step(_snap(rupees=0, link_x=x, link_y=LANE_Y))
        assert act.reason == "farm_leave"
        assert list(act.action) == list(nes_action("LEFT"))
        x -= 1


# --- give-up: a dead corridor must not cost 36,000 frames ------------------


def _drain_to_leave(farm: RupeeFarmController, wait: int) -> None:
    """Run the empty ladder on the farm screen until the leave fires."""
    for _ in range(wait):
        farm.step(_snap(rupees=0, link_y=LANE_Y))


def test_dead_corridor_gives_up_after_one_restock() -> None:
    """Overworld waves are one-shot; a depth-1 restock is a no-op on 0x4A/0x49.

    ``HeartFarmController`` already treats the restock as a give-up detector.
    Without it the rupee farm cycled wait -> leave -> respawn for all 36,000
    frames on a screen that can never restock.
    """
    farm = _farm(empty_wait_frames=5, swing_period=0, max_frames=36000)
    _drain_to_leave(farm, 5)
    assert farm.leaving and farm.leave_pending
    # Scroll out: arriving on the neighbor is the restock.
    farm.step(_snap(screen=NEIGHBOR_SCREEN, rupees=0, link_y=LANE_Y))
    assert farm.restocks == 1
    assert "farm_restock_1" in farm.notes
    # Back on the farm screen, still nothing to farm.
    for _ in range(4):
        act = farm.step(_snap(rupees=0, link_y=LANE_Y))
        assert act.reason == "farm_wait"
    act = farm.step(_snap(rupees=0, link_y=LANE_Y))
    assert act.reason.startswith("farm_screen_dead")
    assert farm.phase is RupeeFarmPhase.FAILED
    assert not farm.success
    assert "farm_screen_dead" in farm.notes[-1]
    assert farm.frames < 100  # not 36,000
    assert set(farm.report()["glance"]) == {"screen", "x", "y", "rupees", "mode"}


def test_prey_seen_after_a_restock_keeps_farming() -> None:
    """The give-up is 'the restock returned nothing', not 'one restock happened'."""
    farm = _farm(empty_wait_frames=5, swing_period=0)
    _drain_to_leave(farm, 5)
    farm.step(_snap(screen=NEIGHBOR_SCREEN, rupees=0, link_y=LANE_Y))
    assert farm.restocks == 1
    prey = ZeldaObject(slot=3, type_id=0x11, x=180, y=LANE_Y, facing=0, hp=1, state=0)
    act = farm.step(_snap(rupees=0, link_y=LANE_Y, objects=(prey,)))
    assert "farm_chase" in act.reason
    assert farm.saw_prey
    # Wave dies again: one more restock cycle is allowed, not a give-up.
    for _ in range(4):
        assert farm.step(_snap(rupees=0, link_y=LANE_Y)).reason == "farm_wait"
    assert farm.step(_snap(rupees=0, link_y=LANE_Y)).reason.startswith("farm_leave")
    assert farm.phase is RupeeFarmPhase.FARM


def test_walled_leave_fails_closed_after_repeated_stalls() -> None:
    """A leave that never reaches the edge is a walled exit, not bad luck."""
    farm = _farm(empty_wait_frames=1, swing_period=0, leave_max_frames=3)
    snap = _snap(rupees=0, link_x=120, link_y=LANE_Y)
    reasons = [farm.step(snap).reason for _ in range(40)]
    assert farm.phase is RupeeFarmPhase.FAILED
    assert "farm_leave_blocked" in farm.notes[-1]
    assert any("farm_leave_stalled" in n for n in farm.notes)
    assert farm.leave_stalls >= MAX_LEAVE_STALLS
    assert len(reasons) == 40

