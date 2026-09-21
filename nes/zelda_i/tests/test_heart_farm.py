"""Unit tests for overworld heart-farm restock. No emulator."""

from __future__ import annotations

from retro_harness.nes import nes_action

from zelda_i import combat as _combat
from zelda_i.dungeon import ids as _ids
from zelda_i.dungeon.behaviors import GORIYA_BOOMERANG_TYPE
from zelda_i.dungeon.ids import HEART_DROP_STATE, RUPEE_DROP_OBJECT_TYPE, RUPEE_DROP_STATE
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


def _heart_drop_type() -> int:
    """Prefer collector/id constants; else a non-rupee hp==0 dummy the fallback accepts."""
    typed = getattr(_ids, "HEART_DROP_OBJECT_TYPE", None)
    if typed is None:
        typed = getattr(_combat, "HEART_DROP_OBJECT_TYPE", None)
    if typed is not None:
        return int(typed)
    types = getattr(_combat, "HEART_OR_FAIRY_TYPES", None)
    if types:
        return int(next(iter(types)))
    return 0x61


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
    assert act.reason.startswith("farm_leave")
    assert farm.leaving
    assert farm.phase is HeartFarmPhase.FARM


def test_leave_is_latched_and_reaches_the_west_edge() -> None:
    """One-frame farm_leave never scrolled: the wait oscillation undid it."""
    farm = _farm()
    for _ in range(5):
        farm.step(_snap(screen=FARM_SCREEN, health=0x22))
    assert farm.leaving
    reasons = []
    for x in range(120, 0, -4):  # walk Link west the way the emulator would
        act = farm.step(_snap(screen=FARM_SCREEN, health=0x22, link_x=x, link_y=141))
        reasons.append(act.reason)
    # Never hands back to the wait oscillation, and never bounces off the
    # x<36 enter guard, which is exactly where the scroll has to happen.
    assert "farm_wait" not in reasons
    assert "farm_enter" not in reasons
    assert "farm_leave_push" in reasons
    assert farm.leaving


def _run_one_empty_restock(farm: HeartFarmController, health: int):
    """Wait -> leave -> neighbor -> back on an empty screen -> next wait."""
    for _ in range(5):
        farm.step(_snap(screen=FARM_SCREEN, health=health))
    assert farm.leaving
    act = farm.step(_snap(screen=NEIGHBOR_SCREEN, health=health))
    assert act.reason == "farm_respawn"
    assert farm.restocks == 1
    assert not farm.leaving
    for _ in range(farm.empty_wait_frames - 1):
        act = farm.step(_snap(screen=FARM_SCREEN, health=health))
        assert act.reason == "farm_wait"
    return farm.step(_snap(screen=FARM_SCREEN, health=health))


def test_restock_that_returns_an_empty_screen_is_dead() -> None:
    """0x4A does not repopulate: depth-1 and depth-2 round trips both empty."""
    farm = _farm()
    act = _run_one_empty_restock(farm, 0x21)  # 1 filled: below the soft floor
    assert act.reason == "farm_screen_dead"
    assert farm.phase is HeartFarmPhase.FAILED
    assert not farm.success
    assert any(n.startswith("farm_screen_dead") for n in farm.notes)


def test_dead_screen_keeps_the_timeout_soft_ok_policy() -> None:
    farm = _farm()
    act = _run_one_empty_restock(farm, 0x22)  # 2 filled, not the 3 asked for
    assert act.reason == "farm_done"
    assert farm.phase is HeartFarmPhase.DONE
    assert farm.success
    assert any(n.startswith("farm_screen_dead") for n in farm.notes)
    assert "farm_soft_ok" in farm.notes


def test_prey_after_a_restock_keeps_farming() -> None:
    """A screen that does repopulate must not be called dead."""
    farm = _farm()
    for _ in range(5):
        farm.step(_snap(screen=FARM_SCREEN, health=0x22))
    farm.step(_snap(screen=NEIGHBOR_SCREEN, health=0x22))
    assert farm.restocks == 1
    live = _snap(
        screen=FARM_SCREEN,
        health=0x22,
        objects=(ZeldaObject(slot=3, type_id=0x11, x=180, y=141, facing=0, hp=1, state=0),),
    )
    act = farm.step(live)
    assert act.reason.startswith("farm_chase")
    assert farm.saw_prey
    for _ in range(farm.empty_wait_frames + 2):
        act = farm.step(_snap(screen=FARM_SCREEN, health=0x22))
    assert act.reason != "farm_screen_dead"
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
        slot=2, type_id=RUPEE_DROP_OBJECT_TYPE, x=180, y=149, facing=0, hp=0, state=RUPEE_DROP_STATE
    )
    farm = _farm()
    act = farm.step(_snap(link_x=120, link_y=149, objects=(drop,)))
    assert "farm_rupee" in act.reason
    assert list(act.action) == list(nes_action("RIGHT"))


def test_walks_onto_heart_drop_when_no_enemies() -> None:
    heart = ZeldaObject(
        slot=2, type_id=_heart_drop_type(), x=180, y=149, facing=0, hp=0, state=HEART_DROP_STATE
    )
    farm = _farm()
    act = farm.step(_snap(link_x=120, link_y=149, objects=(heart,)))
    assert "farm_heart" in act.reason
    assert "farm_wait" not in act.reason
    assert act.reason != "farm"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert farm.phase is HeartFarmPhase.FARM


def test_prefers_heart_drop_over_rupee() -> None:
    heart = ZeldaObject(
        slot=2, type_id=_heart_drop_type(), x=180, y=149, facing=0, hp=0, state=HEART_DROP_STATE
    )
    rupee = ZeldaObject(
        slot=3, type_id=RUPEE_DROP_OBJECT_TYPE, x=80, y=149, facing=0, hp=0, state=RUPEE_DROP_STATE
    )
    farm = _farm()
    act = farm.step(_snap(link_x=120, link_y=149, objects=(rupee, heart)))
    assert "farm_heart" in act.reason
    assert "farm_rupee" not in act.reason
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


def test_min_filled_zero_is_inert_farm_below_hearts_analog() -> None:
    """path.py farm_below_hearts=0 analog: min_filled<=0 skips the farm."""
    farm = HeartFarmController(min_filled=0, farm_screen=NO_RESTOCK_SCREEN)
    snap = _snap(screen=NO_RESTOCK_SCREEN, health=0x00)
    assert farm.already_satisfied(snap) is False
    act = farm.step(snap)
    assert farm.success
    assert farm.phase is HeartFarmPhase.DONE
    assert act.reason == "farm_done"
    assert farm.notes[-1] == "farm_skipped"


def test_already_satisfied_when_filled_meets_min() -> None:
    farm = _farm(min_filled=3)
    assert farm.already_satisfied(_snap(health=0x33))
    assert not farm.already_satisfied(_snap(health=0x22))


def test_occupancy_miss_blocks_cell_and_replans() -> None:
    prey = ZeldaObject(slot=3, type_id=0x11, x=180, y=149, facing=0, hp=1, state=0)
    farm = _farm()
    snap = _snap(link_x=120, link_y=149, objects=(prey,))
    act = farm.step(snap)
    assert "farm_chase" in act.reason
    assert farm._walker is not None
    assert farm._walker.last_dir == "RIGHT"
    act = farm.step(snap)
    assert farm._walker.misses >= 1
    assert (121, 149) in farm._walker.grid.blocked
    assert farm.phase is HeartFarmPhase.FARM
    assert act.reason != "farm_unstick"


def test_occupancy_no_path_stands() -> None:
    prey = ZeldaObject(slot=3, type_id=0x11, x=120, y=110, facing=0, hp=1, state=0)
    farm = _farm()
    snap = _snap(link_x=120, link_y=130, objects=(prey,))
    farm.step(snap)
    walker = farm._walker
    assert walker is not None
    grid = walker.grid
    grid.xmin, grid.xmax, grid.ymin, grid.ymax = 100, 140, 100, 140
    for x in range(100, 141):
        grid.blocked.add((x, 120))
        grid.inferred.add((x, 120))
    walker.path = None
    act = farm.step(snap)
    assert act.reason == "occupancy_stand"
    assert farm.phase is HeartFarmPhase.FARM
    assert not farm.success
    # Inferred blocks are preserved without clearing
    assert (120, 120) in walker.grid.inferred
    assert walker.forgets == 0


def test_occupancy_frozen_chase_stands_without_clearing_inferred() -> None:
    """Issue 2: repeated misses box Link in; controller stands without forgetting."""
    prey = ZeldaObject(slot=3, type_id=0x11, x=120, y=110, facing=0, hp=1, state=0)
    farm = _farm()
    snap = _snap(link_x=120, link_y=130, objects=(prey,))
    # Step repeatedly with Link frozen at (120, 130).
    # Each miss blocks an adjacent cell into inferred.
    # Once all 4 neighbors are blocked, shortest_path is None and Link stands.
    reasons = []
    for _ in range(10):
        act = farm.step(snap)
        reasons.append(act.reason)
    walker = farm._walker
    assert walker is not None
    assert "occupancy_stand" in reasons
    assert reasons[-1] == "occupancy_stand"
    assert walker.forgets == 0
    # Inferred blocks remain intact
    assert (120, 129) in walker.grid.inferred
    assert (119, 130) in walker.grid.inferred
    assert (121, 130) in walker.grid.inferred
    assert (120, 131) in walker.grid.inferred


def test_occupancy_dodge_does_not_fence_cardinal() -> None:
    """Issue 3: projectile dodge overrides cardinal; last_dir reflects dodge."""
    farm = _farm()
    prey = ZeldaObject(slot=3, type_id=0x11, x=180, y=149, facing=0, hp=1, state=0)
    # Goriya boomerang at (140, 149) moving toward Link triggers dodge
    shot = ZeldaObject(
        slot=4, type_id=GORIYA_BOOMERANG_TYPE, x=140, y=149, facing=0, hp=0, state=0
    )
    snap = _snap(link_x=120, link_y=149, objects=(prey, shot))
    act = farm.step(snap)
    assert "_dodge" in act.reason
    walker = farm._walker
    assert walker is not None
    assert walker.last_dir in ("UP", "DOWN")
    assert walker.last_dir != "RIGHT"
    # On the next frame, if Link is still at (120, 149), the blocked cell is in
    # the dodge direction, not the overridden cardinal (RIGHT / (121, 149)).
    farm.step(snap)
    assert (121, 149) not in walker.grid.blocked


# --- phantom bomb slots (ObjState 0x00 == a cleared object slot) ----------


def _phantom_slot(x: int = 144, y: int = 149) -> ZeldaObject:
    """An emptied drop slot: ObjType still 0x60, ObjState cleared to 0x00.

    Bomb is ROM item code 0x00, so this is byte-for-byte what a live bomb drop
    looks like. Only "can Link bank a bomb at all" separates them.
    """
    return ZeldaObject(
        slot=3,
        type_id=_ids.BOMB_DROP_OBJECT_TYPE,
        x=x,
        y=y,
        facing=0,
        hp=0,
        state=_ids.BOMB_DROP_STATE,
    )


def test_phantom_empty_slot_is_not_a_bomb_drop_pre_l1() -> None:
    """bombs=0 means no bomb item: a 0x60/state-0x00 slot is an empty slot."""
    farm = _farm(empty_wait_frames=5)
    snap = _snap(bombs=0, objects=(_phantom_slot(),))
    reasons = [farm.step(snap).reason for _ in range(5)]
    assert "farm_bomb" not in reasons
    # The empty ladder keeps its schedule: 4 waits, then the restock leave.
    assert reasons[:4] == ["farm_wait"] * 4
    assert reasons[4] == "farm_leave"
    assert farm.leaving


def test_phantom_slot_cannot_reset_empty_frames() -> None:
    """The pin that burned the whole budget: empty_frames must still climb."""
    farm = _farm(empty_wait_frames=90)
    snap = _snap(bombs=0, objects=(_phantom_slot(),))
    for _ in range(10):
        farm.step(snap)
    assert farm.empty_frames == 10


def test_bomb_drop_is_scooped_once_link_owns_bombs() -> None:
    """The feature survives the gate: with bombs banked, a bomb drop scoops."""
    farm = _farm()
    snap = _snap(bombs=1, objects=(_phantom_slot(x=144),))
    act = farm.step(snap)
    assert act.reason == "farm_bomb"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_bomb_chase_gives_the_empty_ladder_back_after_its_budget() -> None:
    """An unresolving bomb slot yields; it does not hold the farm open."""
    budget = 12
    farm = _farm(empty_wait_frames=5, bomb_chase_max_frames=budget)
    snap = _snap(bombs=1, objects=(_phantom_slot(),))
    reasons = [farm.step(snap).reason for _ in range(budget + 10)]
    assert reasons[0] == "farm_bomb"
    assert farm.bomb_chase > budget
    assert "farm_wait" in reasons[budget:]
    # The restock ladder owns the farm again (a frozen Link boxes his own
    # walker in, so the leave walk itself may be standing).
    assert farm.leaving


# --- blocked goal: a bumped drop cell is walked to, not stood on ----------


def test_bump_blocked_drop_cell_is_retargeted_not_stood_on() -> None:
    """shortest_path refuses an in-bounds blocked goal; standing is not the answer.

    The farm grid has no measured walls, so any block on it is one inferred
    bump. If that bump lands on the drop's own cell, the pre-fix walk stood in
    ``occupancy_stand`` until ``farm_screen_dead`` — on a drop Link could have
    walked right up to (pickup is contact, not pixel-exact).
    """
    drop = ZeldaObject(
        slot=3,
        type_id=_heart_drop_type(),
        x=144,
        y=149,
        facing=0,
        hp=0,
        state=HEART_DROP_STATE,
    )
    farm = _farm()
    snap = _snap(link_x=120, link_y=149, objects=(drop,))
    act = farm.step(snap)
    assert act.reason in ("farm_heart", "farm_fairy")
    walker = farm._walker
    assert walker is not None
    walker.grid.blocked.add((144, 149))
    walker.grid.inferred.add((144, 149))
    walker.path = None
    act = farm.step(_snap(link_x=121, link_y=149, objects=(drop,)))
    assert act.reason != "occupancy_stand"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert walker.retargets >= 1
    # The forget path is NOT taken: an empty overworld grid just re-learns the
    # same wall (measured_walker docstring), so inferred blocks stay put.
    assert walker.forgets == 0
    assert (144, 149) in walker.grid.inferred
    assert farm.report()["occupancy_retargets"] >= 1
