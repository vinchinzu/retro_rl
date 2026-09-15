"""On-route screen clearing: prey selection, the kill census, the budgets.

No emulator. The census is the part worth pinning: ``kills`` is inferred from
slot transitions and ``kills_counter`` from two ROM counters that
``Link_BeHarmed`` (collision, ``$04F0``) clears, so both have a way to lie
and both are asserted here.
"""

from __future__ import annotations

from zelda_i.dungeon.behaviors import ROCK_PROJECTILE_TYPE
from zelda_i.dungeon.ids import (
    BOMB_DROP_STATE,
    FIVE_RUPEE_DROP_STATE,
    HEART_DROP_STATE,
    RUPEE_DROP_OBJECT_TYPE,
    RUPEE_DROP_STATE,
)
from zelda_i.overworld.hunt import (
    CAVE_TRIGGER_TYPE,
    HUNT_BOX,
    ScreenHunter,
    hunt_prey,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

OCTOROK = 0x38


def _snap(**kwargs) -> ZeldaSnapshot:
    fields = dict(
        mode=PLAY_MODE,
        level=0,
        screen=0x78,
        next_screen=0x78,
        link_x=120,
        link_y=141,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=0x22,  # 3 containers, 2 whole hearts
        heart_partial=0xFF,
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


def _foe(slot: int = 1, x: int = 160, y: int = 141, hp: int = 2, type_id: int = OCTOROK):
    return ZeldaObject(slot=slot, type_id=type_id, x=x, y=y, facing=0, hp=hp, state=0)


def _drop(slot: int = 2, x: int = 80, y: int = 141, state: int = RUPEE_DROP_STATE):
    return ZeldaObject(
        slot=slot, type_id=RUPEE_DROP_OBJECT_TYPE, x=x, y=y, facing=0, hp=0, state=state
    )


# ---------------------------------------------------------------- prey ---


def test_prey_is_in_box_killable_and_not_a_trigger() -> None:
    xlo, xhi, ylo, yhi = HUNT_BOX
    prey = hunt_prey(
        _snap(
            objects=(
                _foe(slot=1, x=160, y=141),
                _foe(slot=2, x=xhi + 4, y=141),  # past the east scroll margin
                _foe(slot=3, x=160, y=ylo - 4),  # above the north margin
                _foe(slot=4, x=160, y=141, type_id=CAVE_TRIGGER_TYPE, hp=240),
                _foe(slot=5, x=160, y=141, hp=0),  # corpse
            )
        )
    )
    assert [int(o.slot) for o in prey] == [1]


def test_drops_are_never_prey() -> None:
    assert hunt_prey(_snap(objects=(_drop(),))) == ()


# -------------------------------------------------------------- census ---


def test_a_vanished_enemy_slot_on_the_same_screen_is_a_kill() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(objects=(_foe(slot=1), _foe(slot=2, x=96))))
    hunter.observe(_snap(objects=(_foe(slot=1), _foe(slot=2, x=96))))
    assert hunter.kills == 0
    hunter.observe(_snap(objects=(_foe(slot=1),)))
    assert hunter.kills == 1


def test_a_slot_that_becomes_its_own_drop_is_a_kill() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(objects=(_foe(slot=1),)))
    hunter.observe(_snap(objects=(_drop(slot=1),)))
    assert hunter.kills == 1


def test_an_octorok_rock_is_not_a_kill() -> None:
    """A rock holds a slot with hp and then vanishes against a wall. That is
    what put the census 4 ahead of the ROM counters on the first live walk."""
    hunter = ScreenHunter()
    rock = _foe(slot=3, x=100, y=141, hp=1, type_id=ROCK_PROJECTILE_TYPE)
    hunter.observe(_snap(objects=(_foe(slot=1), rock)))
    hunter.observe(_snap(objects=(_foe(slot=1),)))
    assert hunter.kills == 0


def test_rupees_the_hunt_banks_are_counted() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(rupees=0))
    hunter.observe(_snap(rupees=5))
    hunter.observe(_snap(rupees=5))
    assert hunter.rupees_banked == 5
    hunter.observe(_snap(rupees=0))  # spent at the counter, not un-earned
    assert hunter.rupees_banked == 5


def test_a_screen_change_is_not_a_wave_of_kills() -> None:
    """Every slot is reused by the next screen; none of them died."""
    hunter = ScreenHunter()
    hunter.observe(_snap(screen=0x78, objects=(_foe(slot=1), _foe(slot=2, x=96))))
    hunter.observe(_snap(screen=0x68, objects=()))
    hunter.observe(_snap(screen=0x68, objects=()))
    assert hunter.kills == 0


def test_counter_census_banks_gains_and_ignores_a_hit_reset() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(world_kill_count=0, help_drop_count=0))
    hunter.observe(_snap(world_kill_count=1, help_drop_count=1))
    hunter.observe(_snap(world_kill_count=2, help_drop_count=2))
    assert hunter.kills_counter == 2
    hunter.observe(_snap(world_kill_count=0, help_drop_count=0))  # Link_BeHarmed
    assert hunter.kills_counter == 2
    hunter.observe(_snap(world_kill_count=1, help_drop_count=1))
    assert hunter.kills_counter == 3


def test_the_larger_counter_delta_is_the_one_banked() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(world_kill_count=15, help_drop_count=5))
    hunter.observe(_snap(world_kill_count=0, help_drop_count=6))  # 16th kill
    assert hunter.kills_counter == 1


# ---------------------------------------------------------------- step ---


def test_a_drop_outranks_a_body() -> None:
    hunter = ScreenHunter()
    snap = _snap(objects=(_foe(slot=1, x=160), _drop(slot=2, x=80)))
    act = hunter.step(snap, 1)
    assert act is not None and act.reason == "hunt_drop"


def test_five_rupee_drop_is_taken_at_full_health() -> None:
    hunter = ScreenHunter()
    snap = _snap(health=0x22, objects=(_drop(state=FIVE_RUPEE_DROP_STATE),))
    act = hunter.step(snap, 1)
    assert act is not None and act.reason == "hunt_drop"


def test_a_heart_is_left_on_the_floor_at_full_hearts() -> None:
    """Full is nibble-equal plus a full partial, not ``filled < containers``."""
    hunter = ScreenHunter()
    snap = _snap(health=0x22, heart_partial=0xFF, objects=(_drop(state=HEART_DROP_STATE),))
    assert hunter.step(snap, 1) is None


def test_a_heart_is_taken_once_the_partial_heart_is_chipped() -> None:
    hunter = ScreenHunter()
    snap = _snap(health=0x22, heart_partial=0x40, objects=(_drop(state=HEART_DROP_STATE),))
    act = hunter.step(snap, 1)
    assert act is not None and act.reason == "hunt_drop"


def test_a_bomb_drop_is_not_chased() -> None:
    """Bomb is item code 0x00 — the same ObjState a cleared slot reads as."""
    hunter = ScreenHunter()
    snap = _snap(objects=(_drop(state=BOMB_DROP_STATE),))
    assert hunter.step(snap, 1) is None


def test_an_empty_screen_retires_after_the_settle_and_spawn_windows() -> None:
    hunter = ScreenHunter(settle_frames=3, spawn_wait_frames=6)
    snap = _snap()
    for frame in range(1, 6):
        assert hunter.step(snap, frame) is None
    assert hunter.done == set()  # settle is up, the spawn window is not
    assert hunter.step(snap, 6) is None
    assert 0x78 in hunter.done
    assert hunter.screens_cleared == 1


def test_a_late_wave_is_still_hunted() -> None:
    """A screen is not written off on an empty first look: a dungeon settle
    spawn runs 80-100f and nothing says the overworld is faster."""
    hunter = ScreenHunter(settle_frames=1, spawn_wait_frames=50)
    for frame in range(1, 10):
        assert hunter.step(_snap(), frame) is None
    act = hunter.step(_snap(objects=(_foe(),)), 10)
    assert act is not None and act.reason == "hunt_78"


def test_a_screen_left_before_the_spawn_window_is_not_written_off() -> None:
    hunter = ScreenHunter(settle_frames=1, spawn_wait_frames=50)
    for frame in range(1, 10):
        hunter.step(_snap(screen=0x78), frame)
    hunter.step(_snap(screen=0x68), 10)
    assert hunter.done == set()


def test_a_retired_screen_is_never_hunted_again() -> None:
    hunter = ScreenHunter(settle_frames=1, spawn_wait_frames=1)
    hunter.step(_snap(), 1)
    assert 0x78 in hunter.done
    assert hunter.step(_snap(objects=(_foe(),)), 2) is None


def test_the_screen_budget_ends_a_hunt_that_will_not_finish() -> None:
    hunter = ScreenHunter(screen_max_frames=5)
    snap = _snap(objects=(_foe(),))
    for frame in range(1, 6):
        assert hunter.step(snap, frame) is not None
    assert hunter.step(snap, 6) is None
    assert hunter.screens_retired == 1
    assert any(note.startswith("hunt_budget") for note in hunter.notes)


def test_a_body_that_will_not_die_is_skipped_not_chased_to_the_budget() -> None:
    hunter = ScreenHunter(target_max_frames=3)
    stubborn = _foe(slot=1, x=200, y=141)
    reachable = _foe(slot=2, x=140, y=141)
    snap = _snap(link_x=120, objects=(stubborn, reachable))
    # Nearest first, so slot 2 is held; drive it past its budget.
    for frame in range(1, 5):
        hunter.step(snap, frame)
    assert 2 in hunter.skipped
    assert hunter.target_slot == 1


def test_the_last_heart_is_not_traded_for_a_rupee() -> None:
    hunter = ScreenHunter()
    snap = _snap(health=0x21, objects=(_foe(),))  # 1 of 3 filled
    assert hunter.step(snap, 1) is None
    assert any(note.startswith("hunt_hurt") for note in hunter.notes)


def test_hunt_yields_off_the_overworld() -> None:
    hunter = ScreenHunter()
    assert hunter.step(_snap(level=1, objects=(_foe(),)), 1) is None
    assert hunter.step(_snap(mode=11, objects=(_foe(),)), 2) is None


def test_report_carries_both_censuses() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(objects=(_foe(slot=1),)))
    hunter.observe(_snap(world_kill_count=1, help_drop_count=1, objects=()))
    rep = hunter.report()
    assert rep["kills"] == 1
    assert rep["kills_counter"] == 1


# --------------------------------------------------------- lane return ---


def test_the_hunt_walks_back_to_the_hop_lane_it_left() -> None:
    """The live wedge: after a fight on 0x49 Link stood at (56,125) holding
    DOWN against a bush, and the stage burned 27,000f on ``unstick_wait``."""
    hunter = ScreenHunter(settle_frames=1, spawn_wait_frames=1)
    hunter.step(_snap(link_x=56, link_y=125), 1)  # clears the empty screen
    assert 0x78 in hunter.done
    hunter.screen_frames = 30  # it did fight here
    act = hunter.step(_snap(link_x=56, link_y=125), 2, lane=("y", 141))
    assert act is not None and act.reason == "hunt_lane"


def test_a_screen_the_hunt_never_fought_on_is_left_to_the_hop() -> None:
    hunter = ScreenHunter(settle_frames=1, spawn_wait_frames=1)
    hunter.step(_snap(link_x=56, link_y=125), 1)
    assert hunter.step(_snap(link_x=56, link_y=125), 2, lane=("y", 141)) is None


def test_the_lane_return_stops_once_link_is_on_the_lane() -> None:
    hunter = ScreenHunter(settle_frames=1, spawn_wait_frames=1)
    hunter.step(_snap(link_y=141), 1)
    hunter.screen_frames = 30
    assert hunter.step(_snap(link_y=141), 2, lane=("y", 141)) is None


def test_the_lane_return_gives_up_on_its_budget() -> None:
    hunter = ScreenHunter(settle_frames=1, spawn_wait_frames=1, lane_max_frames=3)
    hunter.step(_snap(link_x=56, link_y=125), 1)
    hunter.screen_frames = 30
    snap = _snap(link_x=56, link_y=125)
    for frame in range(2, 5):
        assert hunter.step(snap, frame, lane=("y", 141)) is not None
    assert hunter.step(snap, 5, lane=("y", 141)) is None


# ------------------------------------------------------- damage census ---


def test_a_half_heart_hit_is_counted_even_though_hits_taken_cannot_see_it() -> None:
    """``$066F`` does not move for a half-heart; ``$0670`` does, so a run can
    report 0 ``hits_taken`` and still be bleeding."""
    hunter = ScreenHunter()
    hunter.observe(_snap(health=0x22, heart_partial=0xFF))
    hunter.observe(_snap(health=0x22, heart_partial=0x7F))
    assert hunter.damage_taken == 1
    hunter.observe(_snap(health=0x22, heart_partial=0xFF))  # healed
    assert hunter.damage_taken == 1


def test_a_streak_reset_short_of_the_forced_drop_is_recorded() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(world_kill_count=3, help_drop_count=3))
    hunter.observe(_snap(world_kill_count=4, help_drop_count=4))
    hunter.observe(_snap(world_kill_count=0, help_drop_count=0))
    assert hunter.streak_resets == 1
    assert hunter.streak_best == 4


def test_a_collision_is_a_hurt_even_when_assist_has_already_refilled() -> None:
    """``Link_BeHarmed`` grants $04F0=24 and zeros $50/$627. Survival assist
    writes $0670 back to $FF before the next observe, so damage_taken stays
    0; iframes are what still fire."""
    hunter = ScreenHunter()
    hunter.observe(_snap(health=0x22, heart_partial=0xFF, link_iframes=0))
    hunter.observe(
        _snap(
            health=0x22,
            heart_partial=0xFF,
            link_iframes=24,
            world_kill_count=0,
            help_drop_count=0,
        )
    )
    assert hunter.hurt_events == 1
    assert hunter.damage_taken == 0


# ----------------------------------------------- wiring into the hop path ---


def _walker(hunt: bool):
    """A bomb-shop walk parked on its last hop (0x49 --RIGHT y=141--> 0x4A)."""
    from zelda_i.overworld.gathering import ShopP7WalkController

    ctl = ShopP7WalkController(hunt=hunt)
    ctl.hop_index = len(ctl.hops) - 1
    if hunt:
        ctl._hunter = ScreenHunter()
    return ctl


def test_the_stall_escape_is_off_without_the_hunt() -> None:
    """Every other hop table is measured without one; do not change them."""
    ctl = _walker(hunt=False)
    ctl.stuck = ctl.stuck_threshold + 1
    assert ctl._stall_escape(_snap(screen=0x49), ctl.hops[-1]) is None


def test_the_stall_escape_needs_the_stall_to_start() -> None:
    ctl = _walker(hunt=True)
    ctl.stuck = 0
    assert ctl._stall_escape(_snap(screen=0x49), ctl.hops[-1]) is None


def test_the_stall_escape_keeps_the_frame_after_stuck_resets() -> None:
    """The bug it exists for: one pixel of progress resets ``stuck``, and the
    hop rule that wedged Link takes the next 50 frames back."""
    ctl = _walker(hunt=True)
    hop = ctl.hops[-1]
    ctl.stuck = ctl.stuck_threshold + 1
    first = ctl._stall_escape(_snap(screen=0x49, link_x=56, link_y=125), hop)
    assert first is not None and first.reason.endswith("_escape")
    ctl.stuck = 0
    again = ctl._stall_escape(_snap(screen=0x49, link_x=56, link_y=126), hop)
    assert again is not None and again.reason.endswith("_escape")
    assert ctl.stall_escapes == 1


def test_the_stall_escape_lets_go_once_the_screen_scrolls() -> None:
    ctl = _walker(hunt=True)
    hop = ctl.hops[-1]
    ctl.stuck = ctl.stuck_threshold + 1
    ctl._stall_escape(_snap(screen=0x49), hop)
    ctl.stuck = 0
    assert ctl._stall_escape(_snap(screen=0x4A), hop) is None


def test_the_hunt_yields_while_link_is_still_on_the_arrival_edge() -> None:
    """A chase off the arrival edge walks Link back through it, and the hop
    then advances against the wrong screen."""
    ctl = _walker(hunt=True)
    hop = ctl.hops[-1]  # RIGHT: the arrival edge is the east one
    assert ctl._hunt_action(_snap(screen=0x49, link_x=248), hop) is None
