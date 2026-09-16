"""The full-health sword shot: its gate, its geometry, and its census.

No emulator. The gate is the part worth pinning, because it is the one that
surprised the walk: ``Link_BeHarmed`` subtracts damage from ``HeartPartial``
*first*, so one wooden chip takes ``$FF`` to ``$7F`` and the weapon is gone
until a heart or a fairy puts it back. Kinematics come from the live probe
(``scratch/probe_beam.py``); they are asserted here as constants so a later
edit cannot quietly retune a measured number.
"""

from __future__ import annotations

from retro_harness.controls import pressed_nes_buttons
from zelda_i.beam import (
    BEAM_HALF_WIDTH,
    BEAM_MUZZLE,
    BEAM_PARTIAL_MIN,
    BEAM_SPEED,
    BEAM_STAND_OFF,
    BeamPolicy,
    beam_aim,
    beam_live,
    beam_offsets,
    beam_ready,
    beam_stand,
    in_beam_lane,
)
from zelda_i.dungeon.ids import (
    OCTOROK_OBJECT_TYPE,
    TEKTITE_BLUE_OBJECT_TYPE,
    ZORA_OBJECT_TYPE,
)
from zelda_i.overworld.hunt import ScreenHunter
from zelda_i.ram import SWORD_SHOT_SLOT, PLAY_MODE, ZeldaObject, ZeldaSnapshot

BOX = (32, 214, 76, 198)


def _shot(state: int = 0, x: int = 0, y: int = 0) -> ZeldaObject:
    return ZeldaObject(
        slot=SWORD_SHOT_SLOT, type_id=0, x=x, y=y, facing=0, hp=0, state=state
    )


def _snap(**kwargs) -> ZeldaSnapshot:
    fields = dict(
        mode=PLAY_MODE,
        level=0,
        screen=0x7A,
        next_screen=0x7A,
        link_x=120,
        link_y=141,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=0x22,  # 3 of 3: low nibble == high nibble is "full"
        heart_partial=0xFF,
        triforce=0,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_all_dead=0,
        room_obj_count=0,
        room_item_id=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )
    fields.update(kwargs)
    return ZeldaSnapshot(**fields)


def _foe(
    slot: int = 1,
    x: int = 200,
    y: int = 141,
    hp: int = 1,
    type_id: int = TEKTITE_BLUE_OBJECT_TYPE,
) -> ZeldaObject:
    return ZeldaObject(
        slot=slot, type_id=type_id, x=x, y=y, facing=0, hp=hp, state=0
    )


# ----------------------------------------------------------------- gate ---


def test_the_shot_needs_a_sword_full_hearts_and_a_free_slot() -> None:
    assert beam_ready(_snap())
    assert not beam_ready(_snap(sword=0))
    assert not beam_ready(_snap(health=0x21))  # 2 of 3
    assert not beam_ready(_snap(sword_shot=_shot(state=0x10)))


def test_one_wooden_chip_takes_the_weapon_away() -> None:
    """``Link_BeHarmed``: ``HeartPartial -= damage`` before any heart borrow.

    A wooden octorok deals ``$80``, so a full ``$FF`` becomes ``$7F`` — one
    below :data:`BEAM_PARTIAL_MIN`. The beam is an *at full health* weapon,
    not a "most of the time" one, and that is why it is worth firing early.
    """
    assert BEAM_PARTIAL_MIN == 0x80
    assert beam_ready(_snap(heart_partial=0x80))
    assert not beam_ready(_snap(heart_partial=0x7F))
    assert 0xFF - 0x80 == 0x7F


def test_a_live_shot_is_its_own_cooldown() -> None:
    """``MakeSwordShot`` returns early on a non-zero state in slot ``$0E``."""
    assert not beam_live(_snap())
    assert beam_live(_snap(sword_shot=_shot(state=0x10)))
    assert beam_live(_snap(sword_shot=_shot(state=0x11)))


# ------------------------------------------------------------- geometry ---


def test_measured_kinematics_are_not_guesses() -> None:
    """``scratch/probe_beam.py`` on 0x77, hp ``0x22``/``0xFF``, all four ways."""
    assert BEAM_SPEED == 3.0
    assert BEAM_MUZZLE["RIGHT"] == (19, 0)
    assert BEAM_MUZZLE["LEFT"] == (-19, 0)
    assert BEAM_MUZZLE["UP"] == (0, -16)
    assert BEAM_MUZZLE["DOWN"] == (0, 30)


def test_a_body_behind_link_is_not_in_the_lane() -> None:
    assert in_beam_lane(120, 141, "RIGHT", 200, 141)
    assert not in_beam_lane(120, 141, "LEFT", 200, 141)
    assert beam_offsets(120, 141, "RIGHT", 200, 141) == (80, 0)
    assert beam_offsets(120, 141, "LEFT", 200, 141) == (-80, 0)


def test_the_lane_is_narrow_and_the_reach_is_the_screen() -> None:
    assert in_beam_lane(120, 141, "RIGHT", 200, 141 + BEAM_HALF_WIDTH)
    assert not in_beam_lane(120, 141, "RIGHT", 200, 141 + BEAM_HALF_WIDTH + 1)
    # No distance decay: the shot flies to the room bound, so a body on the
    # far side of the screen is still in the lane.
    assert in_beam_lane(16, 141, "RIGHT", 232, 141)


def test_aim_prefers_the_way_link_already_faces() -> None:
    """A turn is a frame; two bodies the same distance out should not cost one."""
    east, west = _foe(slot=1, x=200), _foe(slot=2, x=40)
    aim = beam_aim(120, 141, (east, west), prefer="LEFT")
    assert aim is not None and aim[0] == "LEFT"
    aim = beam_aim(120, 141, (east, west), prefer="RIGHT")
    assert aim is not None and aim[0] == "RIGHT"


def test_aim_takes_the_nearer_body_in_a_lane() -> None:
    near, far = _foe(slot=1, x=160), _foe(slot=2, x=210)
    aim = beam_aim(120, 141, (far, near))
    assert aim is not None and int(aim[1].slot) == 1


def test_nothing_in_a_lane_is_no_aim() -> None:
    assert beam_aim(120, 141, (_foe(x=200, y=100),)) is None


def test_the_stand_is_in_the_body_row_on_link_side_of_it() -> None:
    foe = _foe(x=180, y=141)
    assert beam_stand(120, 141, foe, BOX) == (180 - BEAM_STAND_OFF, 141)
    # East side of the same body, clamped into the chase box rather than
    # walked off the screen (180 + 56 is past ``HUNT_BOX`` xhi).
    assert beam_stand(220, 141, foe, BOX) == (BOX[1], 141)
    assert beam_stand(160, 100, _foe(x=160, y=160), BOX) == (160, 160 - BEAM_STAND_OFF)
    assert beam_stand(36, 141, _foe(x=40, y=141), BOX)[0] >= BOX[0]


def test_the_stand_is_out_of_contact_and_still_in_the_lane() -> None:
    foe = _foe(x=180, y=141)
    gx, gy = beam_stand(120, 141, foe, BOX)
    assert in_beam_lane(gx, gy, "RIGHT", 180, 141)
    assert abs(180 - gx) > 16  # MIN_DODGE_BODY; the point is not touching it


# --------------------------------------------------------------- census ---


def test_a_fire_is_the_slot_going_live_not_an_a_press() -> None:
    policy = BeamPolicy()
    policy.observe(_snap(), BOX)
    assert policy.fired == 0
    policy.observe(_snap(sword_shot=_shot(state=0x10, x=140, y=141)), BOX)
    assert policy.fired == 1
    assert policy.by_screen == {0x7A: 1}
    # Still the same shot: a live slot must not count twice.
    policy.observe(_snap(sword_shot=_shot(state=0x10, x=160, y=141)), BOX)
    assert policy.fired == 1


def test_an_impact_is_a_spread_inside_the_box_not_at_the_wall() -> None:
    """Odd state is ``SpreadShot``; the ROM spreads the frame a move is blocked."""
    policy = BeamPolicy()
    policy.observe(_snap(sword_shot=_shot(state=0x10, x=140, y=141)), BOX)
    policy.observe(_snap(sword_shot=_shot(state=0x11, x=180, y=141)), BOX)
    assert policy.impacts == 1

    wall = BeamPolicy()
    wall.observe(_snap(sword_shot=_shot(state=0x10, x=140, y=141)), BOX)
    wall.observe(_snap(sword_shot=_shot(state=0x11, x=236, y=141)), BOX)
    assert wall.impacts == 0


def test_the_stand_budget_ends_a_wait_that_is_not_paying() -> None:
    """A lane can be walled; a shot that never lands must not hold the chase."""
    policy = BeamPolicy(stand_max_frames=3)
    snap, foe = _snap(), _foe(x=180)
    assert policy.stand(snap, foe, BOX) is not None
    assert policy.stand(snap, foe, BOX) is not None
    assert policy.stand(snap, foe, BOX) is not None
    assert policy.stand(snap, foe, BOX) is None
    # A different body is a different wait.
    assert policy.stand(snap, _foe(slot=2, x=180), BOX) is not None


def test_a_shot_fired_at_a_scroll_line_never_exists() -> None:
    """``SetUpWeaponWithState`` deactivates a horizontal shot at x < $14 or
    >= $EC. A hop arrives at x=0, so the travelling press has to know."""
    from zelda_i.beam import beam_spawns

    assert not beam_spawns(0, "RIGHT")
    assert beam_spawns(40, "RIGHT")
    assert not beam_spawns(30, "LEFT")
    assert beam_spawns(0, "DOWN")  # vertical shots are not bounded in x
    assert beam_aim(0, 141, (_foe(x=200),)) is None
    assert beam_aim(40, 141, (_foe(x=200),)) is not None


def test_ready_is_pure_and_observe_counts_ready_frames() -> None:
    """``ready()`` is a gate. The census lives on ``observe()``."""
    policy = BeamPolicy()
    snap = _snap()
    assert policy.ready(snap)
    assert policy.ready_frames == 0
    assert policy.ready(snap)
    assert policy.ready_frames == 0
    policy.observe(snap, BOX)
    assert policy.ready_frames == 1
    policy.observe(snap, BOX)
    assert policy.ready_frames == 2
    chipped = _snap(heart_partial=0x7F)
    assert not policy.ready(chipped)
    policy.observe(chipped, BOX)
    assert policy.ready_frames == 2


# ------------------------------------------------------ hunt wiring ---


def test_a_lined_up_body_is_answered_with_the_shot_not_a_walk() -> None:
    """Full health plus a shared row is a free kill at any distance.

    The shot carries the blade's own damage points and damage type, and its
    kill runs ``HandleMonsterDied``, so the drop and the streak tick are the
    same ones the walk would have got by closing 80 px first.
    """
    hunter = ScreenHunter()
    foe = _foe(slot=1, x=200, y=141, type_id=TEKTITE_BLUE_OBJECT_TYPE, hp=1)
    snap = _snap(screen=0x78, link_x=120, objects=(foe,))
    hunter.observe(snap)
    act = hunter.take_beam(snap)
    assert act is not None and act.reason == "beam_78"
    buttons = pressed_nes_buttons(list(act.action))
    assert "A" in buttons and "RIGHT" in buttons


def test_the_shot_is_one_a_edge_then_an_idle() -> None:
    """``ButtonsPressed`` is an edge; a held A swings once and never again."""
    hunter = ScreenHunter()
    foe = _foe(slot=1, x=200, y=141, type_id=TEKTITE_BLUE_OBJECT_TYPE, hp=1)
    snap = _snap(screen=0x78, link_x=120, objects=(foe,))
    hunter.observe(snap)
    first = hunter.take_beam(snap)
    second = hunter.take_beam(snap)
    assert first is not None and "A" in pressed_nes_buttons(list(first.action))
    assert second is not None and second.reason == "beam_78_release"
    assert "A" not in pressed_nes_buttons(list(second.action))


def test_one_chip_of_damage_puts_the_shot_away() -> None:
    hunter = ScreenHunter()
    foe = _foe(slot=1, x=200, y=141, type_id=TEKTITE_BLUE_OBJECT_TYPE, hp=1)
    snap = _snap(screen=0x78, link_x=120, heart_partial=0x7F, objects=(foe,))
    hunter.observe(snap)
    act = hunter.take_beam(snap)
    assert act is None or not act.reason.startswith("beam")
    assert hunter.beam.fired == 0


def test_a_zora_is_never_worth_a_shot() -> None:
    """Same reason it is never a target: the slot leaves through
    ``DestroyMonster``, so there is no kill and no streak tick to buy."""
    hunter = ScreenHunter()
    snap = _snap(
        screen=0x78,
        link_x=120,
        objects=(_foe(slot=1, x=200, y=141, type_id=ZORA_OBJECT_TYPE),),
    )
    hunter.observe(snap)
    act = hunter.take_beam(snap)
    assert act is None or not act.reason.startswith("beam")


def test_a_transit_screen_still_shoots_what_walks_into_the_lane() -> None:
    """Transit means do not *chase*. A body already lined up is free."""
    hunter = ScreenHunter(transit_screens=frozenset({0x7B}))
    foe = _foe(slot=1, x=200, y=141, type_id=TEKTITE_BLUE_OBJECT_TYPE, hp=1)
    snap = _snap(screen=0x7B, link_x=120, objects=(foe,))
    hunter.observe(snap)
    act = hunter.take_beam(snap)
    assert act is not None and act.reason == "beam_7b"


def test_a_hopping_body_is_waited_for_at_range_not_closed_on() -> None:
    """A blue tektite's whole attack is the hop that lands on Link.

    Off the lane the hunt walks to a *beam* stand — in the body's row, 56 px
    out — instead of the sword stand 20 px off the sprite.
    """
    hunter = ScreenHunter()
    foe = _foe(slot=1, x=180, y=100, type_id=TEKTITE_BLUE_OBJECT_TYPE, hp=1)
    snap = _snap(screen=0x78, link_x=60, link_y=141, objects=(foe,))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.endswith("_beam")
    assert hunter.beam.stand_frames == 1


def test_a_shooter_keeps_its_axis_off_limits() -> None:
    """``_off_line`` still owns an octorok: its lane is the one place the
    shot's reach does not pay for standing in."""
    hunter = ScreenHunter()
    foe = _foe(slot=1, x=180, y=100, type_id=OCTOROK_OBJECT_TYPE, hp=2)
    snap = _snap(screen=0x78, link_x=60, link_y=141, objects=(foe,))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and not act.reason.endswith("_beam")
    assert hunter.beam.stand_frames == 0


def test_the_shot_is_reported_per_screen() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(screen=0x78))
    hunter.observe(_snap(screen=0x78, sword_shot=_shot(0x10, x=140, y=141)))
    report = hunter.report()
    assert report["beam_fired"] == 1
    assert report["beam_fired_by_screen"] == {"0x78": 1}
