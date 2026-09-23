"""On-route screen clearing: prey selection, the kill census, the budgets.

No emulator. The census is the part worth pinning: ``kills`` is inferred from
slot transitions and ``kills_counter`` from two ROM counters that
``Link_BeHarmed`` (collision, ``$04F0``) clears, so both have a way to lie
and both are asserted here.
"""

from __future__ import annotations

from retro_harness.controls import pressed_nes_buttons
from zelda_i.combat import (
    CAVE_TRIGGER_TYPE,
    FACING_EAST,
    FACING_WEST,
    SWORD_REACH,
    bodies_in_box,
    chebyshev,
)
from zelda_i.dungeon.behaviors import FIREBALL_TYPE as FIREBALL_OBJECT_TYPE
from zelda_i.dungeon.behaviors import ROCK_PROJECTILE_TYPE
from zelda_i.dungeon.ids import (
    BOMB_DROP_STATE,
    FIVE_RUPEE_DROP_STATE,
    HEART_DROP_STATE,
    OCTOROK_OBJECT_TYPE,
    RUPEE_DROP_OBJECT_TYPE,
    RUPEE_DROP_STATE,
    TEKTITE_BLUE_OBJECT_TYPE,
    ZORA_OBJECT_TYPE,
)
from zelda_i.dungeon.threat import MIN_DODGE_BODY
from zelda_i.beam import BeamPolicy
from zelda_i.overworld.hunt import (
    HUNT_BOX,
    HUNT_PICKUP_RADIUS,
    HUNT_TURN_CAP,
    ScreenHunter,
    sword_stand,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

# A real red octorok. The fixture used to be 0x38, which is the Digdogger id:
# harmless while the hunt only measured distance, wrong the moment prey
# selection reads the ROM drop row off the type byte.
OCTOROK = OCTOROK_OBJECT_TYPE


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


def _blade_only(**kwargs) -> ScreenHunter:
    """A hunter with the sword shot switched off. These are chase tests.

    ``_snap`` is 3 of 3 with ``$0670 == $FF``, which is exactly the state the
    full-health shot needs, and every fixture here parks a body on Link's own
    row — so a default hunter answers with a beam before any chase rule is
    consulted. That is the right answer live and the wrong one to assert a
    chase against. The shot has its own tests in ``test_beam``.
    """
    return ScreenHunter(beam=BeamPolicy(enabled=False), **kwargs)


# ---------------------------------------------------------------- prey ---


def test_prey_is_in_box_killable_and_not_a_trigger() -> None:
    xlo, xhi, ylo, yhi = HUNT_BOX
    prey = bodies_in_box(
        _snap(
            objects=(
                _foe(slot=1, x=160, y=141),
                _foe(slot=2, x=xhi + 4, y=141),  # past the east scroll margin
                _foe(slot=3, x=160, y=ylo - 4),  # above the north margin
                _foe(slot=4, x=160, y=141, type_id=CAVE_TRIGGER_TYPE, hp=240),
                _foe(slot=5, x=160, y=141, hp=0),  # corpse
            )
        ),
        HUNT_BOX,
    )
    assert [int(o.slot) for o in prey] == [1]


def test_drops_are_never_prey() -> None:
    assert bodies_in_box(_snap(objects=(_drop(),)), HUNT_BOX) == ()


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
    assert hunter.ledger.rupees_banked == 5
    hunter.observe(_snap(rupees=0))  # spent at the counter, not un-earned
    assert hunter.ledger.rupees_banked == 5


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
    assert hunter.ledger.kills_counter == 2
    hunter.observe(_snap(world_kill_count=0, help_drop_count=0))  # Link_BeHarmed
    assert hunter.ledger.kills_counter == 2
    hunter.observe(_snap(world_kill_count=1, help_drop_count=1))
    assert hunter.ledger.kills_counter == 3


def test_the_larger_counter_delta_is_the_one_banked() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(world_kill_count=15, help_drop_count=5))
    hunter.observe(_snap(world_kill_count=0, help_drop_count=6))  # 16th kill
    assert hunter.ledger.kills_counter == 1


# ---------------------------------------------------------------- step ---


def test_a_body_outranks_a_drop_in_its_pad() -> None:
    """Scooping a drop the wave is standing on walked into the body."""
    hunter = _blade_only()
    snap = _snap(objects=(_foe(slot=1, x=80), _drop(slot=2, x=80)))
    act = hunter.step(snap, 1)
    assert act is not None and act.reason == "hunt_78"


def test_a_clear_drop_is_banked_while_the_wave_is_still_up() -> None:
    """Live 0x79/0x7A let rupees expire because chase ignored every drop
    until the wave was empty. A drop outside every body's pad is money."""
    hunter = _blade_only()
    snap = _snap(link_x=120, objects=(_foe(slot=1, x=200), _drop(slot=2, x=80)))
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.startswith("hunt_scoop")
    # Toward the drop on the left, away from the body on the right.
    assert pressed_nes_buttons(list(act.action)) == ["LEFT"]


def test_a_drop_is_taken_once_the_wave_is_dead() -> None:
    hunter = ScreenHunter()
    snap = _snap(objects=(_drop(slot=2, x=80),))
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
    """``hunt_heal``, not ``hunt_drop``: the heal is its own rung above the
    beam now, because the chase rung below it never won a frame on a screen
    with a live wave — which is where a heart actually drops."""
    hunter = ScreenHunter()
    snap = _snap(health=0x22, heart_partial=0x40, objects=(_drop(state=HEART_DROP_STATE),))
    act = hunter.step(snap, 1)
    assert act is not None and act.reason == "hunt_heal"


def test_the_heal_outranks_the_beam_when_link_is_chipped() -> None:
    """One ``$0670`` chip is exactly what takes the beam away, so on every
    frame the heal can claim, the shot below it is already dead."""
    hunter = ScreenHunter()
    snap = _snap(
        health=0x22, heart_partial=0x7F,
        objects=(_foe(slot=1, x=200, y=141), _drop(slot=2, x=80, y=141,
                                                   state=HEART_DROP_STATE)),
    )
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.startswith("hunt_heal")


def test_a_full_health_link_does_not_walk_to_a_heart() -> None:
    hunter = _blade_only()
    snap = _snap(
        health=0x22, heart_partial=0xFF,
        objects=(_drop(slot=2, x=80, y=141, state=HEART_DROP_STATE),),
    )
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is None or not act.reason.startswith("hunt_heal")


def test_the_heal_budget_is_not_the_collect_budget() -> None:
    """A screen that spent its collect budget on rupees must still be able to
    walk to the fairy that hands the beam back."""
    hunter = ScreenHunter()
    hunter.census.collect_frames = hunter.collect_max_frames
    snap = _snap(
        health=0x22, heart_partial=0x7F,
        objects=(_drop(slot=2, x=80, y=141, state=HEART_DROP_STATE),),
    )
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.startswith("hunt_heal")


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
    assert hunter.census.screens_cleared == 1


def test_a_late_wave_is_still_hunted() -> None:
    """A screen is not written off on an empty first look: a dungeon settle
    spawn runs 80-100f and nothing says the overworld is faster."""
    hunter = _blade_only(settle_frames=1, spawn_wait_frames=50)
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
    hunter = _blade_only(settle_frames=1, spawn_wait_frames=1)
    hunter.step(_snap(), 1)
    assert 0x78 in hunter.done
    assert hunter.step(_snap(objects=(_foe(),)), 2) is None


def test_the_screen_budget_ends_a_hunt_that_will_not_finish() -> None:
    hunter = _blade_only(screen_max_frames=5)
    snap = _snap(objects=(_foe(),))
    for frame in range(1, 6):
        assert hunter.step(snap, frame) is not None
    assert hunter.step(snap, 6) is None
    assert hunter.census.screens_retired == 1
    assert any(note.startswith("hunt_budget") for note in hunter.census.notes)


def test_reset_restores_ordinary_budgets_after_a_destination_hunt() -> None:
    hunter = _blade_only(
        screen_max_frames=17,
        target_max_frames=9,
        destination_frames=2400,
        destination_target_frames=420,
    )

    hunter.take_destination(_snap(), 1)
    assert hunter.screen_max_frames == 2400
    assert hunter.targets.max_frames == 420

    hunter.reset()
    assert hunter.screen_max_frames == 17
    assert hunter.targets.max_frames == 9


def test_a_body_that_will_not_die_is_skipped_not_chased_to_the_budget() -> None:
    hunter = _blade_only(target_max_frames=3)
    stubborn = _foe(slot=1, x=200, y=141)
    # Out of the blade box (pad 40): a body at contact is struck, not chased,
    # so the per-target budget is only about the walk.
    reachable = _foe(slot=2, x=160, y=141)
    snap = _snap(link_x=120, objects=(stubborn, reachable))
    # Nearest first, so slot 2 is held; drive it past its budget.
    for frame in range(1, 5):
        hunter.step(snap, frame)
    assert 2 in hunter.targets.skipped
    assert hunter.targets.slot == 1


def test_sword_stand_is_reach_off_the_body_on_link_s_side() -> None:
    foe = _foe(x=160, y=141)
    assert sword_stand(100, 141, foe) == (160 - SWORD_REACH, 141)
    assert sword_stand(200, 141, foe) == (160 + SWORD_REACH, 141)


def test_a_far_body_is_approached_to_sword_stand_not_onto_the_sprite() -> None:
    hunter = _blade_only()
    foe = _foe(x=160, y=141)
    act = hunter.step(_snap(link_x=100, objects=(foe,)), 1)
    assert act is not None and act.reason == "hunt_78"
    assert "RIGHT" in pressed_nes_buttons(list(act.action))
    assert "A" not in pressed_nes_buttons(list(act.action))


def test_in_hitbox_slashes_in_place() -> None:
    hunter = ScreenHunter()
    foe = _foe(x=160, y=141)
    act = hunter.step(
        _snap(link_x=160 - SWORD_REACH, facing=FACING_EAST, objects=(foe,)), 1
    )
    assert act is not None and act.reason == "hunt_78_slash"
    buttons = pressed_nes_buttons(list(act.action))
    assert "A" in buttons
    # The direction rides along so the blade lands where the body is; the
    # attack state pins Link, so it is not a step onto the sprite.
    assert "RIGHT" in buttons


def test_the_swing_faces_the_body_at_every_pad_in_reach() -> None:
    """No dead band. Live 0x68 f=1763: ten frames facing WEST, foe 17px EAST.

    The old ladder could only turn at ``pad > MIN_DODGE_BODY + 2``, so at
    17-18px Link pulsed A at the wall behind him until the octorok walked in.

    The turn is now its own frame — ``dir+A`` across a facing swings the old
    way (``probe_turn_swing.py``) — so the contract at every pad is *turn
    toward the body, then swing*, never A into the wall behind him.
    """
    hunter = ScreenHunter()
    foe = _foe(x=160, y=141)
    for pad in range(MIN_DODGE_BODY - 6, SWORD_REACH + 1):
        hunter.reset()
        turn = hunter.step(
            _snap(link_x=160 - pad, facing=FACING_WEST, objects=(foe,)), 1
        )
        assert turn is not None, pad
        assert turn.reason == "hunt_78_slash_turn", (pad, turn.reason)
        buttons = pressed_nes_buttons(list(turn.action))
        assert "RIGHT" in buttons and "A" not in buttons, (pad, buttons)
        # The ROM has turned him: now the blade goes out where the body is.
        act = hunter.step(
            _snap(link_x=160 - pad, facing=FACING_EAST, objects=(foe,)), 2
        )
        assert act is not None and act.reason == "hunt_78_slash", (pad, act)
        buttons = pressed_nes_buttons(list(act.action))
        assert "A" in buttons and "RIGHT" in buttons, (pad, buttons)


def test_the_turn_wait_is_capped() -> None:
    """A body crossing a diagonal can ask for a new face every frame.

    ``HUNT_TURN_CAP`` frames of that is a dance, not a turn, so the press
    goes out anyway rather than holding the rung forever.
    """
    hunter = ScreenHunter()
    foe = _foe(x=160, y=141)
    reasons = []
    for frame in range(HUNT_TURN_CAP + 2):
        act = hunter.step(
            _snap(link_x=160 - 12, facing=FACING_WEST, objects=(foe,)), frame + 1
        )
        assert act is not None
        reasons.append(act.reason)
    assert reasons[:HUNT_TURN_CAP] == ["hunt_78_slash_turn"] * HUNT_TURN_CAP
    assert reasons[HUNT_TURN_CAP] == "hunt_78_slash"


def test_the_nearest_body_is_answered_not_the_held_target() -> None:
    """Live 0x49 f=4500: a 7-kill streak died to slot 4 at 9px on Link's lane
    while the walk was aimed at slot 1 thirty pixels north."""
    hunter = _blade_only()
    far = _foe(slot=1, x=120, y=141 - 30)
    near = _foe(slot=4, x=120 + 14, y=141)
    # Slot 1 is the held target: it was the only body when the screen opened.
    hunter.step(_snap(link_x=120, link_y=141, objects=(far,)), 1)
    assert hunter.targets.slot == 1
    act = hunter.step(
        _snap(link_x=120, link_y=141, facing=FACING_EAST, objects=(far, near)), 2
    )
    assert act is not None and act.reason.endswith("_slash")
    buttons = pressed_nes_buttons(list(act.action))
    assert "A" in buttons and "RIGHT" in buttons
    # The hunt answers slot 4 without dropping the target it walked out for.
    assert hunter.targets.slot == 1

    # At 9 px the same body is answered with the step out, not the blade:
    # the sword is an object in front of Link and ``blade1`` never landed a
    # press that close (:data:`HUNT_BLADE_MIN_FWD`).
    hunter = _blade_only()
    touching = _foe(slot=4, x=120 + 9, y=141)
    act = hunter.step(
        _snap(link_x=120, link_y=141, facing=FACING_EAST, objects=(touching,)), 1
    )
    assert act is not None and act.reason.endswith("_peel")
    assert "LEFT" in pressed_nes_buttons(list(act.action))


def test_every_swing_gets_its_own_release_edge() -> None:
    """A held A starts no second swing. Release is idle, never the face."""
    foe = _foe(x=160, y=141)
    link_idle = ZeldaObject(slot=0, type_id=0, x=0, y=0, facing=0, hp=0, state=0)
    link_swinging = ZeldaObject(slot=0, type_id=0, x=0, y=0, facing=0, hp=0, state=1)
    hunter = ScreenHunter()

    seen = []
    for frame in (1, 2, 3, 4, 5, 6, 7, 8):
        act = hunter.step(
            _snap(
                link_x=160 - SWORD_REACH,
                facing=FACING_EAST,
                objects=(link_idle, foe),
            ),
            frame,
        )
        assert act is not None
        seen.append((act.reason, pressed_nes_buttons(list(act.action))))

    reasons = [r for r, _ in seen]
    assert reasons == ["hunt_78_slash", "hunt_78_slash_release"] * 4, reasons
    for (reason, buttons) in seen:
        if reason.endswith("_slash"):
            assert "A" in buttons and "RIGHT" in buttons
        else:
            # No A to re-trigger on, and no direction to close the gap with.
            assert buttons == [] or buttons == ()
    assert hunter.census.release_frames == 4

    act = hunter.step(
        _snap(
            link_x=160 - SWORD_REACH,
            facing=FACING_EAST,
            objects=(link_swinging, foe),
        ),
        9,
    )
    assert act is not None and act.reason == "hunt_78_slash_recover"
    assert "A" not in pressed_nes_buttons(list(act.action))


def test_the_swing_waits_on_links_own_animation_not_a_cadence() -> None:
    """``$00AC`` slot 0 is non-zero for the whole wooden swing.

    The blind ``frames % 8 < 3`` cadence left up to five idle frames per swing
    with a body closing ~1px a frame (0x49 f=3653: ten frames standing while
    slot 4 walked 16 -> 9). While the animation runs the hunt recovers rather
    than pressing, and it never spends a *release* frame there — the
    animation already holds A down.
    """
    foe = _foe(x=160, y=141)
    link_swinging = ZeldaObject(slot=0, type_id=0, x=0, y=0, facing=0, hp=0, state=1)
    hunter = ScreenHunter()

    for frame in (1, 2, 3, 4):
        act = hunter.step(
            _snap(
                link_x=160 - SWORD_REACH,
                facing=FACING_EAST,
                objects=(link_swinging, foe),
            ),
            frame,
        )
        assert act is not None and act.reason == "hunt_78_slash_recover", frame
        assert "A" not in pressed_nes_buttons(list(act.action))
    assert hunter.census.release_frames == 0


def test_blocked_align_swings_through_the_same_release_edge() -> None:
    """``_approach`` is the other producer of a ``_slash`` frame.

    In reach, off-axis, and the strafe step leaves the box: that used to
    hold A every frame the same way ``_strike`` did. Drive ``_approach``
    directly so a contact-ladder test cannot paper over it.
    """
    foe = _foe(x=160, y=157)
    hunter = ScreenHunter(box=(32, 214, 76, 142))
    snap = _snap(link_x=160 - SWORD_REACH, link_y=141, facing=FACING_EAST, objects=(foe,))
    pad = chebyshev(160 - SWORD_REACH, 141, 160, 157)
    assert pad == SWORD_REACH

    seen = []
    for frame in (1, 2, 3, 4):
        act = hunter._approach(snap, frame, foe, foe, pad, "hunt_78")
        seen.append((act.reason, pressed_nes_buttons(list(act.action))))

    reasons = [r for r, _ in seen]
    assert reasons == ["hunt_78_slash", "hunt_78_slash_release"] * 2, reasons
    for reason, buttons in seen:
        if reason.endswith("_slash"):
            assert "A" in buttons
        else:
            assert buttons == [] or buttons == ()
    assert hunter.census.release_frames == 2


def test_a_hop_inside_the_blade_does_not_spin_the_face() -> None:
    """Tektite hop that stays in the current hitbox is not an align.

    Live 0x4A (136,111): dx/dy flipped every bounce and Link turned
    L/R/U/D in place. Keep the covering face and slash.
    """
    hunter = ScreenHunter()
    foe = _foe(x=160, y=141 + 6)
    snap = _snap(
        link_x=160 - SWORD_REACH,
        link_y=141,
        facing=FACING_EAST,
        objects=(foe,),
    )
    pad = chebyshev(160 - SWORD_REACH, 141, 160, 141 + 6)
    assert pad <= SWORD_REACH
    act = hunter._approach(snap, 1, foe, foe, pad, "hunt_4a")
    buttons = pressed_nes_buttons(list(act.action))
    assert act.reason.endswith("_slash")
    assert "A" in buttons and "RIGHT" in buttons
    assert "UP" not in buttons and "DOWN" not in buttons


def test_reopen_on_enter_fights_a_cleared_screen_again() -> None:
    """Contact still answers on a done screen; reopen is the *chase*."""
    hunter = ScreenHunter(reopen_on_enter=True)
    hunter.done.add(0x78)
    hunter.screen = 0x68
    far = _foe(x=200, y=180)
    act = hunter.step(_snap(screen=0x78, link_x=120, link_y=141, objects=(far,)), 1)
    assert 0x78 not in hunter.done
    assert act is not None


def test_one_pass_does_not_reopen_a_cleared_screen() -> None:
    hunter = ScreenHunter()
    hunter.done.add(0x78)
    hunter.screen = 0x68
    far = _foe(x=200, y=180)
    act = hunter.step(_snap(screen=0x78, link_x=120, link_y=141, objects=(far,)), 1)
    assert 0x78 in hunter.done
    assert act is None


def test_report_names_rupees_that_hit_the_floor_and_were_not_banked() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(rupees=0, objects=(_foe(slot=1),)))
    hunter.observe(
        _snap(rupees=0, objects=(_drop(slot=1, state=FIVE_RUPEE_DROP_STATE),))
    )
    hunter.observe(_snap(rupees=0, objects=()))
    hunter.observe(_snap(rupees=0, objects=(_foe(slot=2),)))
    hunter.observe(_snap(rupees=1, objects=(_drop(slot=2, state=RUPEE_DROP_STATE),)))
    assert hunter.ledger.rupees_dropped == 6
    assert hunter.ledger.rupees_banked == 1
    report = hunter.report()
    assert report["rupees_dropped"] == 6
    assert report["rupees_left"] == 5
    assert report["screens"][0]["rupees_dropped"] == 6


def test_off_axis_in_reach_strafes_instead_of_closing() -> None:
    hunter = ScreenHunter()
    foe = _foe(x=160, y=157)
    act = hunter.step(
        _snap(link_x=160 - SWORD_REACH, link_y=141, facing=FACING_EAST, objects=(foe,)),
        1,
    )
    assert act is not None and act.reason == "hunt_78_align"
    buttons = pressed_nes_buttons(list(act.action))
    assert "DOWN" in buttons
    assert "RIGHT" not in buttons


def test_inside_the_body_pad_swings_instead_of_peeling() -> None:
    """A peel started inside ``MIN_DODGE_BODY`` cannot finish.

    The sidestep has to walk the whole pad before it clears the hitbox and
    the body closes ~1px/frame. The measured one (0x49 f=4500) pressed LEFT
    into a bush for eight frames while slot 4 closed 16 -> 8, and it turned
    the swing around to face the wall.
    """
    hunter = ScreenHunter()
    foe = _foe(x=160, y=141)
    act = hunter.step(_snap(link_x=160 - (MIN_DODGE_BODY - 2), objects=(foe,)), 1)
    assert act is not None and act.reason == "hunt_78_slash"
    buttons = pressed_nes_buttons(list(act.action))
    assert "A" in buttons and "RIGHT" in buttons
    assert "LEFT" not in buttons


def _rock(x: int, y: int = 141, slot: int = 11):
    return ZeldaObject(
        slot=slot, type_id=ROCK_PROJECTILE_TYPE, x=x, y=y, facing=0, hp=0, state=0
    )


def test_the_shield_can_still_be_switched_off() -> None:
    """``shield=False`` leaves the hunt swinging at the rock's shooter."""
    hunter = ScreenHunter(shield=False)
    act = _drive(hunter, (160, 156, 152, 148), link_x=120, facing=FACING_EAST)
    assert act is not None and not act.reason.startswith("hunt_shield")
    assert hunter.shield_policy.frames == 0
    assert hunter.shield_policy.frames == 0


def _drive(hunter: ScreenHunter, xs, **snap_kwargs):
    """Feed consecutive frames so the tracker has a velocity to read."""
    act = None
    for frame, x in enumerate(xs, start=1):
        snap = _snap(objects=(_foe(x=200, y=141), _rock(x)), **snap_kwargs)
        hunter.observe(snap)
        act = hunter.step(snap, frame)
    return act


def test_an_inbound_rock_is_blocked_not_swung_at() -> None:
    """Live 0x58 f=2056: a rock hit Link mid-swing while he stood at reach.

    The small shield eats a rock for free while Link faces it and is not
    attacking, and a wooden swing pins him with the shield down for longer
    than the rock takes to arrive.
    """
    hunter = ScreenHunter(shield=True)
    act = _drive(hunter, (160, 156, 152, 148), link_x=120, facing=FACING_EAST)
    assert act is not None and act.reason == "hunt_shield"
    assert pressed_nes_buttons(list(act.action)) == []
    assert hunter.shield_policy.frames > 0


def test_a_rock_from_behind_turns_the_shield_round() -> None:
    hunter = ScreenHunter(shield=True)
    act = _drive(hunter, (160, 156, 152, 148), link_x=120, facing=FACING_WEST)
    assert act is not None and act.reason == "hunt_shield_turn"
    assert "RIGHT" in pressed_nes_buttons(list(act.action))


def test_a_body_at_contact_outranks_a_blockable_rock() -> None:
    """Live ``contact4``: shielding a rock while octoroks closed to 9px three
    times ran Link out of hearts on 0x49 (mode 17). ``assess`` picks the
    soonest hazard, and a body hands the frame back to the sword."""
    hunter = ScreenHunter(shield=True)
    reasons, swings = [], 0
    for frame, rx in enumerate((160, 156, 152, 148, 144, 140, 136, 132), start=1):
        snap = _snap(
            link_x=120,
            link_y=141,
            facing=FACING_WEST,
            # 14 px, not 9: inside 9 the blade reaches nothing, and this test
            # is about which *rung* owns the frame, not about the near end.
            objects=(_foe(slot=1, x=106, y=141), _rock(rx)),
        )
        hunter.observe(snap)
        act = hunter.step(snap, frame)
        assert act is not None
        reasons.append(act.reason)
        swings += "A" in pressed_nes_buttons(list(act.action))
    assert not any(r.startswith("hunt_shield") for r in reasons), reasons
    assert swings > 0, reasons
    assert hunter.shield_policy.frames == 0


def test_a_rock_that_is_not_arriving_does_not_stop_the_hunt() -> None:
    """Travelling away: no shot to block, so the sword keeps its frames."""
    hunter = ScreenHunter(shield=True)
    act = _drive(hunter, (160, 164, 168, 172), link_x=120, facing=FACING_EAST)
    assert act is not None and not act.reason.startswith("hunt_shield")


def test_two_hearts_of_three_still_hunt() -> None:
    """``$066F`` 0x21 is *two* whole hearts, not one: the low nibble reads one
    low (``ram.whole_hearts``). The gate that read it literally retired 0x58,
    0x59 and 0x49 on the 2026-09-15 baseline after a single chip hit."""
    hunter = ScreenHunter()
    snap = _snap(health=0x21, objects=(_foe(x=200),))
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.startswith("hunt_")


def test_the_last_heart_stops_the_chase_but_not_the_blade() -> None:
    """One whole heart (0x20) guards: no walking to a far body, but a body at
    contact is still answered, because ``Link_BeHarmed`` does not care that the
    hunt gave up. The baseline handed a live 0x49 wave to a hop with no combat
    and Link walked into two octoroks at 9px."""
    hunter = ScreenHunter()
    far = _snap(health=0x20, objects=(_foe(x=200),))
    assert hunter.step(far, 1) is None
    assert hunter.census.guard_frames == 1

    hunter = ScreenHunter()
    near = _snap(health=0x20, link_x=120, link_y=141, objects=(_foe(slot=1, x=134, y=141),))
    act = hunter.step(near, 1)
    assert act is not None and "A" in pressed_nes_buttons(list(act.action))

    # And at 9 px, where the blade cannot reach, the answer is the step out.
    hunter = ScreenHunter()
    touching = _snap(
        health=0x20, link_x=120, link_y=141, objects=(_foe(slot=1, x=129, y=141),)
    )
    act = hunter.step(touching, 1)
    assert act is not None and act.reason.endswith("_peel")


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
    assert hunter.ledger.damage_taken == 1
    hunter.observe(_snap(health=0x22, heart_partial=0xFF))  # healed
    assert hunter.ledger.damage_taken == 1


def test_a_streak_reset_short_of_the_forced_drop_is_recorded() -> None:
    hunter = ScreenHunter()
    hunter.observe(_snap(world_kill_count=3, help_drop_count=3))
    hunter.observe(_snap(world_kill_count=4, help_drop_count=4))
    hunter.observe(_snap(world_kill_count=0, help_drop_count=0))
    assert hunter.ledger.streak_resets == 1
    assert hunter.ledger.streak_best == 4


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
    assert hunter.ledger.hurt_events == 1
    assert hunter.ledger.damage_taken == 0


# ----------------------------------------------- wiring into the hop path ---


def _walker(hunt: bool):
    """Stall-escape fixture: the measured 0x49 (56,125) pocket, not shop_p7."""
    from zelda_i.overworld.graph import ScreenHop
    from zelda_i.overworld.path import OverworldPathController

    ctl = OverworldPathController(
        hops=(ScreenHop(0x4A, "RIGHT", align_y=141),),
        hunter=ScreenHunter() if hunt else None,
    )
    ctl.hop_index = 0
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


# -------------------------------------------------------- prey / value ---
# The corridor's money is a drop row and a streak, not a body count
# (``overworld.prey``). These pin the three things that changes: what is
# never a target, which of two bodies is held, and when a cheap chase stops
# being worth the health it risks.


def _fireball(x: int, y: int = 157, slot: int = 10):
    """A Zora's spit. ``behaviors.shield_blocks`` needs the Magical Shield for
    one, so no shield rule in the hunt has anything to say about it."""
    return ZeldaObject(
        slot=slot, type_id=FIREBALL_OBJECT_TYPE, x=x, y=y, facing=0x0A, hp=192, state=0x10
    )


def test_a_zora_is_never_a_target() -> None:
    """``UpdateZora`` submerges it with ``DestroyMonster`` — no
    ``HandleMonsterDied``, so the slot vanishing is not even a streak tick."""
    hunter = ScreenHunter()
    snap = _snap(objects=(_foe(slot=1, x=200, y=141, type_id=ZORA_OBJECT_TYPE),))
    hunter.observe(snap)
    assert hunter.step(snap, 1) is None
    assert hunter.targets.passed.get("zora") == 1


def test_a_near_body_is_not_aligned_off_a_two_pixel_hop() -> None:
    """Live 0x79 hunt_79 twerked 133<->135: ``_approach`` aligned UP/DOWN
    on dy=2 while already in the blade row. Pad 18 is inside sword reach
    and outside the peel pad, so this is the align rung, not the peel."""
    hunter = _blade_only()
    snap = _snap(link_x=120, link_y=141, objects=(_foe(slot=1, x=138, y=143),))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None
    assert not act.reason.endswith("_align")


def test_the_richer_drop_row_is_held_over_the_nearer_body() -> None:
    """A blue tektite is row 1 (0.891 R/kill, two 5-rupees); a red octorok is
    row 0 (0.156). Nearest-first read them as the same body."""
    hunter = _blade_only()
    tektite = _foe(slot=2, x=190, y=141, type_id=TEKTITE_BLUE_OBJECT_TYPE)
    snap = _snap(link_x=120, objects=(_foe(slot=1, x=170, y=141), tektite))
    hunter.observe(snap)
    hunter.step(snap, 1)
    assert hunter.targets.slot == 2


def test_a_cheap_body_is_still_chased_at_full_health() -> None:
    """Nine red octoroks cost 0.00 hearts live; refusing them on value alone
    would be tuning against the arithmetic (``prey.THRIFTY_CHASE_RADIUS``)."""
    hunter = _blade_only()
    snap = _snap(health=0x22, link_x=40, objects=(_foe(slot=1, x=200, y=141),))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.startswith("hunt_")
    assert hunter.targets.slot == 1


def test_a_cheap_body_across_the_box_is_declined_on_short_health() -> None:
    """0x21 is two whole hearts. A contact there costs the streak *and* the
    hearts 0x4A's six tektites need, which halves every break-even."""
    hunter = ScreenHunter()
    snap = _snap(health=0x21, link_x=40, objects=(_foe(slot=1, x=200, y=141),))
    hunter.observe(snap)
    assert hunter.step(snap, 1) is None
    assert hunter.targets.passed.get("octorok") == 1


def test_a_rich_body_is_still_chased_on_short_health() -> None:
    hunter = ScreenHunter()
    snap = _snap(
        health=0x21,
        link_x=40,
        objects=(_foe(slot=1, x=200, y=141, type_id=TEKTITE_BLUE_OBJECT_TYPE),),
    )
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and hunter.targets.slot == 1


def test_tektites_are_not_retired_at_the_ordinary_screen_budget() -> None:
    """Live 0x79/0x7A retired at 600f with 3+3 tektites still up. The 5-rupee
    row keeps the chase until the destination cap."""
    hunter = _blade_only()
    snap = _snap(
        screen=0x79,
        objects=(_foe(slot=1, x=160, type_id=TEKTITE_BLUE_OBJECT_TYPE),),
    )
    hunter.observe(snap)
    hunter.step(snap, 1)
    hunter.screen_frames = hunter.screen_max_frames + 1
    act = hunter.step(snap, 2)
    assert 0x79 not in hunter.done
    assert act is not None and act.reason.startswith("hunt_")


def test_an_octorok_screen_still_retires_at_the_ordinary_budget() -> None:
    hunter = _blade_only()
    snap = _snap(screen=0x78, objects=(_foe(slot=1, x=160),))
    hunter.observe(snap)
    hunter.step(snap, 1)
    hunter.screen_frames = hunter.screen_max_frames + 1
    hunter.step(snap, 2)
    assert 0x78 in hunter.done


def test_a_chase_longer_than_the_screen_budget_never_starts() -> None:
    hunter = _blade_only(screen_max_frames=60)
    snap = _snap(link_x=40, objects=(_foe(slot=1, x=200, y=141),))
    hunter.observe(snap)
    assert hunter.step(snap, 1) is None


# ------------------------------------------------------------- fireball ---


def test_a_dwelling_fireball_is_stepped_away_from() -> None:
    """The measured hole. A Zora's shot holds ``ZORA_MUZZLE_DWELL`` (17)
    frames on the muzzle, so the tracker reads zero velocity and
    ``threat.assess`` calls it safe for the entire dodge window — which is how
    0x59 f=2888 fired 165 px down Link's own row while he walked east into it
    (``scratch/zora1.json``)."""
    hunter = ScreenHunter()
    act = None
    for frame in range(1, 5):
        snap = _snap(link_x=60, link_y=157, objects=(_fireball(x=196, y=157),))
        hunter.observe(snap)
        act = hunter.step(snap, frame)
    assert act is not None and act.reason == "hunt_duck"
    # Perpendicular to the muzzle bearing, never along it.
    assert pressed_nes_buttons(list(act.action))[0] in ("UP", "DOWN")
    assert hunter.shield_policy.census.ducks > 0


def test_a_spit_south_of_the_exit_band_is_dodged_along_it() -> None:
    """pre_l1_shortfall1, 0x7D f=8130: spit at (162, 194), Link at y=138
    inside 137–145. The bearing is mostly east, so the old dodge walked UP
    onto the rock at y=109. The shot is not on the band."""
    hunter = ScreenHunter()
    snap = _snap(
        screen=0x7D, link_x=72, link_y=138, objects=(_fireball(x=162, y=194),)
    )
    hunter.observe(snap)
    act = hunter.step(snap, 1, y_band=(137, 145))
    assert act is not None and act.reason == "hunt_duck"
    assert pressed_nes_buttons(list(act.action))[0] == "LEFT"


def test_the_duck_can_be_switched_off_without_the_shield() -> None:
    hunter = ScreenHunter(duck=False)
    for frame in range(1, 5):
        snap = _snap(link_x=60, link_y=157, objects=(_fireball(x=196, y=157),))
        hunter.observe(snap)
        act = hunter.step(snap, frame)
    assert act is None or act.reason != "hunt_duck"


def test_a_fireball_across_the_map_is_left_to_the_evader() -> None:
    """Past ``HUNT_MUZZLE_ALARM`` the shot arrives with more warning than the
    dwell was worth, and the caller's evader can see it once it moves."""
    hunter = ScreenHunter()
    for frame in range(1, 5):
        snap = _snap(link_x=40, link_y=90, objects=(_fireball(x=230, y=200),))
        hunter.observe(snap)
        act = hunter.step(snap, frame)
    assert act is None or act.reason != "hunt_duck"


# ------------------------------------------------------------- transit ---


def test_a_transit_screen_is_crossed_not_cleared() -> None:
    """0x59 is peahat x4 plus a Zora — ROM drop row 3 — and one live pass cost
    a whole heart, 533 frames and a 5-kill streak for one kill."""
    hunter = _blade_only(transit_screens=frozenset({0x59}))
    snap = _snap(screen=0x59, link_x=40, objects=(_foe(slot=1, x=200, y=141),))
    hunter.observe(snap)
    assert hunter.step(snap, 1) is None
    assert hunter.census.transit_frames == 1
    assert hunter.census.frames_by_screen.get(0x59, 0) == 0


def test_a_transit_screen_still_answers_a_body_at_contact() -> None:
    """Declining the wave is not the same as standing in it."""
    hunter = ScreenHunter(transit_screens=frozenset({0x59}))
    snap = _snap(screen=0x59, link_x=120, link_y=141, objects=(_foe(slot=1, x=134, y=141),))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and "A" in pressed_nes_buttons(list(act.action))

    # Inside the blade the transit screen still answers — with the peel.
    hunter = ScreenHunter(transit_screens=frozenset({0x59}))
    snap = _snap(screen=0x59, link_x=120, link_y=141, objects=(_foe(slot=1, x=129, y=141),))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.endswith("_peel")


def test_a_transit_screen_still_banks_a_drop_it_walks_past() -> None:
    hunter = ScreenHunter(transit_screens=frozenset({0x59}))
    snap = _snap(screen=0x59, link_x=120, link_y=141, objects=(_drop(x=140, y=141),))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.startswith("hunt_scoop")


def test_a_duck_never_steps_into_a_body() -> None:
    """A shot is not a reason to walk into an octorok. ``ShotPolicy.face``
    has kept this rule for the shield since 0x49 killed three shield walks;
    the dodge shipped without it and the first live run died on 0x49 with
    five octorok_fast contacts."""
    hunter = ScreenHunter()
    act = None
    for frame in range(1, 5):
        snap = _snap(
            link_x=60,
            link_y=157,
            objects=(
                _fireball(x=196, y=157),
                # Both perpendicular steps are covered by a body.
                _foe(slot=1, x=60, y=141),
                _foe(slot=2, x=60, y=173),
            ),
        )
        hunter.observe(snap)
        act = hunter.step(snap, frame)
    assert act is None or act.reason != "hunt_duck"


def test_the_target_order_is_a_walk_cost_not_a_contact_pad() -> None:
    """``combat.nearest_to`` — what the value order replaced — measured
    manhattan. Ranking on chebyshev re-picks a different octorok on every
    off-axis wave, which reshuffles a frame-perfect corridor for nothing."""
    hunter = _blade_only()
    # Equal value (both row 0). Off-axis body is nearer by chebyshev (60) and
    # further by manhattan (120) than the on-axis one (80 either way).
    on_axis = _foe(slot=1, x=200, y=141)
    off_axis = _foe(slot=2, x=180, y=81)
    snap = _snap(link_x=120, link_y=141, objects=(on_axis, off_axis))
    hunter.observe(snap)
    hunter.step(snap, 1)
    assert hunter.targets.slot == 1


# ------------------------------------------------- cleared vs retired ---


def test_a_cleared_screen_still_walks_onto_the_rupee_it_dropped() -> None:
    """The 5R-on-the-floor gap (``pre_l1_beam4``: 24 dropped, 19 banked).

    The kill that empties a screen drops on the same frame the screen goes
    ``done``, and ``done`` used to collect ``heal_only`` — so the last drop of
    every screen was only ever banked if it happened to land inside the
    path's own 48 px ``_rupee_scoop`` radius.
    """
    hunter = _blade_only()
    hunter.done.add(0x78)
    hunter.cleared.add(0x78)
    snap = _snap(link_x=120, link_y=141, objects=(_drop(x=160, y=141),))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.startswith("hunt_scoop")


def test_a_budget_retired_screen_does_not_chase_money_through_the_wave() -> None:
    """``_retire`` means the wave outlasted the budget, not that it is gone.
    Only a heart is worth crossing live bodies for, and that is what
    ``heal_only`` says."""
    hunter = _blade_only()
    hunter._retire(0x78, "budget")
    assert 0x78 in hunter.done and 0x78 not in hunter.cleared
    snap = _snap(link_x=120, link_y=141, objects=(_drop(x=160, y=141),))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is None or not act.reason.startswith("hunt_scoop")


def test_the_screen_table_does_not_call_a_budget_retire_cleared() -> None:
    """``cleared`` read ``screen in self.done``, so three retires on the
    2026-09-15 coast walk printed as cleared screens."""
    hunter = _blade_only()
    snap = _snap(screen=0x79, objects=(_foe(slot=1, x=200, y=141),))
    hunter.observe(snap)
    hunter.step(snap, 1)
    hunter._retire(0x79, "budget")
    row = next(r for r in hunter.screen_table() if r["screen"] == "0x79")
    assert row["cleared"] is False and row["retired"] is True


def test_reopening_a_screen_reopens_the_cleared_claim_too() -> None:
    """A lap re-enters a screen it cleared; leaving it in ``cleared`` would
    scoop-only a wave that is live again."""
    hunter = _blade_only(reopen_on_enter=True)
    hunter.done.add(0x78)
    hunter.cleared.add(0x78)
    snap = _snap(objects=(_foe(slot=1, x=200, y=141),))
    hunter.observe(snap)
    hunter.step(snap, 1)
    assert 0x78 not in hunter.done and 0x78 not in hunter.cleared


def test_a_drop_across_a_live_screen_is_not_worth_walking_to() -> None:
    """0x7E is four ``octorok_fast`` plus a Zora. One pass spent 240 frames on
    ``hunt_heal`` and 179 on ``hunt_scoop`` crossing it for one heart and 2R,
    and took five of the walk's twelve hits doing it (``pre_l1_anyrow1``)."""
    hunter = ScreenHunter()
    far = ZeldaObject(
        slot=2, type_id=RUPEE_DROP_OBJECT_TYPE, x=210, y=141, facing=0, hp=0,
        state=HEART_DROP_STATE,
    )
    snap = _snap(link_x=40, link_y=141, health=0x22, heart_partial=0x7F,
                 objects=(far,))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is None or not act.reason.startswith("hunt_heal")


def test_a_drop_inside_the_radius_is_still_taken() -> None:
    hunter = ScreenHunter()
    near = ZeldaObject(
        slot=2, type_id=RUPEE_DROP_OBJECT_TYPE, x=40 + HUNT_PICKUP_RADIUS - 8,
        y=141, facing=0, hp=0, state=HEART_DROP_STATE,
    )
    snap = _snap(link_x=40, link_y=141, health=0x22, heart_partial=0x7F,
                 objects=(near,))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and act.reason.startswith("hunt_heal")


def test_an_empty_screen_still_walks_the_whole_box_for_a_drop() -> None:
    """With nothing alive, distance costs frames and nothing else — the
    radius is about crossing a *wave*, not about reach."""
    hunter = _blade_only()
    far = _drop(slot=2, x=210, y=141)
    snap = _snap(link_x=40, link_y=141, objects=(far,))
    hunter.observe(snap)
    act = hunter.step(snap, 1)
    assert act is not None and act.reason == "hunt_drop"


def test_one_body_cannot_hold_the_contact_strike_forever() -> None:
    """The contact strike was the top of the ladder with no budget on it at
    all: not the per-target one (``TargetBook`` is three rungs down), not the
    per-screen one. ``pre_l1_bound1`` spent 24877 of a 30000 frame timeout on
    0x7C — 3072 A presses at one body that would not die."""
    hunter = _blade_only()
    snap = _snap(link_x=120, link_y=141, objects=(_foe(slot=1, x=132, y=141),))
    reasons = set()
    for frame in range(1, hunter.strike_slot_max_frames + 40):
        hunter.observe(snap)
        act = hunter.step(snap, frame)
        if act is not None:
            reasons.add(act.reason)
    assert any("wedged" in note for note in hunter.census.notes)
    act = hunter.step(snap, 10**6)
    assert act is None or "_slash" not in act.reason


def test_the_wedge_budget_spends_the_screen_budget_too() -> None:
    """A screen that burns its whole chase on one wedged body still has to
    retire and hand the hop back."""
    hunter = _blade_only()
    snap = _snap(link_x=120, link_y=141, objects=(_foe(slot=1, x=132, y=141),))
    for frame in range(1, 60):
        hunter.observe(snap)
        hunter.step(snap, frame)
    assert hunter.screen_frames >= 50
    assert hunter.census.frames_by_screen.get(0x78, 0) >= 50


def test_a_fresh_body_in_the_same_slot_is_still_struck() -> None:
    """Identity is (slot, type): a slot is reused the moment its occupant
    dies, and the replacement has not had its turn."""
    hunter = _blade_only()
    snap = _snap(link_x=120, link_y=141, objects=(_foe(slot=1, x=132, y=141),))
    wedged = snap.objects[0]
    hunter._unkillable.add((int(wedged.slot), int(wedged.type_id)))
    hunter.observe(snap)
    assert hunter._strike_budget(0x78, wedged) is False
    other = ZeldaObject(
        slot=int(wedged.slot), type_id=int(wedged.type_id) + 1, x=132, y=141,
        facing=0x02, hp=0x10, state=1,
    )
    assert hunter._strike_budget(0x78, other) is True
