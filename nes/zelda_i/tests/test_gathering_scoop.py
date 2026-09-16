"""Pre-L1 transit drop scooping on the 0x77 -> 0x6F coast walk. No emulator.

The bomb branch is the phantom-drop trap: bomb is ROM item code ``0x00``
(``ids.BOMB_DROP_STATE``), which is exactly what a cleared object slot reads
as, so a bomb scoop that is not gated on Link owning bombs outranks every real
drop on the one leg where he owns none.
"""

from __future__ import annotations

from retro_harness.nes import nes_action
from zelda_i.dungeon.ids import (
    BOMB_DROP_OBJECT_TYPE,
    BOMB_DROP_STATE,
    HEART_DROP_STATE,
    RUPEE_DROP_OBJECT_TYPE,
    RUPEE_DROP_STATE,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.shop_p7 import (
    SCREEN_79_BEACH_Y,
    SHOP_P7_PRICE,
    ShopP7WalkController,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

HOP = ScreenHop(0x7A, "RIGHT", align_y=SCREEN_79_BEACH_Y)


def _snap(**kwargs) -> ZeldaSnapshot:
    fields = dict(
        mode=PLAY_MODE,
        level=0,
        screen=0x79,
        next_screen=0x79,
        link_x=120,
        link_y=SCREEN_79_BEACH_Y,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=0x22,  # 3 containers, 2 filled
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


def _phantom_slot(slot: int = 3, x: int = 96, y: int = SCREEN_79_BEACH_Y) -> ZeldaObject:
    """An emptied drop slot: ObjType still 0x60, ObjState cleared to 0x00.

    Indistinguishable from a live bomb drop by RAM alone — that is the point.
    """
    return ZeldaObject(
        slot=slot, type_id=BOMB_DROP_OBJECT_TYPE, x=x, y=y, facing=0, hp=0,
        state=BOMB_DROP_STATE,
    )


def _rupee(slot: int = 4, x: int = 144, y: int = SCREEN_79_BEACH_Y) -> ZeldaObject:
    return ZeldaObject(
        slot=slot, type_id=RUPEE_DROP_OBJECT_TYPE, x=x, y=y, facing=0, hp=0,
        state=RUPEE_DROP_STATE,
    )


def test_rupee_scoop_still_fires_when_the_shop_is_short() -> None:
    """Live leftover is 8R of 20R. The walk must still walk onto a floor rupee."""
    ctl = ShopP7WalkController()
    snap = _snap(rupees=8, bombs=0, objects=(_rupee(),))
    act = ctl._rupee_scoop(snap, HOP)
    assert act is not None
    assert act.reason == "scoop_rupee"


def test_real_rupee_wins_over_phantom_bomb_slot_pre_l1() -> None:
    """Pre-L1 Link owns no bombs: the 0x00-state slot must not outrank a rupee."""
    ctl = ShopP7WalkController()
    # Phantom sits WEST (closer); the rupee sits EAST. A bomb-first scoop
    # would walk LEFT, away from both the hop and the shop money.
    snap = _snap(
        rupees=0, bombs=0, objects=(_phantom_slot(x=96), _rupee(x=144))
    )
    act = ctl._rupee_scoop(snap, HOP)
    assert act is not None
    assert act.reason == "scoop_rupee"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_phantom_bomb_slot_alone_is_not_scooped_pre_l1() -> None:
    ctl = ShopP7WalkController()
    snap = _snap(rupees=0, bombs=0, objects=(_phantom_slot(),))
    assert ctl._rupee_scoop(snap, HOP) is None


def test_bomb_scoop_returns_once_link_owns_bombs() -> None:
    """The feature survives the gate: with bombs banked, a bomb drop scoops."""
    ctl = ShopP7WalkController()
    snap = _snap(rupees=SHOP_P7_PRICE, bombs=1, objects=(_phantom_slot(x=144),))
    act = ctl._rupee_scoop(snap, HOP)
    assert act is not None
    assert act.reason == "scoop_bomb"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_rupee_scoop_does_not_stop_at_the_shop_price() -> None:
    """Bombs are purchase 1 of 14. A rupee left on the floor at 20 is one the
    candle at 0x0C still needs, so the scoop is no longer price-gated."""
    ctl = ShopP7WalkController()
    snap = _snap(rupees=SHOP_P7_PRICE, bombs=0, objects=(_rupee(),))
    act = ctl._rupee_scoop(snap, HOP)
    assert act is not None
    assert act.reason == "scoop_rupee"


def test_heart_still_outranks_every_other_drop_when_hurt() -> None:
    ctl = ShopP7WalkController()
    heart = ZeldaObject(
        slot=2, type_id=RUPEE_DROP_OBJECT_TYPE, x=96, y=SCREEN_79_BEACH_Y, facing=0, hp=0,
        state=HEART_DROP_STATE,
    )
    snap = _snap(rupees=0, bombs=0, health=0x21, objects=(heart, _rupee(x=144)))
    act = ctl._rupee_scoop(snap, HOP)
    assert act is not None
    assert act.reason == "scoop_heart"
    assert list(act.action) == list(nes_action("LEFT"))


def _heart(slot: int = 2, x: int = 96, y: int = SCREEN_79_BEACH_Y) -> ZeldaObject:
    return ZeldaObject(
        slot=slot, type_id=RUPEE_DROP_OBJECT_TYPE, x=x, y=y, facing=0, hp=0,
        state=HEART_DROP_STATE,
    )


def test_a_full_health_link_walks_past_a_heart_to_the_rupee() -> None:
    """The heart branch read ``filled_hearts < heart_containers``.

    ``$066F``'s low nibble is whole hearts *minus one*, so at 3 of 3 that is
    ``2 < 3`` — true — and the walk detoured west for a heart it cannot bank
    while the money sat east. ``combat.heal_wanted`` is the honest test.
    """
    ctl = ShopP7WalkController()
    snap = _snap(rupees=0, bombs=0, health=0x22, objects=(_heart(x=96), _rupee(x=144)))
    act = ctl._rupee_scoop(snap, HOP)
    assert act is not None
    assert act.reason == "scoop_rupee"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_a_chipped_link_still_detours_for_the_heart() -> None:
    """A wooden chip is ``$0670 -= 0x80`` and never moves the whole-heart
    nibble — and it is exactly what takes the sword beam away, so the heart
    is worth more than its buffer."""
    ctl = ShopP7WalkController()
    snap = _snap(
        rupees=0, bombs=0, health=0x22, heart_partial=0x7F,
        objects=(_heart(x=96), _rupee(x=144)),
    )
    act = ctl._rupee_scoop(snap, HOP)
    assert act is not None
    assert act.reason == "scoop_heart"
    assert list(act.action) == list(nes_action("LEFT"))
