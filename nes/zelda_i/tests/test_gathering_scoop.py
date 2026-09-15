"""Pre-L1 transit drop scooping on the 0x77 -> 0x4A walk. No emulator.

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
from zelda_i.overworld.gathering import (
    BOMB_SHOP_PRICE,
    ShopP7WalkController,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

HOP = ScreenHop(0x4A, "RIGHT", align_y=141)


def _snap(**kwargs) -> ZeldaSnapshot:
    fields = dict(
        mode=PLAY_MODE,
        level=0,
        screen=0x49,
        next_screen=0x49,
        link_x=120,
        link_y=141,
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


def _phantom_slot(slot: int = 3, x: int = 96, y: int = 141) -> ZeldaObject:
    """An emptied drop slot: ObjType still 0x60, ObjState cleared to 0x00.

    Indistinguishable from a live bomb drop by RAM alone — that is the point.
    """
    return ZeldaObject(
        slot=slot, type_id=BOMB_DROP_OBJECT_TYPE, x=x, y=y, facing=0, hp=0,
        state=BOMB_DROP_STATE,
    )


def _rupee(slot: int = 4, x: int = 144, y: int = 141) -> ZeldaObject:
    return ZeldaObject(
        slot=slot, type_id=RUPEE_DROP_OBJECT_TYPE, x=x, y=y, facing=0, hp=0,
        state=RUPEE_DROP_STATE,
    )


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
    snap = _snap(rupees=BOMB_SHOP_PRICE, bombs=1, objects=(_phantom_slot(x=144),))
    act = ctl._rupee_scoop(snap, HOP)
    assert act is not None
    assert act.reason == "scoop_bomb"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_rupee_scoop_stops_at_the_shop_price() -> None:
    ctl = ShopP7WalkController()
    snap = _snap(rupees=BOMB_SHOP_PRICE, bombs=0, objects=(_rupee(),))
    assert ctl._rupee_scoop(snap, HOP) is None


def test_heart_still_outranks_every_other_drop_when_hurt() -> None:
    ctl = ShopP7WalkController()
    heart = ZeldaObject(
        slot=2, type_id=RUPEE_DROP_OBJECT_TYPE, x=96, y=141, facing=0, hp=0,
        state=HEART_DROP_STATE,
    )
    snap = _snap(rupees=0, bombs=0, health=0x21, objects=(heart, _rupee(x=144)))
    act = ctl._rupee_scoop(snap, HOP)
    assert act is not None
    assert act.reason == "scoop_heart"
    assert list(act.action) == list(nes_action("LEFT"))
