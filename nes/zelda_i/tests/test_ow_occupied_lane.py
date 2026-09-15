"""Occupied-lane: hop travel cell in a body pad → parallel lane, not TTC.

The 0x49 RIGHT hop walks Link into an octorok whose velocity points away
(stand TTC > horizon). Catch it on the frame before contact. Do not peel
when already in-pad (that is evade / walk_or_swing).
"""

from __future__ import annotations

from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot


def _snap(
    *,
    screen: int = 0x49,
    x: int = 140,
    y: int = 141,
    health: int = 0x33,
    objects: tuple[ZeldaObject, ...] = (),
    mode: int = PLAY_MODE,
    level: int = 0,
) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=mode,
        level=level,
        screen=screen,
        next_screen=screen,
        link_x=x,
        link_y=y,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=health,
        triforce=0,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=objects,
    )


def _octorok(*, x: int = 160, y: int = 141, hp: int = 1) -> ZeldaObject:
    return ZeldaObject(slot=1, type_id=0x07, x=x, y=y, facing=0, hp=hp, state=0)


def _cardinal(act) -> str:
    if act.action == nes_idle_action():
        return "IDLE"
    for name in ("LEFT", "RIGHT", "UP", "DOWN"):
        if act.action == nes_action(name) or act.action == nes_action(name, "A"):
            return name
    return "OTHER"


def _lane_ctrl(**kwargs) -> OverworldPathController:
    hops = kwargs.pop("hops", (ScreenHop(0x4A, "RIGHT", align_y=141),))
    return OverworldPathController(
        hops=hops,
        farm_below_hearts=0,
        occupied_lane=True,
        **kwargs,
    )


def test_occupied_lane_peels_off_inbound_travel_pad() -> None:
    """RIGHT into an octorok on the hop y: first lane action is parallel y."""
    ctrl = _lane_ctrl(evade=True)
    act = ctrl.step(
        _snap(x=140, y=141, objects=(_octorok(x=160, y=141),))
    )
    direction = _cardinal(act)
    # Occupied-lane, not idle-in-pad. Parallel lane for a RIGHT hop is UP/DOWN.
    assert direction in {"UP", "DOWN"}, (direction, act.reason)
    assert direction != "RIGHT"
    assert "lane" in act.reason


def test_occupied_lane_far_body_is_not_a_false_positive() -> None:
    ctrl = _lane_ctrl(evade=True)
    act = ctrl.step(
        _snap(x=140, y=141, objects=(_octorok(x=220, y=141),))
    )
    assert _cardinal(act) == "RIGHT"
    assert "lane" not in act.reason


def test_occupied_lane_already_in_pad_yields() -> None:
    """Link 160,138 / octorok 160,130: TTC 0. Do not start a new peel."""
    ctrl = _lane_ctrl(evade=True)
    act = None
    for _ in range(8):
        act = ctrl.step(
            _snap(x=160, y=138, objects=(_octorok(x=160, y=130),))
        )
    assert act is not None
    assert "lane" not in act.reason
    assert ctrl.evade_reasons.get("evade_in_pad")
    assert ctrl.evades == 0


def test_occupied_lane_with_evade_off_still_observes() -> None:
    """evade=False must not crash; the tracker still feeds occupied-lane."""
    ctrl = _lane_ctrl(evade=False)
    assert ctrl.evade is False
    act = ctrl.step(
        _snap(x=140, y=141, objects=(_octorok(x=160, y=141),))
    )
    assert ctrl._tracker is not None
    assert any(t.slot == 1 for t in ctrl._tracked)
    direction = _cardinal(act)
    assert direction in {"UP", "DOWN", "RIGHT", "IDLE"}
    assert act.reason
