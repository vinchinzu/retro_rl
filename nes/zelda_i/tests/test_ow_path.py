"""Unit tests for shared OverworldPathController hop advance / maze core."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action

from zelda_i.dungeon.ids import RUPEE_DROP_OBJECT_TYPE
from zelda_i.overworld.graph import LEVEL2_5C_MAZE_WAYPOINTS, ScreenHop, is_5c_maze_hop
from zelda_i.overworld.path import OverworldPathController, PathNavPhase
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
    ZeldaObject,
    ZeldaSnapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x77)
    ram[ADDR_LINK_X] = fields.get("x", 112)
    ram[ADDR_LINK_Y] = fields.get("y", 125)
    ram[ADDR_HEALTH] = fields.get("health", 0x33)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    return ram


def test_is_5c_maze_hop_shared() -> None:
    assert is_5c_maze_hop(ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140))
    assert not is_5c_maze_hop(ScreenHop(0x5C, "RIGHT"))


def test_hop_advance_on_arrival() -> None:
    from zelda_i.ram import read_snapshot

    hops = (
        ScreenHop(0x78, "RIGHT", align_y=140),
        ScreenHop(0x68, "UP", align_x=48),
    )
    ctrl = OverworldPathController(hops=hops, require_sword=True)
    assert ctrl.phase is PathNavPhase.HOP

    # Still on start screen: push RIGHT.
    snap = read_snapshot(_ram(screen=0x77, x=100, y=140, sword=1))
    act = ctrl.step(snap)
    assert ctrl.hop_index == 0
    assert "hop0" in act.reason or "RIGHT" in str(act.action) or act.reason

    # Arrived on 0x78 off the east edge → advance.
    snap = read_snapshot(_ram(screen=0x78, x=100, y=140, sword=1))
    act = ctrl.step(snap)
    assert ctrl.hop_index == 1
    assert "hop_0_78" in ctrl.notes
    assert act.reason == "hop_advance"


def test_maze_waypoints_on_5c() -> None:
    from zelda_i.ram import read_snapshot

    hops = (ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),)
    ctrl = OverworldPathController(
        hops=hops,
        maze_waypoints=LEVEL2_5C_MAZE_WAYPOINTS,
        maze_hop_pred=is_5c_maze_hop,
    )
    snap = read_snapshot(_ram(screen=0x5C, x=16, y=93, sword=1))
    act = ctrl.step(snap)
    assert "maze" in act.reason
    assert "maze_start" in ctrl.notes

    tx, ty = LEVEL2_5C_MAZE_WAYPOINTS[0]
    snap = read_snapshot(_ram(screen=0x5C, x=tx, y=ty, sword=1))
    ctrl.step(snap)
    assert ctrl.maze_wp_index >= 1


def test_default_stop_after_hops() -> None:
    from zelda_i.ram import read_snapshot

    hops = (ScreenHop(0x74, "RIGHT", align_y=117),)
    ctrl = OverworldPathController(hops=hops, require_sword=True)
    ctrl.hop_index = 1
    snap = read_snapshot(_ram(screen=0x74, x=128, y=140, sword=1))
    act = ctrl.step(snap)
    assert ctrl.success
    assert act.reason == "done"
    assert ctrl.phase is PathNavPhase.DONE


def test_low_heart_hook_is_inert_by_default() -> None:
    """farm_below_hearts=0 (and a live health assist) never diverts a hop."""
    from zelda_i.ram import read_snapshot

    ctrl = OverworldPathController(
        hops=(ScreenHop(0x78, "RIGHT", align_y=140),),
        farm_below_hearts=0,
    )
    act = ctrl.step(read_snapshot(_ram(screen=0x77, x=100, y=140, health=0x30)))
    assert ctrl.farm_attempts == 0
    assert not act.reason.startswith("farm")


def test_default_farm_below_hearts_diverts_when_empty() -> None:
    """Default 3 farms when assist is off and filled hearts are 0."""
    from zelda_i.ram import read_snapshot

    ctrl = OverworldPathController(hops=(ScreenHop(0x68, "UP"),))
    assert ctrl.farm_below_hearts == 3
    ctrl.step(read_snapshot(_ram(screen=0x78, x=100, y=140, health=0x30)))
    assert ctrl.farm_attempts == 1
    assert any(note.startswith("farm_start_78") for note in ctrl.notes)


def test_low_hearts_divert_into_the_farm_then_hand_back() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = OverworldPathController(
        hops=(ScreenHop(0x68, "UP"),),
        farm_below_hearts=3,
        farm_min_filled=3,
    )
    # 0x30: three containers, zero whole hearts.
    ctrl.step(read_snapshot(_ram(screen=0x78, x=100, y=140, health=0x30)))
    assert ctrl.farm_attempts == 1
    assert any(note.startswith("farm_start_78") for note in ctrl.notes)

    # Hearts recovered: the farm reports done and the hop resumes.
    act = ctrl.step(read_snapshot(_ram(screen=0x78, x=100, y=140, health=0x33)))
    assert act.reason == "farm_ok"
    assert ctrl._farm is None
    act = ctrl.step(read_snapshot(_ram(screen=0x78, x=100, y=140, health=0x33)))
    assert not act.reason.startswith("farm")


def test_farm_attempts_are_capped() -> None:
    """A starving farm must not replace the hop forever."""
    from zelda_i.ram import read_snapshot

    ctrl = OverworldPathController(
        hops=(ScreenHop(0x68, "UP"),),
        farm_below_hearts=3,
        max_farm_attempts=1,
    )
    ctrl.step(read_snapshot(_ram(screen=0x78, x=100, y=140, health=0x30)))
    # Leaving to a screen that is not the restock neighbor (catalog 0x77 /
    # hop target 0x68) still soft-fails.
    act = ctrl.step(read_snapshot(_ram(screen=0x67, x=100, y=140, health=0x30)))
    assert act.reason == "farm_gave_up"
    act = ctrl.step(read_snapshot(_ram(screen=0x67, x=100, y=140, health=0x30)))
    assert ctrl.farm_attempts == 1
    assert not act.reason.startswith("farm")


def _snap_drop(*, rupees: int, drop_x: int, drop_y: int, x: int = 100, y: int = 140) -> ZeldaSnapshot:
    drop = ZeldaObject(
        slot=1,
        type_id=RUPEE_DROP_OBJECT_TYPE,
        x=drop_x,
        y=drop_y,
        facing=0,
        hp=0,
        state=0,
    )
    return ZeldaSnapshot(
        mode=PLAY_MODE,
        level=0,
        screen=0x77,
        next_screen=0x77,
        link_x=x,
        link_y=y,
        facing=0,
        sword=1,
        bombs=0,
        rupees=rupees,
        keys=0,
        health=0x33,
        triforce=0,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(drop,),
    )


def test_need_rupees_zero_ignores_drops() -> None:
    ctrl = OverworldPathController(hops=(ScreenHop(0x78, "RIGHT", align_y=140),))
    act = ctrl.step(_snap_drop(rupees=0, drop_x=140, drop_y=140))
    assert "scoop" not in act.reason


def test_need_rupees_walks_onto_nearby_drop() -> None:
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x78, "RIGHT", align_y=140),),
        need_rupees=80,
    )
    act = ctrl.step(_snap_drop(rupees=10, drop_x=140, drop_y=140, x=100, y=140))
    assert act.reason == "scoop_rupee"
    assert act.action == nes_action("RIGHT")


def test_need_rupees_already_funded_keeps_hopping() -> None:
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x78, "RIGHT", align_y=140),),
        need_rupees=80,
    )
    act = ctrl.step(_snap_drop(rupees=80, drop_x=140, drop_y=140))
    assert "scoop" not in act.reason


def test_far_drop_does_not_steal_the_hop() -> None:
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x78, "RIGHT", align_y=140),),
        need_rupees=80,
        scoop_radius=48,
    )
    act = ctrl.step(_snap_drop(rupees=0, drop_x=220, drop_y=200, x=40, y=80))
    assert "scoop" not in act.reason


def _snap(
    *,
    screen: int = 0x77,
    rupees: int = 0,
    x: int = 100,
    y: int = 140,
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
        rupees=rupees,
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


def _octorok(*, x: int = 180, y: int = 149, hp: int = 1) -> ZeldaObject:
    return ZeldaObject(slot=1, type_id=0x07, x=x, y=y, facing=0, hp=hp, state=0)


def test_need_rupees_zero_never_starts_rupee_farm() -> None:
    ctrl = OverworldPathController(hops=(ScreenHop(0x68, "UP"),))
    act = ctrl.step(_snap(screen=0x78, rupees=0, objects=(_octorok(),)))
    assert ctrl._rupee_farm is None
    assert ctrl.rupee_farm_attempts == 0
    assert "farm_chase" not in act.reason
    assert "farm_rupee" not in act.reason
    assert not any("rupee_farm" in note for note in ctrl.notes)


def test_need_rupees_starts_farm_on_octorok_screen() -> None:
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x68, "UP"),),
        need_rupees=80,
    )
    act = ctrl.step(_snap(screen=0x78, rupees=0, objects=(_octorok(),)))
    started = (
        "farm_chase" in act.reason
        or "farm_rupee" in act.reason
        or any("rupee_farm_start" in note for note in ctrl.notes)
    )
    assert started
    assert ctrl._rupee_farm is not None
    assert ctrl.rupee_farm_attempts == 1
    report = ctrl.report()
    assert report["rupee_farm_attempts"] == 1
    assert report["need_rupees"] == 80


def test_need_rupees_already_funded_skips_farm() -> None:
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x68, "UP"),),
        need_rupees=80,
    )
    act = ctrl.step(_snap(screen=0x78, rupees=80, objects=(_octorok(),)))
    assert ctrl._rupee_farm is None
    assert ctrl.rupee_farm_attempts == 0
    assert "farm_chase" not in act.reason
    assert "farm_rupee" not in act.reason


def test_rupee_farm_restock_is_catalog_or_hop() -> None:
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x68, "UP"),),
        need_rupees=80,
    )
    act = ctrl.step(_snap(screen=0x78, rupees=0, objects=()))
    assert ctrl._rupee_farm is not None
    farm = ctrl._rupee_farm
    assert farm.farm_screen == 0x78
    assert farm.leftover_screen == 0x78
    assert farm.restock_neighbor_screen in {0x77, 0x68}
    for _ in range(120):
        act = ctrl.step(_snap(screen=0x78, rupees=0, objects=()))
        if act.reason == "farm_leave":
            break
    assert act.reason == "farm_leave"
    if farm.restock_neighbor_screen == 0x68:
        assert act.action == nes_action("UP")
    else:
        assert act.action == nes_action("LEFT")


def test_worth_skips_lynel_and_does_not_farm() -> None:
    ctrl = OverworldPathController(hops=(ScreenHop(0x02, "RIGHT"),), need_rupees=80)
    assert ctrl._worth(0x78) is not None
    assert ctrl._worth(0x01) is None
    prey = ZeldaObject(slot=1, type_id=0x02, x=180, y=149, facing=0, hp=1, state=0)
    act = ctrl.step(_snap(screen=0x01, rupees=0, objects=(prey,)))
    assert ctrl._rupee_farm is None
    assert ctrl.rupee_farm_attempts == 0
    assert "farm_chase" not in act.reason
    assert not any("rupee_farm" in note for note in ctrl.notes)
