"""Unit tests for shared OverworldPathController hop advance / maze core."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.dungeon import ids as dungeon_ids
from zelda_i.dungeon.ids import RUPEE_DROP_OBJECT_TYPE, RUPEE_DROP_STATE
from zelda_i.overworld.graph import (
    LEVEL2_5C_MAZE_WAYPOINTS,
    LEVEL2_DOOR_HOPS,
    ScreenHop,
    is_5c_maze_hop,
)
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


def test_heart_farm_skips_leever_screen() -> None:
    """0x48 leevers: do not chase north. Natural L2 died farm_chase at (186,93)."""
    from zelda_i.ram import read_snapshot

    ctrl = OverworldPathController(
        hops=(ScreenHop(0x58, "DOWN", align_x=112),),
        farm_below_hearts=3,
    )
    act = ctrl.step(read_snapshot(_ram(screen=0x48, x=112, y=140, health=0x32)))
    assert ctrl.farm_attempts == 0
    assert ctrl._farm is None
    assert "farm" not in str(act.reason)
    assert not any(note.startswith("farm_start_48") for note in ctrl.notes)


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
        state=RUPEE_DROP_STATE,
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


def test_scoop_rupees_walks_onto_nearby_drop_when_need_is_zero() -> None:
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x78, "RIGHT", align_y=140),),
        scoop_rupees=True,
        need_rupees=0,
    )
    act = ctrl.step(_snap_drop(rupees=0, drop_x=140, drop_y=140, x=100, y=140))
    assert act.reason == "scoop_rupee"
    assert act.action == nes_action("RIGHT")


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
    # The leave is an occupancy walk to the edge lane (``rupee_farm._edge_walk``
    # / ``heart_farm.LEAVE_GOALS``), not a bare directional hold, so the first
    # frame may step onto the y=141 lane instead of pressing the travel button.
    # What must hold is the aim: the walk targets the restock neighbour's edge,
    # and at the edge cell the raw direction comes back (the scroll needs the
    # button, not a path).
    from zelda_i.overworld.heart_farm import LEAVE_GOALS

    direction = "UP" if farm.restock_neighbor_screen == 0x68 else "LEFT"
    assert farm.restock_direction == direction
    gx, gy = LEAVE_GOALS[direction]
    edge = ctrl.step(_snap(screen=0x78, rupees=0, x=gx, y=gy))
    assert edge.reason.startswith("farm_leave")
    assert edge.action == nes_action(direction)


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


def _cardinal(act) -> str:
    if act.action == nes_idle_action():
        return "IDLE"
    for name in ("LEFT", "RIGHT", "UP", "DOWN"):
        if act.action == nes_action(name) or act.action == nes_action(name, "A"):
            return name
    return "OTHER"


def test_east_mouth_4c_never_pushes_up_off_column() -> None:
    """Leftover (240,157) on 0x4C: LEFT or y-peel, never hop UP at x≥232."""
    from zelda_i.ram import read_snapshot

    hop = LEVEL2_DOOR_HOPS[10]
    assert hop.target == 0x3C and hop.direction == "UP" and hop.align_x == 112
    ctrl = OverworldPathController(hops=LEVEL2_DOOR_HOPS, farm_below_hearts=0)
    ctrl.hop_index = 10
    snap = read_snapshot(_ram(screen=0x4C, x=240, y=157, sword=1))

    first = ctrl.step(snap)
    assert _cardinal(first) == "LEFT"
    assert _cardinal(first) != "UP"
    assert not str(first.reason).endswith("_wait")

    # Frozen leftover: occupancy miss → block → y-peel. Never RIGHT (would
    # scroll to 0x4D). Never unstick_wait. First action is not hop UP.
    peels = []
    for _ in range(16):
        act = ctrl.step(snap)
        direction = _cardinal(act)
        peels.append(direction)
        assert direction != "RIGHT"
        assert not str(act.reason).endswith("_wait")
        assert direction in {"LEFT", "DOWN", "UP", "IDLE"}
    assert any(d in {"LEFT", "DOWN", "UP"} for d in peels)


def test_align_x_column_still_pushes_up() -> None:
    """On the door column, occupancy hands back to align-and-push UP."""
    from zelda_i.ram import read_snapshot

    ctrl = OverworldPathController(
        hops=(ScreenHop(0x3C, "UP", align_x=112),),
        farm_below_hearts=0,
    )
    act = ctrl.step(read_snapshot(_ram(screen=0x4C, x=112, y=157, sword=1)))
    assert _cardinal(act) == "UP"


def test_farm_below_hearts_zero_still_never_farms_but_scoops_a_heart(
    monkeypatch,
) -> None:
    """farm_below_hearts=0 never farms; a heart 20px ahead is scoop_heart."""
    from zelda_i.overworld import path as path_mod

    heart_type = int(getattr(dungeon_ids, "HEART_DROP_OBJECT_TYPE", 0xFE))
    monkeypatch.setattr(path_mod, "HEART_FAIRY_DROP_TYPES", frozenset({heart_type}))
    drop = ZeldaObject(
        slot=1,
        type_id=heart_type,
        x=120,
        y=140,
        facing=0,
        hp=0,
        state=int(getattr(dungeon_ids, "HEART_DROP_STATE", 0x22)),
    )
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x78, "RIGHT", align_y=140),),
        farm_below_hearts=0,
    )
    act = ctrl.step(
        _snap(screen=0x77, x=100, y=140, health=0x30, objects=(drop,))
    )
    assert ctrl.farm_attempts == 0
    assert ctrl._farm is None
    assert act.reason == "scoop_heart"
    assert act.action == nes_action("RIGHT")
    assert "hop0" not in act.reason


# --- Opt-in reactive threat step (rr-ps7.3) --------------------------------


def _evade_ctrl(
    hops: tuple[ScreenHop, ...] = (ScreenHop(0x4A, "RIGHT", align_y=141),),
) -> OverworldPathController:
    return OverworldPathController(hops=hops, farm_below_hearts=0, evade=True)


def test_evade_is_off_by_default_and_never_tracks() -> None:
    ctrl = OverworldPathController(
        hops=(ScreenHop(0x4A, "RIGHT", align_y=141),), farm_below_hearts=0
    )
    assert ctrl.evade is False
    for _ in range(6):
        ctrl.step(
            _snap(screen=0x49, x=100, y=141, objects=(_octorok(x=104, y=141),))
        )
    assert ctrl._tracker is None
    assert ctrl._tracked == ()
    assert ctrl.evades == 0


def test_only_the_level2_path_controller_opts_into_evade() -> None:
    from zelda_i.level1.arrow_shop import make_arrow_shop_controller
    from zelda_i.level2.overworld import OverworldToLevel2Controller
    from zelda_i.level3.overworld import OverworldToLevel3Controller
    from zelda_i.level5.overworld import OverworldToLevel5Controller
    from zelda_i.level6.overworld import OverworldToLevel6Controller
    from zelda_i.level8.overworld import OverworldToLevel8Controller

    assert OverworldToLevel2Controller().evade is True
    for other in (
        OverworldPathController(),
        OverworldToLevel3Controller(),
        OverworldToLevel5Controller(),
        OverworldToLevel6Controller(),
        OverworldToLevel8Controller(),
        make_arrow_shop_controller(),
    ):
        assert other.evade is False, type(other).__name__


def test_evade_tracker_is_fed_every_frame() -> None:
    ctrl = _evade_ctrl()
    for k in range(6):
        ctrl.step(
            _snap(screen=0x49, x=100, y=141, objects=(_octorok(x=200 - k, y=141),))
        )
    tracked = [t for t in ctrl._tracked if t.slot == 1]
    assert tracked and tracked[0].vx < 0


def test_evade_yields_the_frame_when_standing_is_safe() -> None:
    ctrl = _evade_ctrl()
    act = None
    for _ in range(8):
        act = ctrl.step(
            _snap(screen=0x49, x=100, y=141, objects=(_octorok(x=220, y=60),))
        )
    assert act is not None and act.reason.startswith("hop0")
    assert ctrl.evades == 0


def test_evade_steps_off_an_inbound_body() -> None:
    ctrl = _evade_ctrl()
    reasons = []
    for k in range(24):
        reasons.append(
            ctrl.step(
                _snap(
                    screen=0x49,
                    x=100,
                    y=141,
                    objects=(_octorok(x=150 - k, y=141),),
                )
            ).reason
        )
    assert any(r.startswith("evade_evade") for r in reasons), reasons
    assert ctrl.evades > 0


def test_evade_yields_to_the_sword_inside_the_contact_pad() -> None:
    """No sidestep clears a pad Link is already in; the peel belongs to a room."""
    ctrl = _evade_ctrl()
    act = None
    for _ in range(8):
        act = ctrl.step(
            _snap(screen=0x49, x=100, y=141, objects=(_octorok(x=104, y=141),))
        )
    assert ctrl.evade_reasons.get("evade_in_pad")
    assert ctrl.evades == 0
    assert act is not None and not act.reason.startswith("evade_evade")


def test_evade_never_steps_across_a_scroll_line() -> None:
    ctrl = _evade_ctrl()
    assert ctrl._evade_blocked_dirs(_snap(screen=0x48, x=112, y=205)) == {"DOWN"}
    assert ctrl._evade_blocked_dirs(_snap(screen=0x48, x=240, y=141)) == {"RIGHT"}
    assert ctrl._evade_blocked_dirs(_snap(screen=0x48, x=10, y=70)) == {
        "LEFT",
        "UP",
    }
    assert ctrl._evade_blocked_dirs(_snap(screen=0x48, x=112, y=141)) == set()


def test_evade_resets_with_the_controller() -> None:
    ctrl = _evade_ctrl()
    for k in range(24):
        ctrl.step(
            _snap(screen=0x49, x=100, y=141, objects=(_octorok(x=150 - k, y=141),))
        )
    assert ctrl.evades > 0
    ctrl.reset()
    assert ctrl.evades == 0
    assert ctrl._tracker is None
    assert ctrl._tracked == ()
    assert ctrl.evade_reasons == {}


# --- Wall-column x-align is opt-in per hop (``align_x_at_wall``) --------- #


def _cardinal(act) -> str:
    if act.action == nes_idle_action():
        return "IDLE"
    for name in ("LEFT", "RIGHT", "UP", "DOWN"):
        if act.action == nes_action(name) or act.action == nes_action(name, "A"):
            return name
    return "OTHER"


def test_opted_in_0x48_down_hop_still_x_aligns_at_the_south_wall() -> None:
    """LEVEL2_PATH_HOPS[2] (0x48→0x58) keeps align_x=120 at y>=205."""
    from zelda_i.overworld.graph import LEVEL2_PATH_HOPS
    from zelda_i.ram import read_snapshot

    hop = LEVEL2_PATH_HOPS[2]
    assert hop.target == 0x58 and hop.direction == "DOWN"
    assert hop.align_x == 120 and hop.align_x_at_wall

    ctrl = OverworldPathController(hops=LEVEL2_PATH_HOPS, farm_below_hearts=0)
    ctrl.hop_index = 2
    act = ctrl.step(read_snapshot(_ram(screen=0x48, x=112, y=205, sword=1)))
    assert _cardinal(act) == "RIGHT"
    assert _cardinal(act) != "DOWN"


def test_plain_down_hop_does_not_x_align_at_the_south_wall() -> None:
    """Default (no opt-in) restores the ``80 < y < 205`` interior band.

    The post-L6 leftover on 0x22 is (120,221) with ``align_x=112``, and 112
    is ``L6_CAVE_MOUTH_X``: a blanket wall rule strafed Link LEFT back into
    the Level 6 mouth. DOWN is the measured travel.
    """
    from zelda_i.ram import read_snapshot

    hop = ScreenHop(0x32, "DOWN", align_x=112)
    assert not hop.align_x_at_wall
    ctrl = OverworldPathController(hops=(hop,), farm_below_hearts=0)
    act = ctrl.step(read_snapshot(_ram(screen=0x22, x=120, y=221, sword=1)))
    assert _cardinal(act) == "DOWN"
    assert _cardinal(act) != "LEFT"


def test_post_l6_bait_hop_does_not_opt_into_wall_align() -> None:
    """The production 0x22 DOWN hop must stay off the wall rule."""
    from zelda_i.level7.overworld import L6_CAVE_MOUTH_X, POST_L6_TO_BAIT_HOPS

    hop = POST_L6_TO_BAIT_HOPS[0]
    assert hop.direction == "DOWN" and hop.align_x == L6_CAVE_MOUTH_X
    assert not hop.align_x_at_wall


def test_align_and_push_wall_flag_is_off_by_default() -> None:
    """Function level: same pose, both answers, flag is the only difference."""
    from zelda_i.overworld.common import align_and_push
    from zelda_i.ram import read_snapshot

    snap = read_snapshot(_ram(screen=0x48, x=112, y=205, sword=1))
    default = align_and_push(snap, direction="DOWN", reason="hop0", align_x=120)
    assert _cardinal(default) == "DOWN"
    assert not default.reason.endswith("_ax")

    opted = align_and_push(
        snap, direction="DOWN", reason="hop0", align_x=120, align_x_at_wall=True
    )
    assert _cardinal(opted) == "RIGHT"
    assert opted.reason.endswith("_ax")

    # North-wall mirror, and a horizontal hop never gets the wall rule.
    north = read_snapshot(_ram(screen=0x48, x=112, y=80, sword=1))
    assert _cardinal(
        align_and_push(north, direction="UP", reason="hop0", align_x=120)
    ) == "UP"
    assert _cardinal(
        align_and_push(
            north, direction="UP", reason="hop0", align_x=120, align_x_at_wall=True
        )
    ) == "RIGHT"
    side = read_snapshot(_ram(screen=0x48, x=128, y=205, sword=1))
    assert _cardinal(
        align_and_push(
            side, direction="RIGHT", reason="hop0", align_x=120, align_x_at_wall=True
        )
    ) == "RIGHT"


# --- Occupied lane keeps the hop's align_y (parallel lane, not abandon) -- #


def _lane_ctrl(
    hops: tuple[ScreenHop, ...] = (ScreenHop(0x4A, "RIGHT", align_y=141),),
) -> OverworldPathController:
    """Occupied-lane only: isolate the lane rule from the evader."""
    return OverworldPathController(
        hops=hops, farm_below_hearts=0, occupied_lane=True, evade=False
    )


def _walk_lane(ctrl: OverworldPathController, *, x: int, y: int, objects):
    """Feed poses back at ~1px/frame so a peel can actually land."""
    acts: list[str] = []
    for _ in range(64):
        act = ctrl.step(_snap(screen=0x49, x=x, y=y, objects=objects))
        step = _cardinal(act)
        acts.append(step)
        if step == "DOWN":
            y += 1
        elif step == "UP":
            y -= 1
        else:
            break
    return acts, x, y


def test_occupied_lane_peels_parallel_instead_of_dropping_align_y() -> None:
    """Off the hop row with the hop row blocked: steer to a lane, then travel.

    ``ScreenHop(0x4A, "RIGHT", align_y=141)`` with an octorok parked on
    y=141: the old ladder pushed RIGHT on whatever row Link stood on, so he
    crossed into 0x4A on an unmeasured pose. The lane he takes now is a
    body-pad offset from 141, not a drifted row.
    """
    from zelda_i.dungeon.threat import MIN_DODGE_BODY

    ctrl = _lane_ctrl()
    acts, _x, y = _walk_lane(ctrl, x=140, y=120, objects=(_octorok(x=160, y=141),))
    assert acts[0] in {"UP", "DOWN"}, acts[:4]
    assert acts[0] != "RIGHT"
    assert acts[-1] == "RIGHT", acts[-4:]
    assert y != 120, "Link stayed on his drifted row"
    assert abs(y - 141) <= MIN_DODGE_BODY, (y, "lane is not a step off align_y")


def test_occupied_lane_picks_the_open_side_of_the_blocker() -> None:
    """Both lanes are candidates; only the south one is open, so peel south.

    Link starts *north* of the hop row, so a side rule that just follows
    Link would pick the blocked north lane.
    """
    hazards = (_octorok(x=160, y=141), _octorok(x=160, y=125))
    ctrl = _lane_ctrl()
    acts, _x, y = _walk_lane(ctrl, x=140, y=120, objects=hazards)
    assert acts[0] == "DOWN", acts[:4]
    assert acts[-1] == "RIGHT", acts[-4:]
    assert y > 141, (y, "peel ended north of the blocked lane")


def test_occupied_lane_on_row_still_peels_perpendicular() -> None:
    """Link on the hop row, blocked: unchanged behaviour (UP/DOWN, not RIGHT)."""
    ctrl = _lane_ctrl()
    act = ctrl.step(_snap(screen=0x49, x=140, y=141, objects=(_octorok(x=160, y=141),)))
    assert _cardinal(act) in {"UP", "DOWN"}
    assert "lane" in act.reason


def test_occupied_lane_clear_hop_row_is_not_a_lane_action() -> None:
    """No blocker on the hop row → the lane rule yields to align_and_push."""
    ctrl = _lane_ctrl()
    act = ctrl.step(_snap(screen=0x49, x=140, y=120, objects=(_octorok(x=220, y=141),)))
    assert "lane" not in act.reason


def test_occupied_lane_steer_cap_falls_back_to_the_old_ladder() -> None:
    """A body camping the hop lane cannot hold the hop forever.

    Link is pinned (the pose never answers the peel), so after
    ``_OCCUPIED_LANE_STEER_CAP`` frames the rule drops back to travelling on
    his own clear row, then to the stand cap, exactly as before.
    """
    from zelda_i.overworld.path import _OCCUPIED_LANE_STEER_CAP

    ctrl = _lane_ctrl()
    reasons = []
    for _ in range(_OCCUPIED_LANE_STEER_CAP + 2):
        act = ctrl.step(
            _snap(screen=0x49, x=140, y=120, objects=(_octorok(x=160, y=141),))
        )
        reasons.append((_cardinal(act), act.reason))
    assert reasons[0][0] in {"UP", "DOWN"}
    assert reasons[-1][0] == "RIGHT"
    assert reasons[-1][1].endswith("_row")


def test_occupied_lane_stand_cap_still_yields_the_hop() -> None:
    """Boxed in on every lane: short stand, then hand the frame back."""
    from zelda_i.overworld.path import _OCCUPIED_LANE_STAND_CAP

    ctrl = _lane_ctrl()
    wall = tuple(
        ZeldaObject(slot=i + 1, type_id=0x07, x=160, y=y, facing=0, hp=1, state=0)
        for i, y in enumerate(range(64, 232, 8))
    )
    reasons = []
    for _ in range(_OCCUPIED_LANE_STAND_CAP + 2):
        act = ctrl.step(_snap(screen=0x49, x=140, y=141, objects=wall))
        reasons.append(act.reason)
    stands = [i for i, r in enumerate(reasons) if r.endswith("_stand")]
    yields_ = [i for i, r in enumerate(reasons) if "lane" not in r]
    assert stands, reasons
    assert yields_, reasons
    assert min(yields_) > max(stands[:_OCCUPIED_LANE_STAND_CAP]), reasons
    assert min(yields_) <= _OCCUPIED_LANE_STAND_CAP + 1, reasons


# ------------------------------------------------- shot over the sword ---


def _fireball_snap(*, shot_x: int, leever_x: int, link_y: int = 141) -> ZeldaSnapshot:
    """Link on 0x7C between a leever at contact and inbound Zora spit."""
    from zelda_i.dungeon.ids import FIREBALL_OBJECT_TYPE

    return ZeldaSnapshot(
        mode=PLAY_MODE, level=0, screen=0x7C, next_screen=0x7C,
        link_x=120, link_y=link_y, facing=0x01, sword=1, bombs=0, rupees=0, keys=0,
        health=0x22, heart_partial=0xFF, triforce=0, compass=0, dialog_timer=0,
        colliding_tile=0, room_item_id=0, room_all_dead=0, room_obj_count=0,
        cur_opened_doors=0, open_doorway_mask=0,
        objects=(
            # 0x0E leever in the blade box (reach 20) but outside the
            # contact pad (16): exactly the geometry ``hunter.striking``
            # claims a frame for.
            ZeldaObject(slot=1, type_id=0x0E, x=leever_x, y=141, facing=0x02,
                        hp=0x20, state=1),
            ZeldaObject(slot=10, type_id=FIREBALL_OBJECT_TYPE, x=shot_x, y=141,
                        facing=0x0A, hp=0, state=0x10),
        ),
    )


_DRIVE_STEP = {"UP": -1, "DOWN": 1}


def _drive_shot(controller, *, frames: int = 8, step: int = 2) -> str | None:
    """Walk the spit west toward Link a frame at a time, return the last reason.

    Link *moves* when the controller says to move: a fixture that pins him in
    place is a wall as far as ``_spit_duck`` is concerned, and the rung would
    write its own escape off after three frames (which is exactly what it is
    supposed to do to the 0x7B rock it kept pressing UP into).
    """
    reason = None
    link_y = 141
    for i in range(frames):
        snap = _fireball_snap(shot_x=200 - step * i, leever_x=139, link_y=link_y)
        controller._observe_threats(snap)
        controller.hunter.observe(snap)
        act = controller._threat_action(snap, None)
        reason = None if act is None else act.reason
        if act is not None:
            link_y += _DRIVE_STEP.get(_direction_of(act), 0)
    return reason


def _direction_of(act) -> str | None:
    from retro_harness.nes import nes_action

    for name in ("UP", "DOWN", "LEFT", "RIGHT"):
        if list(act.action) == list(nes_action(name)):
            return name
    return None


def test_an_inbound_spit_outranks_a_body_in_the_blade_box() -> None:
    """``evade_yield_to_sword`` was 687 of 6384 walk frames (``zhit1``) and
    the leever screens keep a body in the box almost continuously, so the
    Zora fired into an evader that had been handed off for the whole window.
    0x55 is neither killable nor small-shield blockable: the swing cannot
    answer it, so ``_spit_duck`` takes the frame above the yield."""
    from zelda_i.overworld.shop_p7 import ShopP7WalkController

    ctl = ShopP7WalkController()
    # 30 frames leaves the spit 22 px out and still closing; past ~39 it has
    # gone by Link, and a shot going away is geometry, not a threat.
    reason = _drive_shot(ctl, frames=30, step=2)
    assert reason == "spit_duck"
    assert ctl.spit_ducks > 0
    # The two velocity samples the tracker needs, and nothing after them:
    # once the shot is closing, the sword never gets the window back.
    assert ctl.evade_reasons.get("evade_yield_to_sword", 0) <= 3


def test_the_spit_duck_leaves_the_shot_row_not_the_shot_lane() -> None:
    """The shot flies west down Link's own row, so the escape is vertical.

    The old ``answer_projectile`` crossed the *travel* axis and flipped at a
    wall, which is how Link walked east into the 0x7C spit at x=16.
    """
    from retro_harness.nes import nes_action
    from zelda_i.overworld.shop_p7 import ShopP7WalkController

    ctl = ShopP7WalkController()
    link_y = 141
    for i in range(6):
        snap = _fireball_snap(shot_x=200 - 2 * i, leever_x=139, link_y=link_y)
        ctl._observe_threats(snap)
        ctl.hunter.observe(snap)
        act = ctl._threat_action(snap, None)
        if act is not None and list(act.action) == list(nes_action("DOWN")):
            link_y += 1
    assert act is not None and act.reason == "spit_duck"
    assert list(act.action) == list(nes_action("DOWN"))
    assert link_y > 141  # off the shot's row, not along its lane


def test_a_duck_that_moves_nobody_is_written_off_as_wall() -> None:
    """Live 0x7B (``zhit2`` f=4684): eight frames of UP at (48, 133) against a
    rock while a leever closed from 12 px to 8 — the walk's first hit, taken
    at full health with the streak on 10. ``_EVADE_BOUNDS`` is the scroll
    rectangle and cannot see a rock, so the wall has to be *measured*: three
    frames that move Link nowhere retire that direction for the screen."""
    from zelda_i.overworld.shop_p7 import ShopP7WalkController

    ctl = ShopP7WalkController()
    seen = []
    for i in range(12):
        # Link never moves: every direction this fixture offers is wall.
        snap = _fireball_snap(shot_x=200 - 2 * i, leever_x=139, link_y=141)
        ctl._observe_threats(snap)
        ctl.hunter.observe(snap)
        act = ctl._threat_action(snap, None)
        seen.append(None if act is None else act.reason)
    assert any(r == "spit_duck" for r in seen), seen
    # Both sides of the bearing are wall, so the rung stops claiming frames
    # instead of pressing one of them forever.
    assert seen[-1] != "spit_duck", seen
    walls = {direction for direction, _cx, _cy in ctl._duck_walls}
    assert {"DOWN", "UP"} <= walls, ctl._duck_walls
    assert any(n.startswith("duck_wall_7c_") for n in ctl.notes), ctl.notes


def test_a_body_in_the_blade_box_still_owns_a_quiet_frame() -> None:
    """The yield is the right default: with no shot inbound the hunt keeps
    the frame and swings."""
    from zelda_i.overworld.shop_p7 import ShopP7WalkController

    ctl = ShopP7WalkController()
    snap = ZeldaSnapshot(
        mode=PLAY_MODE, level=0, screen=0x7C, next_screen=0x7C,
        link_x=120, link_y=141, facing=0x01, sword=1, bombs=0, rupees=0, keys=0,
        health=0x22, heart_partial=0xFF, triforce=0, compass=0, dialog_timer=0,
        colliding_tile=0, room_item_id=0, room_all_dead=0, room_obj_count=0,
        cur_opened_doors=0, open_doorway_mask=0,
        objects=(
            ZeldaObject(slot=1, type_id=0x0E, x=139, y=141, facing=0x02,
                        hp=0x20, state=1),
        ),
    )
    for _ in range(6):
        ctl._observe_threats(snap)
        ctl.hunter.observe(snap)
        assert ctl._threat_action(snap, None) is None
    assert ctl.evade_reasons.get("evade_yield_to_sword", 0) > 0
    assert "evade_shot_over_sword" not in ctl.evade_reasons


# --------------------------------------------- the hop grinds to a push ---


def _grind_ctl():
    from zelda_i.overworld.shop_p7 import ShopP7WalkController

    return ShopP7WalkController()


def test_a_hop_that_spends_its_screen_budget_stops_being_clever() -> None:
    """The optional rungs each have a local cap and none of them compose:
    ``pre_l1_wedge1`` alternated the occupied-lane steer and the plain push
    one frame each for 24914 frames on 0x7C (``hop`` 12459, ``hop_lane``
    12455), because a steer that picks the travel direction resets the steer
    counter and ``track_stuck`` sees a Link who is moving."""
    from zelda_i.ram import read_snapshot

    ctl = _grind_ctl()
    snap = read_snapshot(_ram(screen=0x7C, x=120, y=141, health=0x22))
    for _ in range(ctl.hop_screen_max_frames):
        assert ctl._grinding(snap) is False
    assert ctl._grinding(snap) is True
    assert any(note.startswith("hop_grind_") for note in ctl.notes)


def test_the_budget_is_per_hop_and_per_screen() -> None:
    """Crossing a screen, or advancing the hop, is progress and resets it."""
    from zelda_i.ram import read_snapshot

    ctl = _grind_ctl()
    here = read_snapshot(_ram(screen=0x7C, x=120, y=141, health=0x22))
    for _ in range(ctl.hop_screen_max_frames + 1):
        ctl._grinding(here)
    assert ctl._grinding(here) is True
    next_screen = read_snapshot(_ram(screen=0x7D, x=20, y=141, health=0x22))
    assert ctl._grinding(next_screen) is False
    assert ctl._grinding(here) is False  # a different key, counted afresh


def test_the_lane_tug_of_war_writes_itself_off() -> None:
    """Live 0x7C (``scratch/probe_walk_trace.py`` w1, the committed walk).

    The peel steps LEFT and the plain push steps RIGHT, so x reads 24, 25,
    24, 25 at y=109 for 3593 of the screen's 4301 frames — 1795 ``hop5_lane``
    against 1982 ``hop5``. Neither existing cap can see it: every push frame
    takes an early return that zeroes ``_lane_steer`` and ``_lane_stand``, so
    both counters restart before either reaches its own cap. Travel-axis
    progress is the only thing that separates a peel going around a body from
    a stand-off, and that is what ``_lane_no_gain`` measures.
    """
    from zelda_i.overworld.path import (
        _OCCUPIED_LANE_NO_GAIN_CAP,
        _OCCUPIED_LANE_STEER_CAP,
    )

    ctrl = _lane_ctrl()
    on_lane = (_octorok(x=160, y=141),)
    off_lane = (_octorok(x=160, y=64),)
    x = 24
    lane_frames = 0
    reasons = []
    for i in range(4 * _OCCUPIED_LANE_NO_GAIN_CAP):
        act = ctrl.step(
            _snap(screen=0x49, x=x, y=141, objects=on_lane if i % 2 == 0 else off_lane)
        )
        reasons.append(act.reason)
        if "lane" in act.reason:
            lane_frames += 1
        x = 25 if x == 24 else 24

    assert ctrl._lane_steer <= _OCCUPIED_LANE_STEER_CAP, "steer cap never fires here"
    assert any(n.startswith("lane_nogain_") for n in ctrl.notes), ctrl.notes
    assert "lane" not in reasons[-1], reasons[-4:]
    # The branch pays the cap once and is written off for this (hop, screen);
    # it does not buy itself another cap's worth on the next pixel.
    assert lane_frames <= _OCCUPIED_LANE_NO_GAIN_CAP + 2, lane_frames


def test_the_lane_keeps_the_frame_while_the_peel_is_buying_travel() -> None:
    """A peel that gains ground resets the budget; only a stand-off pays it."""
    from zelda_i.overworld.path import _OCCUPIED_LANE_NO_GAIN_CAP

    ctrl = _lane_ctrl()
    x = 24
    lane_frames = 0
    frames = _OCCUPIED_LANE_NO_GAIN_CAP + 30  # past the cap, inside the screen
    for _ in range(frames):
        act = ctrl.step(
            _snap(screen=0x49, x=x, y=141, objects=(_octorok(x=x + 20, y=141),))
        )
        if "lane" in act.reason:
            lane_frames += 1
        x += 1  # the push wins a pixel every frame
    assert not any(n.startswith("lane_nogain_") for n in ctrl.notes), ctrl.notes
    assert lane_frames == frames, (lane_frames, frames)
