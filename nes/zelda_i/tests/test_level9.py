"""Level 9 public seam: fail-closed hops, hypothesis graph, write-free credits."""

from __future__ import annotations

import inspect

from retro_harness.nes import nes_action
from zelda_i.combat import BOMB_DROP_OBJECT_TYPE, BOMB_DROP_STATE
from zelda_i.door_graph import (
    DoorDir,
    InventoryCaps,
    L9_ENTRY,
    L9_PATRA,
    L9_ROOM_41,
    L9_ROOM_51,
    L9_ROOM_62,
    L9_SILVER_ARROWS,
    LEVEL_9_NATURAL_DOOR_GRAPH,
    natural_route_requires_51_to_41,
)
from zelda_i.level9.dungeon import (
    BOMBS_NOT_NATURAL,
    FULL_TRIFORCE,
    L9_PUBLIC_THROUGH,
    L9_SELECTED_JOIN_ROOMS,
    MISSING_POST_L8_LEFTOVER,
    MISSING_SILVER_ARROW_ROOM,
    ROOM_LEVEL9_ENTRY,
    ROOM_SILVER_ARROWS_HYP,
    TRIFORCE_NOT_FULL,
    UNMEASURED_POST_L8_HANDOFF,
    MEASURED_POST_L8_HANDOFF,
    MISSING_SPECTACLE_BOMB,
    level9_credits_stop,
    level9_entry_stop,
    level9_silver_arrows_stop,
)
from zelda_i.level9.ganon import MODE_ENDING
from zelda_i.level9.hops import (
    LEVEL9_BOMBS_WANTED,
    Level9NaturalRouteSelection,
    SELECTED_NATURAL_ROUTE,
    l9_hops,
    level9_credits_chapter,
    level9_entry_chapter,
    level9_patra_chapter,
    level9_silver_arrows_chapter,
)
from zelda_i.level9.natural_path import (
    NaturalGanonController,
    NaturalSelectSilverArrowsController,
    make_post_l8_overworld_controller,
    make_spectacle_rock_bomb_controller,
)
from zelda_i.level9.overworld import (
    B_ITEM_BOMBS,
    REVERSE_5C_MAZE_WAYPOINTS,
    FixtureEntryPhase,
    Level9FixtureEntryController,
    Level9PostL8OverworldController,
    Level9SpectacleRockBombController,
    SpectacleRockBombPhase,
)
from zelda_i.level9.spine import L9_THROUGH
from zelda_i.ram import (
    ADDR_MAGIC_KEY,
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    ZeldaObject,
    ZeldaSnapshot,
)


def _snap(**kwargs) -> ZeldaSnapshot:
    fields = dict(
        mode=PLAY_MODE,
        level=9,
        screen=ROOM_LEVEL9_ENTRY,
        next_screen=ROOM_LEVEL9_ENTRY,
        link_x=120,
        link_y=189,
        facing=0,
        sword=3,
        bombs=8,
        rupees=0,
        keys=0,
        health=0xFF,
        triforce=FULL_TRIFORCE,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
        bow=1,
        arrows=1,
    )
    fields.update(kwargs)
    return ZeldaSnapshot(**fields)


def test_public_through_names_are_exactly_four_chapters() -> None:
    assert L9_THROUGH == L9_PUBLIC_THROUGH
    hops = l9_hops(None)
    assert tuple(h.through for h in hops) == L9_THROUGH


class _FixtureEnv:
    def __init__(self, selected: int = 2) -> None:
        self.ram = [0] * 0x800
        self.ram[ADDR_SELECTED_ITEM] = selected
        self.ram[ADDR_MAGIC_KEY] = 1

    def get_ram(self):
        return self.ram


def _fixture_controller(
    phase: FixtureEntryPhase, *, selected: int = 2
) -> Level9FixtureEntryController:
    controller = Level9FixtureEntryController(phase=phase)
    controller.bind_env(_FixtureEnv(selected))
    controller._start_checked = True
    controller.bombs_before = 16
    controller.selected_before = selected
    return controller


def test_fixture_route_uses_verified_0x27_and_0x17_waypoints() -> None:
    drop = _fixture_controller(FixtureEntryPhase.DROP_27)
    assert drop.step(_snap(level=0, screen=0x27, link_x=240, link_y=101)).action == nes_action("DOWN")
    align = _fixture_controller(FixtureEntryPhase.DROP_27)
    assert align.step(_snap(level=0, screen=0x27, link_x=240, link_y=133)).action == nes_action("LEFT")
    mouth = _fixture_controller(FixtureEntryPhase.ALIGN_27_X)
    assert mouth.step(_snap(level=0, screen=0x27, link_x=144, link_y=133)).action == nes_action("UP")

    climb = _fixture_controller(FixtureEntryPhase.CLIMB_17)
    assert climb.step(_snap(level=0, screen=0x17, link_x=144, link_y=221)).action == nes_action("UP")
    raft = _fixture_controller(FixtureEntryPhase.ALIGN_17_X)
    assert raft.step(_snap(level=0, screen=0x17, link_x=64, link_y=133)).action == nes_action("UP")


def test_fixture_0x58_rejects_the_blocked_loose_x104_lane() -> None:
    controller = _fixture_controller(FixtureEntryPhase.ALIGN_58_X)
    action = controller.step(_snap(level=0, screen=0x58, link_x=104, link_y=157))
    assert action.action == nes_action("RIGHT")
    assert controller.phase is FixtureEntryPhase.ALIGN_58_X


def test_fixture_0x38_uses_bridge_before_west_alignment() -> None:
    climb = _fixture_controller(FixtureEntryPhase.INLAND_38)
    assert climb.step(_snap(level=0, screen=0x38, link_x=128, link_y=189)).action == nes_action("UP")
    reenter = _fixture_controller(FixtureEntryPhase.ALIGN_38_X)
    assert reenter.step(_snap(level=0, screen=0x38, link_x=112, link_y=205)).action == nes_action("UP")
    bridge = _fixture_controller(FixtureEntryPhase.ALIGN_38_X)
    assert bridge.step(_snap(level=0, screen=0x38, link_x=112, link_y=141)).action == nes_action("RIGHT")
    central = _fixture_controller(FixtureEntryPhase.ALIGN_38_X)
    assert central.step(_snap(level=0, screen=0x38, link_x=120, link_y=141)).reason == "screen38_north_0x28"
    blocked = _fixture_controller(FixtureEntryPhase.NORTH_38)
    action = blocked.step(_snap(level=0, screen=0x38, link_x=48, link_y=133))
    assert action.reason == "known_blocked_0x38_x48_y133_replan"
    assert blocked.failed


def test_fixture_left_rock_prediction_is_top_gap_then_one_bomb() -> None:
    top = _fixture_controller(FixtureEntryPhase.ROCK_TOP_Y, selected=B_ITEM_BOMBS)
    assert top.step(_snap(level=0, screen=0x05, link_x=240, link_y=141)).action == nes_action("UP")
    gap = _fixture_controller(FixtureEntryPhase.ROCK_GAP_X, selected=B_ITEM_BOMBS)
    assert gap.step(_snap(level=0, screen=0x05, link_x=240, link_y=93)).action == nes_action("LEFT")
    south = _fixture_controller(FixtureEntryPhase.ROCK_BOTTOM_Y, selected=B_ITEM_BOMBS)
    assert south.step(_snap(level=0, screen=0x05, link_x=120, link_y=93)).action == nes_action("DOWN")
    left = _fixture_controller(FixtureEntryPhase.ROCK_LEFT_X, selected=B_ITEM_BOMBS)
    assert left.step(_snap(level=0, screen=0x05, link_x=120, link_y=173)).action == nes_action("LEFT")

    fire = _fixture_controller(FixtureEntryPhase.ROCK_FACE_UP, selected=B_ITEM_BOMBS)
    stand = _snap(level=0, screen=0x05, link_x=80, link_y=173, bombs=16)
    assert fire.step(stand).action == nes_action("UP")
    assert fire.step(stand).action == nes_action("B")
    assert fire.b_presses == 1


def test_fixture_bomb_selection_is_pause_input_and_never_a_ram_write() -> None:
    controller = _fixture_controller(FixtureEntryPhase.PAUSE_OPEN, selected=2)
    assert controller.step(_snap(level=0, screen=0x05, bombs=16)).action == nes_action("START")
    source = inspect.getsource(Level9FixtureEntryController)
    assert "set_value" not in source
    assert "ADDR_SELECTED_ITEM] =" not in source
    report = controller.report()
    assert report["fixture_only"] is True
    assert report["natural_entry"] is False
    assert report["route_eligible"] is False
    assert report["position_writes"] == 0
    assert report["selected_item_writes"] == 0
    assert report["progression_writes"] == 0
    assert report["capacity_writes"] == 0


def test_fixture_right_does_not_count_until_slot_changes() -> None:
    from zelda_i.dungeon.pause_select import CURSOR_SETTLE_FRAMES, OPEN_SETTLE_FRAMES

    controller = _fixture_controller(FixtureEntryPhase.PAUSE_OPEN, selected=2)
    snap = _snap(level=0, screen=0x05, bombs=16)
    assert controller.step(snap).reason == "pause_open"
    for _ in range(OPEN_SETTLE_FRAMES):
        controller.step(snap)
    controller.step(snap)  # RIGHT, $0656 still arrows
    assert controller.cursor_moves == 0
    for _ in range(CURSOR_SETTLE_FRAMES):
        controller.step(snap)
    assert controller.cursor_moves == 0


def test_entry_stop_requires_tf_magic_key_and_bombs() -> None:
    ok = _snap()
    assert level9_entry_stop(ok, magic_key=True)
    assert not level9_entry_stop(ok, magic_key=False)
    assert not level9_entry_stop(_snap(triforce=0x7F), magic_key=True)
    assert not level9_entry_stop(_snap(bombs=0), magic_key=True)
    assert not level9_entry_stop(_snap(screen=0x66), magic_key=True)


def test_silver_arrows_stop_uses_selected_room_0x10() -> None:
    assert SELECTED_NATURAL_ROUTE.silver_arrow_room == ROOM_SILVER_ARROWS_HYP == 0x10
    got = _snap(screen=0x10, arrows=2, bow=1)
    assert level9_silver_arrows_stop(got, room=0x10)
    assert not level9_silver_arrows_stop(got, room=None)
    assert not level9_silver_arrows_stop(_snap(screen=0x10, arrows=1), room=0x10)
    assert not level9_silver_arrows_stop(_snap(screen=0x10, arrows=2, triforce=0x7F), room=0x10)


def test_prefix_hops_fail_closed_when_triforce_not_full() -> None:
    snap = _snap(triforce=0x7F, level=0, screen=0x6D, bombs=8)
    for chapter in (
        level9_entry_chapter(),
        level9_silver_arrows_chapter(),
    ):
        _name, controller, max_frames = chapter[0]
        assert max_frames == 1
        act = controller.step(snap)
        assert controller.failed
        assert act.reason == TRIFORCE_NOT_FULL
        report = controller.report()
        assert report["inventory_writes"] == 0
        assert report["triforce_writes"] == 0
        assert report["route_eligible"] is False

    # Live join controller also fails closed without full triforce
    _name, join_ctl, max_frames = level9_patra_chapter()[0]
    assert max_frames == 24000
    join_ctl.step(snap)
    assert join_ctl.failed
    report = join_ctl.report()
    assert report["inventory_writes"] == 0
    assert report["triforce_writes"] == 0
    assert report["route_eligible"] is False


def test_entry_refuses_without_natural_bombs_and_does_not_write_capacity() -> None:
    ctl = make_post_l8_overworld_controller()
    act = ctl.step(_snap(triforce=FULL_TRIFORCE, bombs=0, level=0, screen=0x6D))
    assert act.reason == BOMBS_NOT_NATURAL
    report = ctl.report()
    assert report["bomb_capacity_writes"] == 0
    assert report["capacity_writes"] == 0


def test_entry_missing_evidence_is_unmeasured_post_l8_leftover() -> None:
    assert not UNMEASURED_POST_L8_HANDOFF.complete()
    ctl = make_post_l8_overworld_controller()
    act = ctl.step(_snap(level=0, screen=0x6D, triforce=FULL_TRIFORCE, bombs=8))
    assert act.reason == MISSING_POST_L8_LEFTOVER


def test_silver_and_patra_missing_evidence_are_exact() -> None:
    _n, silver, _ = level9_silver_arrows_chapter()[0]
    silver.step(_snap())
    assert silver.blocked_reason == MISSING_SILVER_ARROW_ROOM
    empty_route = Level9NaturalRouteSelection(suffix_join_room=None)
    _n, join, _ = level9_patra_chapter(empty_route)[0]
    join.step(_snap())
    assert join.blocked_reason == "natural_suffix_join_not_selected"


def test_credits_hop_does_not_write_inventory_or_load_fixture() -> None:
    src = inspect.getsource(level9_credits_chapter)
    assert "ReconFixture" not in src
    assert "FULL_LOADOUT" not in src
    assert "set_value" not in src
    stages = level9_credits_chapter()
    names = [name for name, _, _ in stages]
    assert names[0] == "level9_select_silver_arrows"
    assert names[-1] == "level9_wait_credits"
    select = stages[0][1]
    assert isinstance(select, NaturalSelectSilverArrowsController)
    ganon = next(c for n, c, _ in stages if n == "level9_ganon")
    assert isinstance(ganon.inner, NaturalGanonController)
    for _name, controller, _max in stages:
        report = controller.report()
        assert report["fixture_loaded"] is False
        assert report["controller_memory_writes"] == 0
        assert report["inventory_writes"] == 0
        assert report["progression_writes"] == 0
        assert report["capacity_writes"] == 0
        assert report.get("selected_item_writes", 0) == 0


def test_credits_stop_requires_deaths_zero() -> None:
    rolling = _snap(mode=MODE_ENDING, is_updating_mode=1, submode=3)
    assert level9_credits_stop(rolling, deaths=0)
    assert not level9_credits_stop(rolling, deaths=1)
    assert not level9_credits_stop(_snap(), deaths=0)


def test_0x62_is_not_patra_south_on_natural_graph() -> None:
    g = LEVEL_9_NATURAL_DOOR_GRAPH
    assert g.exit_between(L9_ROOM_62, L9_PATRA) is None
    assert g.exit_between(L9_PATRA, L9_ROOM_62) is None
    assert L9_ROOM_62 not in L9_SELECTED_JOIN_ROOMS
    north = [e for e in g.edges_from(L9_ROOM_62) if e.direction is DoorDir.UP]
    assert north and all(not e.is_pathfinding for e in north)
    west = g.exit_between(L9_ROOM_62, 0x61, direction=DoorDir.LEFT)
    assert west is not None and west.verification == "observed"


def test_0x51_to_0x41_required_because_selected_route_says_so() -> None:
    assert natural_route_requires_51_to_41() is True
    assert SELECTED_NATURAL_ROUTE.requires_51_to_41 is True
    assert SELECTED_NATURAL_ROUTE.suffix_join_room == L9_ROOM_41
    assert SELECTED_NATURAL_ROUTE.route_eligible is False
    edge = LEVEL_9_NATURAL_DOOR_GRAPH.exit_between(
        L9_ROOM_51, L9_ROOM_41, direction=DoorDir.UP
    )
    assert edge is not None
    assert edge.verification == "observed"
    assert "statue diamond" in edge.notes


def test_magic_key_hypothesis_reaches_silver_arrows_and_patra() -> None:
    caps = InventoryCaps(keys=99, bombs=16, can_clear=True)
    g = LEVEL_9_NATURAL_DOOR_GRAPH
    to_silver = g.bfs_path(L9_ENTRY, L9_SILVER_ARROWS, caps)
    assert to_silver is not None
    silver_rooms = [L9_ENTRY, *[e.target_room for e in to_silver]]
    assert silver_rooms[-1] == L9_SILVER_ARROWS == 0x10
    assert 0x07 not in silver_rooms
    assert L9_PATRA not in silver_rooms
    to_patra = g.bfs_path(L9_SILVER_ARROWS, L9_PATRA, caps)
    assert to_patra is not None
    join_rooms = [L9_SILVER_ARROWS, *[e.target_room for e in to_patra]]
    assert join_rooms[-1] == L9_PATRA
    assert L9_ROOM_51 in join_rooms
    assert L9_ROOM_41 in join_rooms
    assert L9_ROOM_62 not in join_rooms


def test_patra_census_objects_are_body_and_eight_eyes() -> None:
    body = ZeldaObject(slot=1, type_id=0x47, x=120, y=120, facing=0, hp=0xB0, state=0)
    eyes = tuple(
        ZeldaObject(slot=i, type_id=0x25, x=100, y=100, facing=0, hp=0x60, state=0)
        for i in range(2, 10)
    )
    snap = _snap(screen=0x52, arrows=2, bow=1, objects=(body, *eyes))
    from zelda_i.level9.dungeon import level9_live_patra_stop

    assert level9_live_patra_stop(snap)
    assert not level9_live_patra_stop(_snap(screen=0x52, arrows=2, objects=(body,)))


def test_post_l8_overworld_reverse_maze_navigation() -> None:
    assert REVERSE_5C_MAZE_WAYPOINTS == ((192, 132), (192, 92), (16, 92))
    ctl = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ctl._handoff_checked = True
    ctl.hop_index = 2  # Hop 2 target is 0x5B, Link is on 0x5C
    assert ctl.hops[ctl.hop_index].target == 0x5B

    # Entering from 0x5D at (240, 133): moves LEFT
    snap0 = _snap(level=0, screen=0x5C, link_x=240, link_y=133)
    act0 = ctl.step(snap0)
    assert act0.action == nes_action("LEFT")
    assert act0.reason == "5c_reverse_maze_wp0"
    assert ctl.reverse_maze_wp_index == 0

    # Reached x <= 192 at y=132: advances to wp 1, moves UP
    snap1 = _snap(level=0, screen=0x5C, link_x=192, link_y=132)
    act1 = ctl.step(snap1)
    assert act1.action == nes_action("UP")
    assert act1.reason == "5c_reverse_maze_wp1"
    assert ctl.reverse_maze_wp_index == 1

    # Reached y <= 92 at x=192: advances to wp 2, moves LEFT
    snap2 = _snap(level=0, screen=0x5C, link_x=192, link_y=92)
    act2 = ctl.step(snap2)
    assert act2.action == nes_action("LEFT")
    assert act2.reason == "5c_reverse_maze_wp2"
    assert ctl.reverse_maze_wp_index == 2

    # Reached x <= 16: exits into 0x5B
    snap3 = _snap(level=0, screen=0x5C, link_x=16, link_y=92)
    act3 = ctl.step(snap3)
    assert act3.action == nes_action("LEFT")
    assert act3.reason == "5c_reverse_maze_exit"
    assert ctl.reverse_maze_wp_index == 3


def test_post_l8_overworld_screens_5a_59_58_navigation() -> None:
    ctl = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ctl._handoff_checked = True

    # Screen 0x5A -> 0x59
    ctl.hop_index = 4
    assert ctl.hops[ctl.hop_index].target == 0x59
    assert ctl.step(_snap(level=0, screen=0x5A, link_x=240, link_y=93)).action == nes_action("DOWN")
    assert ctl.step(_snap(level=0, screen=0x5A, link_x=240, link_y=140)).action == nes_action("LEFT")

    # Screen 0x59 -> 0x58: direct east arrival crosses at y=141. A
    # bomb-shop return enters from the north at (112,61) and must descend
    # through the central passage before moving west.
    ctl.hop_index = 5
    assert ctl.hops[ctl.hop_index].target == 0x58
    assert ctl.step(_snap(level=0, screen=0x59, link_x=112, link_y=61)).action == nes_action("DOWN")
    assert ctl.step(_snap(level=0, screen=0x59, link_x=240, link_y=141)).action == nes_action("LEFT")
    assert ctl.step(_snap(level=0, screen=0x59, link_x=155, link_y=141)).action == nes_action("LEFT")

    # Screen 0x58 -> 0x48: the arrival band (y~141) has an obstacle
    # blocking LEFT around x=155-224 (confirmed empirically) that isn't
    # present a few px south (y>=149); descend once before the x-align.
    ctl.hop_index = 6
    assert ctl.hops[ctl.hop_index].target == 0x48
    assert not ctl._cleared_58_south_wall
    assert ctl.step(_snap(level=0, screen=0x58, link_x=240, link_y=141)).action == nes_action("DOWN")
    assert not ctl._cleared_58_south_wall
    assert ctl.step(_snap(level=0, screen=0x58, link_x=240, link_y=150)).action == nes_action("LEFT")
    assert ctl._cleared_58_south_wall
    assert ctl.step(_snap(level=0, screen=0x58, link_x=112, link_y=150)).action == nes_action("UP")
    # Once cleared, the wall check never re-arms even if y dips back under
    # 149 from a single UP tap (the ping-pong this latch exists to avoid).
    assert ctl.step(_snap(level=0, screen=0x58, link_x=112, link_y=145)).action == nes_action("UP")


def test_post_l8_5a_recovers_from_south_wall_with_lattice(monkeypatch) -> None:
    ctl = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ctl._handoff_checked = True
    ctl.hop_index = 4
    calls = []

    def route(_env, _snap, direction, lo, hi):
        calls.append((direction, lo, hi))
        return "UP"

    monkeypatch.setattr("zelda_i.level9.overworld.ow_edge_band_step", route)
    action = ctl.step(_snap(level=0, screen=0x5A, link_x=112, link_y=205))
    assert action.action == nes_action("UP")
    assert action.reason == "5a_recover_west_lattice"
    assert calls == [("LEFT", 137, 145)]


def test_post_l8_banks_natural_5d_bombs_before_shop() -> None:
    stages = level9_entry_chapter(handoff=MEASURED_POST_L8_HANDOFF)
    ctl = stages[0][1]
    assert ctl.bomb_goal == LEVEL9_BOMBS_WANTED == 8
    assert stages[5][1].bomb_goal == 0  # the return walk has no shop target
    ctl._handoff_checked = True
    ctl.hop_index = 1
    drop = ZeldaObject(
        slot=1, type_id=BOMB_DROP_OBJECT_TYPE, x=32, y=141,
        facing=0, hp=0, state=BOMB_DROP_STATE,
    )
    snap = _snap(level=0, screen=0x5D, link_x=48, link_y=141, bombs=4, objects=(drop,))
    action = ctl.step(snap)
    assert action.action == nes_action("LEFT")
    assert action.reason == "5d_scoop_bomb"
    assert ctl.step(_snap(level=0, screen=0x5D, link_x=48, link_y=141, bombs=8, objects=(drop,))).reason != "5d_scoop_bomb"


def test_post_l8_overworld_screen_38_bridge_latch() -> None:
    # rr-sz8.5 follow-on: the old "realign to exactly y=141" branch pressed
    # DOWN whenever y<137, which re-fired on every later frame once the
    # final UP commit had already overshot north past the hazard band
    # (observed reaching y~105 before being walked back to ~134) -- undoing
    # real progress instead of just a re-checked-threshold ping-pong.
    ctl = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ctl._handoff_checked = True
    ctl.hop_index = 8
    assert ctl.hops[ctl.hop_index].target == 0x28

    assert not ctl._cleared_38_bridge
    assert ctl.step(_snap(level=0, screen=0x38, link_x=120, link_y=189)).action == nes_action("UP")
    assert not ctl._cleared_38_bridge
    assert ctl.step(_snap(level=0, screen=0x38, link_x=120, link_y=137)).action == nes_action("UP")
    assert ctl._cleared_38_bridge
    # Once cleared, a real UP overshoot to well below 141 must never see the
    # old branch re-arm and press DOWN -- always UP from here.
    assert ctl.step(_snap(level=0, screen=0x38, link_x=120, link_y=105)).action == nes_action("UP")
    assert ctl.step(_snap(level=0, screen=0x38, link_x=120, link_y=133)).action == nes_action("UP")


def test_post_l8_overworld_screen_27_gap_latch() -> None:
    # Same latch bug as 0x38: the final "UP" commit routinely overshoots
    # y<133, and the old "drop below mountain" branch re-fired on every
    # later frame with y<133, walking Link back south and undoing the
    # northward progress it just made.
    ctl = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ctl._handoff_checked = True
    ctl.hop_index = 10
    assert ctl.hops[ctl.hop_index].target == 0x17

    assert not ctl._cleared_27_gap
    assert ctl.step(_snap(level=0, screen=0x27, link_x=144, link_y=101)).action == nes_action("DOWN")
    assert not ctl._cleared_27_gap
    assert ctl.step(_snap(level=0, screen=0x27, link_x=144, link_y=133)).action == nes_action("UP")
    assert ctl._cleared_27_gap
    # Once cleared, an overshoot to well below 133 must never re-arm the
    # old DOWN-pressing branch -- always UP from here.
    assert ctl.step(_snap(level=0, screen=0x27, link_x=144, link_y=101)).action == nes_action("UP")
    assert ctl.step(_snap(level=0, screen=0x27, link_x=144, link_y=131)).action == nes_action("UP")


def test_post_l8_overworld_controller_accepts_measured_handoff() -> None:
    ctl = make_post_l8_overworld_controller(MEASURED_POST_L8_HANDOFF)
    assert isinstance(ctl, Level9PostL8OverworldController)
    assert ctl.max_frames == 12_000

    snap = _snap(level=0, screen=0x6D, link_x=96, link_y=93, triforce=FULL_TRIFORCE, bombs=14)
    act = ctl.step(snap)
    assert not ctl.failed
    assert act.action == nes_action("LEFT")
    assert act.reason == "6d_walk_left_x48"

    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["evidence"] == "spine-green"
    assert report["writes"] == 0
    assert report["controller_memory_writes"] == 0


def test_post_l8_overworld_controller_arrival_at_0x05() -> None:
    ctl = make_post_l8_overworld_controller(MEASURED_POST_L8_HANDOFF)
    ctl._handoff_checked = True
    ctl.hop_index = len(ctl.hops)

    snap = _snap(level=0, screen=0x05, link_x=240, link_y=141, triforce=FULL_TRIFORCE, bombs=14)
    act = ctl.step(snap)
    assert ctl.success
    assert act.reason == "done"
    report = ctl.report()
    assert report["success"] is True
    assert report["route_eligible"] is True
    assert report["writes"] == 0


def test_spectacle_rock_bomb_controller_unmeasured_fails_closed() -> None:
    ctl = make_spectacle_rock_bomb_controller()
    assert ctl.max_frames == 1
    snap = _snap(level=0, screen=0x05, link_x=240, link_y=141, triforce=FULL_TRIFORCE, bombs=14)
    act = ctl.step(snap)
    assert ctl.failed
    assert act.reason == MISSING_SPECTACLE_BOMB
    report = ctl.report()
    assert report["failed"] is True
    assert report["route_eligible"] is False
    assert report["writes"] == 0

    # TF mismatch fails on frame 1
    ctl_tf = make_spectacle_rock_bomb_controller(MEASURED_POST_L8_HANDOFF)
    act_tf = ctl_tf.step(_snap(level=0, screen=0x05, link_x=240, link_y=141, triforce=0x7F, bombs=14))
    assert ctl_tf.failed
    assert act_tf.reason == TRIFORCE_NOT_FULL

    # Bombs 0 fails on frame 1
    ctl_bombs = make_spectacle_rock_bomb_controller(MEASURED_POST_L8_HANDOFF)
    act_bombs = ctl_bombs.step(_snap(level=0, screen=0x05, link_x=240, link_y=141, triforce=FULL_TRIFORCE, bombs=0))
    assert ctl_bombs.failed
    assert act_bombs.reason == BOMBS_NOT_NATURAL

    # Screen mismatch fails on frame 1
    ctl_screen = make_spectacle_rock_bomb_controller(MEASURED_POST_L8_HANDOFF)
    act_screen = ctl_screen.step(_snap(level=0, screen=0x06, link_x=240, link_y=141, triforce=FULL_TRIFORCE, bombs=14))
    assert ctl_screen.failed
    assert act_screen.reason == "not_on_spectacle_rock_0x05"


def test_spectacle_rock_bomb_controller_phases_navigation() -> None:
    ctl = make_spectacle_rock_bomb_controller(MEASURED_POST_L8_HANDOFF)
    assert ctl.max_frames == 4000

    # Step 1 at (240, 141): the first lattice leg heads for (216, 93)
    snap0 = _snap(level=0, screen=0x05, link_x=240, link_y=141, triforce=FULL_TRIFORCE, bombs=14)
    act0 = ctl.step(snap0)
    assert act0.action in (nes_action("LEFT"), nes_action("UP"))
    assert ctl.phase is SpectacleRockBombPhase.ROCK_TOP_Y

    # Reached x=216: moves UP to y=93
    snap1 = _snap(level=0, screen=0x05, link_x=216, link_y=141, triforce=FULL_TRIFORCE, bombs=14)
    act1 = ctl.step(snap1)
    assert act1.action == nes_action("UP")
    assert act1.reason == "rock_climb_top_y93"
    assert ctl.phase is SpectacleRockBombPhase.ROCK_TOP_Y

    # Reached y=93 at x=216: moves LEFT to x=120
    snap2 = _snap(level=0, screen=0x05, link_x=216, link_y=93, triforce=FULL_TRIFORCE, bombs=14)
    act2 = ctl.step(snap2)
    assert act2.action == nes_action("LEFT")
    assert act2.reason == "rock_top_to_center_gap_x120"
    assert ctl.phase is SpectacleRockBombPhase.ROCK_GAP_X

    # Reached x=120 at y=93: moves DOWN to y=173
    snap3 = _snap(level=0, screen=0x05, link_x=120, link_y=93, triforce=FULL_TRIFORCE, bombs=14)
    act3 = ctl.step(snap3)
    assert act3.action == nes_action("DOWN")
    assert act3.reason == "rock_center_gap_to_south_y173"
    assert ctl.phase is SpectacleRockBombPhase.ROCK_BOTTOM_Y

    # Reached y=173 at x=120: moves LEFT to x=80
    snap4 = _snap(level=0, screen=0x05, link_x=120, link_y=173, triforce=FULL_TRIFORCE, bombs=14)
    act4 = ctl.step(snap4)
    assert act4.action == nes_action("LEFT")
    assert act4.reason == "rock_south_to_left_stand_x80"
    assert ctl.phase is SpectacleRockBombPhase.ROCK_LEFT_X

    # Reached x=80 at y=173 facing west: turn UP first, no bomb yet (B drops
    # it the way Link faces).
    snap_turn = _snap(level=0, screen=0x05, link_x=80, link_y=173, triforce=FULL_TRIFORCE, bombs=14, facing=0x02)
    act_turn = ctl.step(snap_turn)
    assert act_turn.action == nes_action("UP")
    assert act_turn.reason == "left_rock_face_up"
    assert ctl.phase is SpectacleRockBombPhase.ROCK_FACE_UP

    # Facing UP on the stand: arm the bomb
    snap5 = _snap(level=0, screen=0x05, link_x=80, link_y=173, triforce=FULL_TRIFORCE, bombs=14, facing=0x08)
    act5 = ctl.step(snap5)
    assert act5.action == nes_action("UP")
    assert act5.reason == "left_rock_face_up"
    assert ctl.phase is SpectacleRockBombPhase.ROCK_FIRE

    # Presses B to place bomb
    act6 = ctl.step(snap5)
    assert act6.action == nes_action("B")
    assert act6.reason == "left_spectacle_rock_bomb"
    assert ctl.b_presses == 1
    assert ctl.phase is SpectacleRockBombPhase.ROCK_BLAST_WAIT

    # Blast wait: 180 frames idle while bomb explodes (bombs decrease to 13)
    snap_exploded = _snap(level=0, screen=0x05, link_x=80, link_y=173, triforce=FULL_TRIFORCE, bombs=13)
    for _ in range(179):
        act_wait = ctl.step(snap_exploded)
        assert act_wait.reason == "left_rock_blast_wait"
    act_blast_end = ctl.step(snap_exploded)
    assert ctl.phase is SpectacleRockBombPhase.ROCK_ENTER
    assert act_blast_end.action == nes_action("UP")
    assert act_blast_end.reason == "enter_left_spectacle_rock"

    # Knockback during blast wait can place Link north of the entrance (e.g. y=149)
    snap_knocked = _snap(level=0, screen=0x05, link_x=80, link_y=149, triforce=FULL_TRIFORCE, bombs=13)
    act_knocked = ctl.step(snap_knocked)
    assert act_knocked.action == nes_action("DOWN")
    assert act_knocked.reason == "enter_left_spectacle_rock_from_north"

    # Transition into Level 9
    snap_trans = _snap(level=9, screen=0x76, link_x=120, link_y=205, mode=6, bombs=13)
    act_trans = ctl.step(snap_trans)
    assert act_trans.reason in ("transition_wait", "dungeon_loader_wait")

    # Dungeon settle in room 0x76: 24 frames
    snap_l9 = _snap(level=9, screen=0x76, link_x=120, link_y=205, mode=PLAY_MODE, bombs=13)
    for _ in range(23):
        act_settle = ctl.step(snap_l9)
        assert act_settle.reason == "dungeon_0x76_settle"
    act_done = ctl.step(snap_l9)
    assert act_done.reason == "done"
    assert ctl.success is True

    report = ctl.report()
    assert report["success"] is True
    assert report["route_eligible"] is True
    assert report["writes"] == 0
    assert report["bombs_before"] == 14
    assert report["bombs_after"] == 13
    assert report["b_presses"] == 1
    assert report["controller_memory_writes"] == 0
    assert report["progression_writes"] == 0
    assert report["capacity_writes"] == 0


def test_level9_entry_chapter_chaining() -> None:
    stages = level9_entry_chapter(handoff=MEASURED_POST_L8_HANDOFF)
    assert [stage[0] for stage in stages] == [
        "level9_post_l8_overworld",
        "bomb_restock_l8",
        "exit_bomb_restock_l8",
        "bomb_restock_l8_second",
        "exit_bomb_restock_l8_second",
        "level9_post_l8_to_rock",
        "level9_white_sword",
        "level9_spectacle_rock_bomb",
    ]
    assert stages[0][2] == 12_000
    assert stages[2][2] == 600
    assert stages[4][2] == 600
    assert stages[5][2] == 12_000
    assert stages[6][2] == 20_000
    assert stages[7][2] == 4000

    # Both controllers accept measured handoff and have route_eligible when complete
    ow_ctl = stages[0][1]
    bomb_ctl = stages[7][1].inner
    assert isinstance(ow_ctl, Level9PostL8OverworldController)
    assert isinstance(bomb_ctl, Level9SpectacleRockBombController)

    from zelda_i.level9.overworld import POST_L8_TO_LEVEL9_HOPS

    direct_stages = level9_entry_chapter(
        handoff=MEASURED_POST_L8_HANDOFF, post_l8_hops=POST_L8_TO_LEVEL9_HOPS
    )
    assert len(direct_stages) == 3
    assert direct_stages[0][0] == "level9_post_l8_overworld"
    assert direct_stages[1][0] == "level9_white_sword"
    assert direct_stages[2][0] == "level9_spectacle_rock_bomb"
