"""Hop tables and stop predicates for gathering segments. No emulator."""

from __future__ import annotations

from retro_harness.nes import nes_action
from zelda_i.overworld.gather_segments import (
    CANDLE_HOPS,
    CANDLE_PRICE,
    HEART_L8_HOPS,
    HEART_L8_SCREEN,
    HEART_WALK_HOPS,
    LETTER_FROM_0F_HOPS,
    RETURN_7C_HOPS,
    LETTER_HOPS,
    NE_HOPS,
    POTION_HOPS,
    RING_HOPS,
    RING_RETURN_HOPS,
    RING_PRICE,
    WHITE_HOPS,
    ArrivalController,
    BombWallController,
    CaveMouthController,
    NortheastController,
    SEGMENTS,
    main,
    make_candle_controller,
    make_heart_l8_controller,
    make_heart_m3_controller,
    make_letter_controller,
    make_potion_controller,
    make_ring_controller,
    make_white_controller,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import CAVE_MODE, PLAY_MODE, ZeldaObject, ZeldaSnapshot


def _snap(**kwargs) -> ZeldaSnapshot:
    base = dict(
        mode=PLAY_MODE,
        level=0,
        screen=0x7B,
        next_screen=0x7B,
        link_x=120,
        link_y=80,
        facing=8,
        sword=1,
        bombs=4,
        rupees=20,
        keys=0,
        health=0x22,
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
        candle=1,
    )
    base.update(kwargs)
    return ZeldaSnapshot(**base)


def _targets(hops) -> tuple[int, ...]:
    return tuple(hop.target for hop in hops)


def test_heart_l8_comes_from_the_east() -> None:
    assert _targets(HEART_L8_HOPS) == (HEART_L8_SCREEN,)
    assert HEART_L8_HOPS[0].direction == "LEFT"
    assert 0x79 not in _targets(HEART_L8_HOPS)


def test_northeast_walk_avoids_lost_hills() -> None:
    assert _targets(NE_HOPS) == (0x2D, 0x1D, 0x1E, 0x1F, 0x0F)
    assert 0x1B not in _targets(NE_HOPS)
    assert _targets(LETTER_HOPS) == (0x0E,)


_STEP = {"UP": -0x10, "DOWN": 0x10, "LEFT": -1, "RIGHT": 1}


def test_every_hop_direction_matches_the_screen_grid() -> None:
    tables = {
        0x2C: NE_HOPS,
        0x1E: LETTER_HOPS,
        0x0E: CANDLE_HOPS,
        0x0C: WHITE_HOPS,
        0x65: POTION_HOPS,
        0x47: RING_HOPS,
        0x34: RING_RETURN_HOPS,
        0x7C: HEART_L8_HOPS,
        0x7B: HEART_WALK_HOPS,
        0x0F: LETTER_FROM_0F_HOPS,
        0x6F: RETURN_7C_HOPS,
    }
    for start, hops in tables.items():
        here = start
        for hop in hops:
            assert hop.target - here == _STEP[hop.direction], (hex(here), hop)
            here = hop.target


def test_candle_shop_is_0x0c_at_60() -> None:
    assert _targets(CANDLE_HOPS) == (0x1E, 0x1D, 0x0D, 0x0C)
    assert CANDLE_PRICE == 60
    assert 0x5E not in _targets(CANDLE_HOPS)
    assert 0x1B not in _targets(CANDLE_HOPS)


def test_white_approach_crosses_at_0x27_not_up_from_0x2a() -> None:
    """0x29 and 0x2A are walled on top. Row 2 west to 0x27, row 1 east to 0x1A."""
    assert _targets(WHITE_HOPS) == (
        0x1C, 0x2C, 0x2B, 0x2A, 0x29, 0x28, 0x27, 0x17, 0x18, 0x19, 0x1A,
    )
    assert 0x1B not in _targets(WHITE_HOPS)
    up = WHITE_HOPS[_targets(WHITE_HOPS).index(0x17)]
    assert up.direction == "UP" and 112 <= up.align_x <= 160


def test_white_0x28_waypoints_are_single_axis_moves() -> None:
    from zelda_i.overworld.gather_segments import WAYPOINTS, WHITE_28_ENTRY_Y

    corners = WAYPOINTS[0x28]
    assert corners[0][1] == WHITE_28_ENTRY_Y
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        assert (ax == bx) != (ay == by)


def test_white_waypoints_walk_then_decline() -> None:
    from zelda_i.overworld.gather_segments import waypoint_action

    ctrl = make_white_controller()
    act = waypoint_action(ctrl, _snap(screen=0x28, link_x=240, link_y=117))
    assert act is not None and act.reason.startswith("waypoint")
    assert ctrl._way_leg == 0
    act = waypoint_action(ctrl, _snap(screen=0x28, link_x=224, link_y=117))
    assert ctrl._way_leg == 1
    ctrl._way_leg = 6
    assert waypoint_action(ctrl, _snap(screen=0x28, link_x=104, link_y=133)) is None
    assert waypoint_action(ctrl, _snap(screen=0x27, link_x=104, link_y=133)) is None


def test_white_cave_under_five_containers_fails_fast() -> None:
    ctrl = make_white_controller()
    ctrl.hop_index = len(ctrl.hops)
    ctrl._after_hops(_snap(screen=0x0A, mode=CAVE_MODE, health=0x22))
    assert "white_cave_reached_containers_3_of_5" in ctrl.notes
    assert ctrl.success is False
    assert SEGMENTS["gather_white"].save_red is False
    assert ctrl._at_stop(_snap(screen=0x0A, mode=CAVE_MODE, sword=2)) is True


def test_late_arrivals_do_not_enter_level_1() -> None:
    assert _targets(POTION_HOPS) == (0x64,)
    assert POTION_HOPS[0].direction == "LEFT"
    assert _targets(RING_HOPS) == (
        0x48, 0x58, 0x57, 0x56, 0x55, 0x65, 0x64, 0x54, 0x44, 0x34
    )
    assert _targets(RING_RETURN_HOPS)[-1] == 0x58


def test_ring_stage_buys_the_middle_shop_item() -> None:
    from zelda_i.ram import ADDR_RING

    ctrl = make_ring_controller()
    assert ctrl.price == RING_PRICE == 250
    assert ctrl.success_addr == ADDR_RING
    assert ctrl.shop_screen == 0x34
    assert ctrl.buy_x == 120
    assert ctrl.success_getter(_snap(ring=0)) == 0
    assert ctrl.success_getter(_snap(ring=1)) == 1


def _moblin() -> ZeldaObject:
    return ZeldaObject(slot=1, type_id=0x7C, x=120, y=128, facing=0, hp=0, state=0)


def test_northeast_stop_is_the_whole_100_from_cave_entry() -> None:
    """2026-09-22: a floor rupee on the walk (6 -> 7) graded the old stop green."""
    ctrl = NortheastController()
    ctrl.hop_index = len(ctrl.hops)
    assert ctrl._at_stop(_snap(screen=0x0F, mode=CAVE_MODE, rupees=7)) is False
    entry = _snap(screen=0x0F, mode=CAVE_MODE, rupees=7, link_x=112, link_y=213)
    ctrl._after_hops(entry)
    assert ctrl._cave_rupees == 7
    counting = _snap(screen=0x0F, mode=CAVE_MODE, rupees=64)
    assert ctrl._at_stop(counting) is False
    assert ctrl._at_stop(_snap(screen=0x0F, mode=PLAY_MODE, rupees=107)) is False
    assert ctrl._at_stop(_snap(screen=0x0F, mode=CAVE_MODE, rupees=107)) is True


def test_northeast_secret_walk_lines_up_on_x_120_first() -> None:
    """UP at x=112 stops at (112, 141) beside the rupee and never takes it."""
    ctrl = NortheastController()
    frozen = ctrl._after_hops(_snap(screen=0x0F, mode=CAVE_MODE, link_x=112, link_y=213))
    assert frozen.reason == "cave_settle"
    up = (_moblin(),)
    act = ctrl._after_hops(
        _snap(screen=0x0F, mode=CAVE_MODE, link_x=112, link_y=213, objects=up)
    )
    assert act.reason == "secret_align"
    assert act.action == nes_action("RIGHT")
    act = ctrl._after_hops(
        _snap(screen=0x0F, mode=CAVE_MODE, link_x=120, link_y=213, objects=up)
    )
    assert act.action == nes_action("UP")


def test_candle_byte_without_the_debit_is_not_a_stop() -> None:
    """The 0x0C trial flipped the candle at 80 rupees. That is not a buy."""
    ctrl = make_candle_controller()
    assert ctrl.price == CANDLE_PRICE
    start = _snap(screen=0x0C, mode=CAVE_MODE, candle=0, rupees=80)
    assert ctrl._at_stop(start) is False
    ctrl._rupees_at_buy = 80
    unpaid = _snap(screen=0x0C, mode=CAVE_MODE, candle=1, rupees=80)
    assert ctrl._at_stop(unpaid) is False
    paid = _snap(screen=0x0C, mode=CAVE_MODE, candle=1, rupees=20)
    assert ctrl._at_stop(paid) is True


def test_dispatcher_lists_names_and_does_not_walk() -> None:
    assert main([]) == 2
    assert main(["nope"]) == 2
    assert main(["gather_letter", "gather_ne"]) == 2


def test_letter_stop_needs_the_letter_byte_not_the_cave() -> None:
    """Cave mode on 0x0E read $0666 = 0 on 2026-09-22. The item is the stop."""
    ctrl = make_letter_controller()
    assert isinstance(ctrl, CaveMouthController)
    assert ctrl._at_stop(_snap(screen=0x0E, mode=PLAY_MODE)) is False
    assert ctrl._at_stop(_snap(screen=0x0E, mode=CAVE_MODE)) is False
    assert ctrl._at_stop(_snap(screen=0x0E, mode=CAVE_MODE, letter=1)) is True
    keeper = ZeldaObject(slot=1, type_id=0x72, x=120, y=128, facing=0, hp=0, state=0)
    act = ctrl._after_hops(
        _snap(screen=0x0E, mode=CAVE_MODE, link_x=112, link_y=213, objects=(keeper,))
    )
    assert act.action == nes_action("RIGHT")


def test_heart_l8_stop_needs_container_gain() -> None:
    ctrl = make_heart_l8_controller()
    assert isinstance(ctrl, BombWallController)
    play = _snap(screen=0x7B, mode=PLAY_MODE, health=0x22)
    assert ctrl._at_stop(play) is False
    same = _snap(screen=0x7B, mode=CAVE_MODE, health=0x22)
    assert ctrl._at_stop(same) is False
    gained = _snap(screen=0x7B, mode=CAVE_MODE, health=0x33)
    assert ctrl._at_stop(gained) is True


def test_heart_l8_interior_budget_fails_before_timeout() -> None:
    ctrl = make_heart_l8_controller()
    ctrl.interior_budget = 3
    ctrl._entry_containers = 3
    far = _snap(screen=0x7B, mode=CAVE_MODE, link_x=112, link_y=141, health=0x22)
    for _ in range(100):
        ctrl._after_hops(far)
        if not ctrl.success and "heart_not_taken" in ctrl.notes:
            break
    assert ctrl.success is False
    assert "heart_not_taken" in ctrl.notes


def _heart_sprite() -> ZeldaObject:
    return ZeldaObject(slot=1, type_id=0x6B, x=120, y=128, facing=0, hp=0, state=2)


def test_heart_l8_mouth_freeze_is_not_a_wall() -> None:
    """Mode 11 at the bomb mouth still has leevers. Do not grade that lock."""
    ctrl = make_heart_l8_controller()
    ctrl._entry_containers = 3
    leever = ZeldaObject(slot=1, type_id=16, x=144, y=61, facing=0, hp=32, state=0)
    frozen = _snap(
        screen=0x7B,
        mode=CAVE_MODE,
        link_x=144,
        link_y=93,
        health=0x22,
        objects=(leever,),
    )
    action = ctrl._after_hops(frozen)
    assert action.reason == "cave_settle"
    assert ctrl._cave_walker is None
    assert ctrl.success is False


def test_heart_l8_seeded_miss_stays_and_a_walled_cave_stands() -> None:
    """(112, 140) is the measured UP miss. A miss on the next cell blocks it."""
    ctrl = make_heart_l8_controller()
    ctrl._entry_containers = 3
    stall = _snap(
        screen=0x7B,
        mode=CAVE_MODE,
        link_x=112,
        link_y=141,
        health=0x22,
        objects=(_heart_sprite(),),
    )
    action = ctrl._after_hops(stall)
    walker = ctrl._cave_walker
    assert walker is not None
    assert (112, 140) in walker.grid.blocked
    assert (112, 140) not in walker.grid.inferred
    assert walker.last_dir == "DOWN"
    assert action.reason == "heart"
    assert action.action == nes_action("DOWN")
    assert ctrl.success is False

    missed = ctrl._after_hops(stall)
    assert (112, 142) in walker.grid.blocked
    assert missed.reason == "heart"
    assert ctrl._interior == 2

    for x in range(walker.grid.xmin, walker.grid.xmax + 1):
        for y in range(walker.grid.ymin, walker.grid.ymax + 1):
            walker.grid.blocked.add((x, y))
    walker.path = None
    ctrl._after_hops(stall)
    assert ctrl.success is False
    assert "heart_not_taken" in ctrl.notes
    assert ctrl._interior < ctrl.interior_budget


def test_heart_l8_aims_at_the_right_item_not_the_old_man() -> None:
    """0x7B is take-any: potion left, heart right. (120, 128) is the old man."""
    ctrl = make_heart_l8_controller()
    assert (ctrl.interior_x, ctrl.interior_y) == (152, 149)
    ctrl._entry_containers = 3
    stood = _snap(
        screen=0x7B,
        mode=CAVE_MODE,
        link_x=115,
        link_y=141,
        health=0x21,
        objects=(_heart_sprite(),),
    )
    action = ctrl._after_hops(stood)
    assert action.reason == "heart"
    # x=115 is off a column: DOWN would only slide him sideways. Row first.
    assert action.action == nes_action("RIGHT")


def test_heart_l8_red_pose_is_not_a_leave() -> None:
    from zelda_i.overworld.gather_run import written_leave

    assert SEGMENTS["heart_l8"].save_red is False
    assert written_leave("GatherHeartL8Leave", False, "/tmp/x.state", save_red=False) is None
    assert (
        written_leave("GatherHeartL8Leave", True, "/tmp/x.state", save_red=False)
        == "/tmp/x.state"
    )


def test_heart_m3_goes_down_the_west_column_before_east() -> None:
    """BFS_2C spawns at (0, 85). DOWN at x=144 would walk into the rock."""
    ctrl = make_heart_m3_controller()
    action = ctrl._after_hops(_snap(screen=0x2C, link_x=16, link_y=85, facing=4))
    assert action.reason.startswith("approach")
    assert ctrl._leg == 0
    ctrl._after_hops(_snap(screen=0x2C, link_x=16, link_y=165, facing=4))
    assert ctrl._leg == 1


def test_heart_m3_turns_up_before_the_bomb() -> None:
    """The measured doorway is the rock's bottom face at x 136..152."""
    ctrl = make_heart_m3_controller()
    ctrl._leg = len(ctrl.approach)
    facing_east = _snap(screen=0x2C, link_x=144, link_y=165, facing=1)
    action = ctrl._after_hops(facing_east)
    assert action.reason == "face_wall"
    assert action.action == nes_action("UP")
    assert ctrl._bombed == 0
    facing_up = _snap(screen=0x2C, link_x=144, link_y=165, facing=8)
    action = ctrl._after_hops(facing_up)
    assert action.reason == "place_bomb"
    assert action.action == nes_action("B")


def test_heart_m3_is_the_same_take_any_cave() -> None:
    ctrl = make_heart_m3_controller()
    assert (ctrl.interior_x, ctrl.interior_y) == (152, 149)
    assert SEGMENTS["heart_m3"].save_red is False
    ctrl._entry_containers = 3
    ctrl._after_hops(
        _snap(
            screen=0x2C,
            mode=CAVE_MODE,
            link_x=112,
            link_y=141,
            health=0x22,
            objects=(_heart_sprite(),),
        )
    )
    assert (112, 140) in ctrl._cave_walker.grid.blocked


def test_potion_shut_opening_fails_and_cave_finishes() -> None:
    ctrl = make_potion_controller()
    assert isinstance(ctrl, ArrivalController)
    aim = _snap(screen=0x64, mode=PLAY_MODE, link_x=128, link_y=77)
    assert ctrl._at_stop(aim) is False
    for _ in range(40):
        ctrl._after_hops(aim)
        if "opening_closed" in ctrl.notes:
            break
    assert ctrl.success is False
    assert "opening_closed" in ctrl.notes

    open_ctrl = make_potion_controller()
    cave = _snap(screen=0x64, mode=CAVE_MODE, link_x=128, link_y=77)
    assert open_ctrl._at_stop(cave) is False
    open_ctrl._after_hops(cave)
    assert open_ctrl.success is True


def test_every_waypoint_list_is_single_axis_moves() -> None:
    from zelda_i.overworld.gather_segments import WAYPOINTS

    for screen, corners in WAYPOINTS.items():
        for (ax, ay), (bx, by) in zip(corners, corners[1:]):
            assert (ax == bx) != (ay == by), hex(screen)


def test_heart_walk_goes_up_column_b_not_the_sealed_0x7d() -> None:
    assert _targets(HEART_WALK_HOPS) == (0x6B, 0x5B, 0x4B, 0x3B, 0x2B, 0x2C)
    assert 0x7D not in _targets(HEART_WALK_HOPS)
    assert HEART_WALK_HOPS[-1].align_y == 85


def test_candle_lateral_row_is_below_the_key_pedestal() -> None:
    """y=149 crosses the 100-rupee key at x~120 (keys 0->1, 101->1 rupees)."""
    ctrl = make_candle_controller()
    assert ctrl.buy_y > 149
    assert ctrl.buy_x == 152


def test_cave_exit_lines_up_then_clears_the_mouth() -> None:
    from zelda_i.overworld.gather_segments import CAVE_EXIT_CLEAR, CaveExitController

    ctrl = CaveExitController()
    act = ctrl.step(_snap(mode=CAVE_MODE, link_x=152, link_y=149))
    assert act.action == nes_action("LEFT")
    act = ctrl.step(_snap(mode=CAVE_MODE, link_x=112, link_y=149))
    assert act.action == nes_action("DOWN")
    out = ctrl.step(_snap(mode=PLAY_MODE, screen=0x0C, link_x=128, link_y=77))
    assert out.action == nes_action("DOWN") and ctrl.success is False
    ctrl.step(_snap(mode=PLAY_MODE, screen=0x0C, link_x=128, link_y=77 + CAVE_EXIT_CLEAR))
    assert ctrl.success is True


def test_chain_order_runs_bomb_shop_to_level_1_mouth() -> None:
    from zelda_i.overworld.gather_segments import chain_stages

    names = [name for name, _ in chain_stages()]
    assert names == [
        "exit_6f", "walk_7c", "heart_7b", "exit_7b", "walk_pond", "pond_39",
        "walk_2c", "heart_2c",
        "exit_2c", "ne_100", "exit_0f", "letter", "exit_0e", "candle",
        "exit_0c", "white", "back_1a", "walk_48", "select_candle",
        "rupees_48", "exit_48", "heart_47", "exit_47",
        "ring", "exit_ring", "ring_return", "walk_pond_l1", "pond_39_l1", "walk_37",
    ]
    stages = dict(chain_stages())
    assert _targets(stages["letter"].hops) == (0x1F, 0x1E, 0x0E)
    assert stages["select_candle"].want == 4
    assert _targets(stages["walk_37"].hops)[-1] == 0x37
    assert stages["ring"].price == 250
    assert _targets(stages["walk_pond_l1"].hops)[0] == 0x59
    # Burn caves exit by stairs: nothing to clear, DOWN would re-enter.
    assert stages["exit_47"].clear == 0 and stages["exit_48"].clear == 0
    assert stages["exit_0c"].clear > 0


def test_burn_stances_leave_the_flame_room_to_walk() -> None:
    """Flush with the tree ((192, 93) RIGHT) never reveals it; (188, 93) does."""
    from zelda_i.overworld.gather_segments import (
        make_burn_47_controller,
        make_burn_48_controller,
    )

    b48 = make_burn_48_controller()
    assert (b48.bomb_x, b48.bomb_y, b48.bomb_face) == (188, 93, "RIGHT")
    assert b48.retreat is None and b48.reward == "rupees" and b48.keeper == 0x7B
    b47 = make_burn_47_controller()
    assert (b47.bomb_x, b47.bomb_y, b47.bomb_face) == (176, 157, "DOWN")
    assert (b47.interior_x, b47.interior_y) == (152, 149)


def test_burn_waits_in_place_for_the_flame() -> None:
    from zelda_i.overworld.gather_segments import make_burn_48_controller

    ctrl = make_burn_48_controller()
    ctrl._bombed = 1
    act = ctrl._after_hops(_snap(screen=0x48, link_x=188, link_y=93, facing=1))
    assert act.reason == "flame_wait"
    assert act.action == nes_action()


def test_rupee_reward_stop_counts_from_cave_entry() -> None:
    from zelda_i.overworld.gather_segments import make_burn_48_controller

    ctrl = make_burn_48_controller()
    keeper = ZeldaObject(slot=1, type_id=0x7B, x=120, y=128, facing=0, hp=0, state=2)
    ctrl._after_hops(_snap(screen=0x48, mode=CAVE_MODE, rupees=42, objects=(keeper,)))
    assert ctrl._at_stop(_snap(screen=0x48, mode=CAVE_MODE, rupees=71)) is False
    assert ctrl._at_stop(_snap(screen=0x48, mode=CAVE_MODE, rupees=72)) is True


def test_return_walk_ignores_the_east_to_west_0x28_corners() -> None:
    from zelda_i.overworld.gather_segments import HopWalkController, waypoint_action

    ctrl = HopWalkController(hops=(ScreenHop(0x38, "DOWN", align_x=120),), waypoints={})
    assert waypoint_action(ctrl, _snap(screen=0x28, link_x=0, link_y=133)) is None


def test_cave_exit_idles_through_the_exit_mode() -> None:
    from zelda_i.overworld.gather_segments import CaveExitController

    ctrl = CaveExitController(clear=0)
    act = ctrl.step(_snap(mode=10, screen=0x47, link_x=112, link_y=221))
    assert act.action == nes_action() and ctrl.success is False
    ctrl.step(_snap(mode=PLAY_MODE, screen=0x47, link_x=176, link_y=157))
    assert ctrl.success is True


def test_heart_m3_fixes_the_row_before_walking_east() -> None:
    """Knocked to (72, 157) under the rock corner, RIGHT jammed 4916 frames."""
    ctrl = make_heart_m3_controller()
    ctrl._leg = len(ctrl.approach)
    act = ctrl._after_hops(_snap(screen=0x2C, link_x=72, link_y=157, facing=1))
    assert act.action[:] == ctrl._swing("DOWN", "bomb_cell").action[:]
    assert ctrl._bombed == 0
