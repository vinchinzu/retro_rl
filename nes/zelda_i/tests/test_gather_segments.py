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
    RING_HOPS,
    RING_RETURN_HOPS,
    RING_PRICE,
    WHITE_HOPS,
    BombWallController,
    CaveMouthController,
    NortheastController,
    main,
    make_candle_controller,
    make_heart_l8_controller,
    make_heart_m3_controller,
    make_letter_controller,
    make_ring_controller,
    make_white_controller,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import CAVE_MODE, PLAY_MODE, ZeldaObject, ZeldaSnapshot

WALLET_MAX = 255  # $066D saturates; the HUD stops counting there


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
    assert ctrl._at_stop(_snap(screen=0x0A, mode=CAVE_MODE, sword=2)) is True


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


def test_heart_m3_bombs_only_facing_the_rock() -> None:
    """The measured doorway is the rock's bottom face at x 136..152; a bomb
    dropped facing east lands on open sand (0x2C's first try)."""
    ctrl = make_heart_m3_controller()
    facing_east = _snap(screen=0x2C, link_x=144, link_y=165, facing=1)
    action = ctrl._after_hops(facing_east)
    assert action.action != nes_action("B")
    assert ctrl._bombed == 0
    ctrl._back_frames = 0
    facing_up = _snap(screen=0x2C, link_x=144, link_y=165, facing=8)
    action = ctrl._after_hops(facing_up)
    assert action.reason == "place_bomb"
    assert action.action == nes_action("B")


def test_heart_m3_is_the_same_take_any_cave() -> None:
    ctrl = make_heart_m3_controller()
    assert (ctrl.interior_x, ctrl.interior_y) == (152, 149)
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


def test_cave_exit_is_done_when_link_is_not_in_a_cave() -> None:
    # A help-drop bomb ends the coast walk on 0x7F: no 0x6F cave to leave.
    from zelda_i.overworld.gather_segments import CaveExitController

    ctrl = CaveExitController()
    act = ctrl.step(_snap(mode=PLAY_MODE, screen=0x7F, link_x=96, link_y=141))
    assert ctrl.success is True and act.action == nes_action()


def test_cell_nudge_presses_along_the_shared_row() -> None:
    # 0x48's burn stand x=188 is off the turn lattice: the nudge walks the
    # row toward it instead of flipping around the nearest column.
    from zelda_i.overworld.gather_segments import _nudge_dir

    assert _nudge_dir(_snap(link_x=184, link_y=93), (188, 93)) == "RIGHT"
    assert _nudge_dir(_snap(link_x=185, link_y=93), (188, 93)) == "RIGHT"
    assert _nudge_dir(_snap(link_x=80, link_y=91), (80, 85)) == "UP"
    assert _nudge_dir(_snap(link_x=72, link_y=93), (80, 85)) is None


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
        "exit_6f", "walk_7c", "heart_7b", "exit_7b",
        "walk_pond", "pond_39", "walk_2c", "walk_2c_direct",
        "heart_2c", "exit_2c", "rupees_2d", "exit_2d",
        "ne_100", "exit_0f", "letter", "exit_0e",
        "walk_0d", "bomb_0d", "potion_0d", "exit_0d",
        "candle", "exit_0c", "select_candle",
        "white", "back_1a", "walk_28", "rupees_28", "exit_28",
        "walk_48", "rupees_48", "exit_48", "heart_47", "exit_47",
        "walk_5b", "rupees_5b", "exit_5b", "rupees_6b", "exit_6b",
        "walk_56", "rupees_56", "exit_56",
        "ring", "exit_ring", "bait_first", "exit_bait_first",
        "rupees_62", "exit_62",
        "bait", "exit_bait", "ring_second", "exit_ring_second",
        "potion_restock_gather", "exit_potion_gather", "ring_return",
        "walk_pond_l1", "pond_39_l1", "walk_37", "walk_37_direct",
    ]
    stages = {name: getattr(ctl, "inner", ctl) for name, ctl in chain_stages()}
    assert _targets(stages["letter"].hops) == (0x1F, 0x1E, 0x0E)
    assert stages["select_candle"].want == 4
    assert _targets(stages["walk_37"].hops)[-1] == 0x37
    assert _targets(stages["walk_37_direct"].hops) == (0x48, 0x38, 0x37)
    assert _targets(stages["white"].hops) == (0x1C, 0x1B, 0x1A)
    assert stages["ring"].price == stages["ring_second"].price == 250
    assert stages["bait"].price == stages["bait_first"].price == 60
    assert stages["bait"].farm_below_hearts == stages["bait_first"].farm_below_hearts == 0
    assert _targets(stages["bait"].hops) == (0x52, 0x53, 0x54, 0x44, 0x34)
    assert _targets(stages["ring_return"].hops)[0] == 0x44
    assert _targets(stages["walk_pond_l1"].hops)[0] == 0x59
    assert stages["heart_2c"].reward == "container"
    # Burn caves exit by stairs: nothing to clear, DOWN would re-enter.
    for name, ctl in stages.items():
        if name.startswith("rupees_") and not ctl.consumes_bomb:
            assert stages["exit_" + name[len("rupees_"):]].clear == 0, name
    assert stages["exit_47"].clear == 0
    assert stages["exit_0c"].clear > 0 and stages["exit_2d"].clear > 0


def test_every_take_any_gives_its_heart() -> None:
    """100%: 0x7B, 0x2C and 0x47 each give the container, never the potion."""
    from zelda_i.overworld.gather_segments import chain_stages

    stages = {name: getattr(ctl, "inner", ctl) for name, ctl in chain_stages()}
    for name in ("heart_7b", "heart_2c", "heart_47"):
        assert stages[name].reward == "container", name
    assert all(
        getattr(ctl, "reward", "") != "potion" for ctl in stages.values()
    )


def test_pond_and_direct_branches_share_one_latch() -> None:
    """Full hearts skip the pond loop and walk direct; hurt, the reverse."""
    from zelda_i.overworld.gather_segments import chain_stages

    def branch(health: int) -> list[str]:
        rows = chain_stages()
        names = [n for n, _ in rows]
        snap = _snap(screen=0x7B, health=health)
        played = []
        for name in ("walk_pond", "pond_39", "walk_2c", "walk_2c_direct"):
            leg = rows[names.index(name)][1]
            if leg.plan.decide(snap):
                played.append(name)
        return played

    assert branch(0x33) == ["walk_2c_direct"]  # 4/4 hearts
    assert branch(0x31) == ["walk_pond", "pond_39", "walk_2c"]  # 2/4


def test_secret_payouts_fund_the_ring_and_bait_without_drops() -> None:
    """The 0x34 buys are paid from hidden rupees, not a wallet write.

    Payouts are the ROM's (``SECRET_RUPEE_CAVES``, 0x0F's 100 on the NE
    walk). Enemy drops are margin, not budget: with none at all the wallet
    reaches 0x34 short of the ring, so the first visit buys the Bait and
    0x62's 100R pays the ring on the second. The wallet caps at 255, so a
    payout past that is lost, not banked.
    """
    from zelda_i.overworld.cave_shop import BLUE_POTION_PRICE
    from zelda_i.overworld.gather_segments import SECRET_REWARD, chain_stages

    budget = 0
    bought: list[str] = []
    ring_first: bool | None = None
    for name, ctl in chain_stages():
        inner = getattr(ctl, "inner", ctl)
        if name == "ne_100":
            budget += SECRET_REWARD
        elif name == "potion_0d":
            budget -= BLUE_POTION_PRICE
        elif name == "candle":
            budget -= CANDLE_PRICE
        elif name in ("ring", "bait_first", "bait", "ring_second"):
            if ring_first is None:
                ring_first = budget >= RING_PRICE
            if name in (("ring", "bait") if ring_first else ("bait_first", "ring_second")):
                assert budget >= inner.price, (name, budget)
                budget -= inner.price
                bought.append(name)
        if isinstance(inner, BombWallController) and inner.reward == "rupees":
            assert budget < WALLET_MAX, f"{name} pays into a full wallet"
            budget = min(WALLET_MAX, budget + inner.reward_rupees)
    assert bought == ["bait_first", "ring_second"]
    assert budget >= 0


def test_bomb_cell_turn_steps_back_and_comes_in_facing() -> None:
    """On the cell facing the wrong way, a turn in place walks Link on
    toward the target (0x48: flush with the tree at (192, 93), which never
    reveals) and re-walking the cell swapped UP/DOWN for 2000 frames on
    0x2D. So: step back off the cell, re-approach along the face axis
    (which faces it), then bomb. No press ever heads back past the cell."""
    from zelda_i.overworld.gather_segments import make_secret_rupee_controller

    ctrl = make_secret_rupee_controller(0x2D)
    x, y = ctrl.bomb_x, ctrl.bomb_y
    presses = []
    pose, facing = (x, y), 1  # on the cell, facing RIGHT
    for _ in range(30):
        act = ctrl._after_hops(_snap(screen=0x2D, link_x=pose[0], link_y=pose[1], facing=facing))
        if act.action == nes_action("B"):
            break
        press = next(d for d in ("UP", "DOWN", "LEFT", "RIGHT") if act.action == nes_action(d))
        presses.append(press)
        dx, dy = {"UP": (0, -2), "DOWN": (0, 2), "LEFT": (-2, 0), "RIGHT": (2, 0)}[press]
        pose, facing = (pose[0] + dx, max(y - 2, pose[1] + dy)), {"UP": 8, "DOWN": 4, "LEFT": 2, "RIGHT": 1}[press]
    else:
        raise AssertionError(f"never bombed: {presses}")
    assert set(presses) == {"DOWN", "UP"} and presses[0] == "DOWN"
    assert presses.index("UP") > presses.count("DOWN") - 1  # back, then in


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




def test_cli_takes_one_token_and_rejects_unknown_stages() -> None:
    assert main([]) == 2
    assert main(["nope"]) == 2
    assert main(["chain:not_a_stage"]) == 2
