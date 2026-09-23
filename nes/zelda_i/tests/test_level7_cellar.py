"""Durable L7 nose cellar 0x7B B→A cross: DOWN, floor LEFT, never source UP.

No emulator. Fake snapshots at the right-ladder spawn (192,93) must emit
DOWN, never UP (CheckSubroom AttrB → play 0x0D). Floor LEFT to x=48, then
UP the west ladder. OccupancyWalker is banned. Spine tip-stairs is the
walk-on INTO cellar 0x7B (`level7.stairs0d`, live 2/2). RAM claim dest is
play 0x29.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.dungeon.ids import GORIYA_OBJECT_TYPE
from zelda_i.level7.path import ROOM_1A
from zelda_i.level7.cellar import (
    CELLAR_ROOM,
    DEST_ROOM,
    EAST_X,
    EXIT_STAIRS,
    FLOOR_Y,
    MOUTH_Y,
    PIT_TILE,
    RAM_CLAIM,
    ROOM_4A,
    SOURCE_ROOM,
    SPAWN_XY,
    WEST_X,
    Level7NoseCellarCrossController,
    ROOM1A_RUNG_SCRIPTED,
    ROOM1A_RUNG_SOLVER,
    Room1ACandleController,
    room1a_segments,
    room1a_unkillable,
    make_nose_cellar_cross_controller,
    nose_cellar_cross_step,
    nose_cellar_cross_success,
)
from zelda_i.level7.hops import (
    make_aquamentus_heart_controller,
    make_level7_shard_leave_controller,
    make_nose_cellar_cross_controller as hops_make_nose_cellar_cross,
    make_tip_stairs_controller,
)
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    PASSAGE_MODE,
    PLAY_MODE,
    ZeldaObject,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PASSAGE_MODE,
    "level": 7,
    "screen": CELLAR_ROOM,
    "x": SPAWN_XY[0],
    "y": SPAWN_XY[1],
    "tile": 36,
    "keys": 2,
    "bombs": 6,
    "candle": 2,
    "triforce": 0,
}

DOWN = list(nes_action("DOWN"))
LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "cellar controllers must not write RAM"
    return act


def test_ram_claim_is_play_0x29_not_0x0d() -> None:
    assert DEST_ROOM == 0x29
    assert SOURCE_ROOM == 0x0D
    assert "0x29" in RAM_CLAIM
    assert "0x0D" in RAM_CLAIM
    assert "Never UP at x>=$80" in RAM_CLAIM


def test_right_spawn_goes_down_never_up() -> None:
    snap = read_snapshot(_ram(x=EAST_X, y=MOUTH_Y))
    act = nose_cellar_cross_step(snap)
    assert act.reason == "cellar_east_drop"
    assert list(act.action) == DOWN
    assert list(act.action) != UP

    ctl = make_nose_cellar_cross_controller()
    act = _step(ctl, _ram(x=EAST_X, y=MOUTH_Y))
    assert not ctl.failed
    assert ctl.arrival_seen
    assert act.reason == "cellar_east_drop"
    assert list(act.action) == DOWN
    assert list(act.action) != UP


def test_mid_top_channel_goes_east_never_up() -> None:
    """Previous miss: UP at any x on the top channel returns to 0x0D."""
    snap = read_snapshot(_ram(x=160, y=MOUTH_Y))
    act = nose_cellar_cross_step(snap)
    assert act.reason == "cellar_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != UP
    assert list(act.action) != LEFT


def test_floor_walks_west() -> None:
    act = nose_cellar_cross_step(read_snapshot(_ram(x=100, y=FLOOR_Y)))
    assert act.reason == "cellar_floor_west"
    assert list(act.action) == LEFT
    assert list(act.action) != UP

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=100, y=FLOOR_Y))
    assert not ctl.failed
    assert ctl.on_floor
    assert act.reason == "cellar_floor_west"
    assert list(act.action) == LEFT


def test_west_floor_climbs() -> None:
    act = nose_cellar_cross_step(read_snapshot(_ram(x=WEST_X, y=FLOOR_Y)))
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=WEST_X, y=FLOOR_Y))
    assert not ctl.failed
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP


def test_west_climb_holds_up() -> None:
    act = nose_cellar_cross_step(read_snapshot(_ram(x=WEST_X, y=120)))
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=WEST_X, y=120))
    assert not ctl.failed
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP


def test_west_lip_keeps_up() -> None:
    snap = read_snapshot(_ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x6F))
    act = nose_cellar_cross_step(snap)
    assert act.reason == "cellar_west_lip"
    assert list(act.action) == UP

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x6F))
    assert not ctl.failed
    assert act.reason == "cellar_west_lip"
    assert list(act.action) == UP


def test_stairs_tile_idles_exit_warp() -> None:
    snap = read_snapshot(_ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x71))
    act = nose_cellar_cross_step(snap)
    assert act.reason == "cellar_exit_warp"
    assert list(act.action) == IDLE


def test_pit_tile_250_fails_closed_idle() -> None:
    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=112, y=141, tile=PIT_TILE))
    assert ctl.failed and not ctl.success
    assert "pit_tile_250" in ctl.notes
    assert act.reason == "pit_tile_250"
    assert list(act.action) == IDLE


def test_west_floor_tile_250_still_climbs() -> None:
    """C1: (48,189) reported tile 250; that is not the y=141 pit."""
    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=WEST_X, y=FLOOR_Y, tile=PIT_TILE))
    assert not ctl.failed
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP


def test_pit_step_goes_east_not_left() -> None:
    act = nose_cellar_cross_step(read_snapshot(_ram(x=112, y=141, tile=PIT_TILE)))
    assert act.reason == "cellar_pit_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != LEFT


def test_source_ladder_up_fails() -> None:
    """UP at the right/source ladder is CheckSubroom AttrB → 0x0D."""
    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    ctl.on_floor = True
    # Force a west-climb reason at source x by being on west? No: at east floor
    # the step emits LEFT, not UP. Climb at east mid-height is east_drop DOWN.
    # Simulate a west-climb-shaped pose that is actually still on the source
    # column at mouth y after a mistaken UP: policy checks UP reasons.
    # Direct: emerge to 0x0D play is the miss.
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=SOURCE_ROOM, x=96, y=157))
    assert ctl.failed and not ctl.success
    assert "returned_source_0x0d" in ctl.notes
    assert act.reason == "returned_source_0x0d"
    assert list(act.action) == IDLE


def test_emerge_requires_exact_attr_a_0x29() -> None:
    emerge = _ram(mode=PLAY_MODE, screen=DEST_ROOM, x=96, y=157)
    assert nose_cellar_cross_success(read_snapshot(emerge))
    still = _ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM)
    assert not nose_cellar_cross_success(read_snapshot(still))
    back = _ram(mode=PLAY_MODE, screen=SOURCE_ROOM, x=96, y=157)
    assert not nose_cellar_cross_success(read_snapshot(back))

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, emerge)
    assert ctl.success and not ctl.failed
    assert act.reason == "emerged_0x29"


def test_wrong_play_fails() -> None:
    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=0x1A, x=96, y=157))
    assert ctl.failed and not ctl.success
    assert "wrong_play_0x1a" in ctl.notes
    assert list(act.action) == IDLE


def test_factory_report_is_fixture_live_not_route_eligible() -> None:
    ctl = make_nose_cellar_cross_controller()
    assert isinstance(ctl, Level7NoseCellarCrossController)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["dest_screen"] == DEST_ROOM == 0x29
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["door"] == "STAIRS"
    assert report["spec_id"] == "level7_nose_cellar_0x7b"
    assert "OccupancyWalker" not in (ctl.__class__.__module__ + ctl.__class__.__name__)


def test_hops_factory_is_the_live_cross_not_the_spine_tip() -> None:
    """0x0D->0x7B and 0x7B->0x29 are two different live hops.

    ``make_tip_stairs_controller`` is the walk-on INTO cellar 0x7B
    (`level7.stairs0d`, live 2/2); the cross OUT of it to play 0x29 stays
    ``Level7NoseCellarCrossController``. The boss/leave factories are live
    2/2 as of rr-8t4.3 (`20260904_W3`-`W6`) but stay ``route_eligible=false``.
    """
    from zelda_i.level7.stairs0d import Level7Stairs0DController

    live = hops_make_nose_cellar_cross()
    assert isinstance(live, Level7NoseCellarCrossController)
    spine = make_tip_stairs_controller()
    assert isinstance(spine, Level7Stairs0DController)
    assert not isinstance(spine, Level7NoseCellarCrossController)
    assert spine.report()["dest_screen"] == 0x7B
    assert spine.report()["route_eligible"] is False
    from zelda_i.level7.aquamentus import Level7AquamentusHeartController
    from zelda_i.level7.shard import Level7ShardLeaveController

    aqua = make_aquamentus_heart_controller()
    leave = make_level7_shard_leave_controller()
    assert isinstance(aqua, Level7AquamentusHeartController)
    assert isinstance(leave, Level7ShardLeaveController)
    assert aqua.report()["route_eligible"] is False
    assert leave.report()["route_eligible"] is False


def test_no_occupancy_walker_import() -> None:
    import ast

    import zelda_i.level7.cellar as cellar

    assert not hasattr(cellar, "OccupancyWalker")
    source = cellar.__file__
    assert source is not None
    tree = ast.parse(open(source, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "OccupancyWalker" not in imported
    assert "zelda_i.walk.physics" not in imported


def test_candle_pin_already_red_never_greens() -> None:
    ctl = Room1ACandleController()
    act = _step(ctl, _ram(screen=ROOM_4A, mode=PASSAGE_MODE, x=135, y=141, candle=2))
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "already_red_candle"


def test_candle_cellar_drops_from_ladder_not_up() -> None:
    ctl = Room1ACandleController()
    act = _step(ctl, _ram(screen=ROOM_4A, mode=PASSAGE_MODE, x=96, y=93, candle=0))
    assert not ctl.failed
    assert not ctl.success
    assert act.reason == "cellar_drop"
    assert list(act.action) == DOWN
    assert list(act.action) != UP


def test_candle_cellar_keeps_dropping_until_the_floor() -> None:
    # blue_ring_full_poweron2: DOWN stopped at y=181 on the rungs, RIGHT was
    # dead there, and keese knocked Link back up for 36k frames.
    ctl = Room1ACandleController()
    ram = _ram(screen=ROOM_4A, mode=PASSAGE_MODE, x=48, y=181, candle=1)
    act = _step(ctl, ram)
    assert act.reason == "cellar_drop"
    assert list(act.action) == DOWN


def test_candle_cellar_does_not_oscillate_at_y180() -> None:
    ctl = Room1ACandleController()
    ram = _ram(screen=ROOM_4A, mode=PASSAGE_MODE, x=96, y=93, candle=0)
    assert _step(ctl, ram).reason == "cellar_drop"
    ram[ADDR_LINK_Y] = 189
    assert _step(ctl, ram).reason == "cellar_east"
    ram[ADDR_LINK_X] = 172
    act = _step(ctl, ram)
    assert act.reason == "cellar_climb"
    assert list(act.action) == UP
    ram[ADDR_LINK_Y] = 179
    act = _step(ctl, ram)
    assert act.reason == "cellar_climb"
    assert list(act.action) == UP
    assert list(act.action) != DOWN


def test_candle_rising_edge_greens() -> None:
    ctl = Room1ACandleController()
    ram = _ram(screen=ROOM_4A, mode=PASSAGE_MODE, x=124, y=141, candle=0)
    _step(ctl, ram)
    assert not ctl.success
    ram[ADDR_CANDLE] = 2
    act = _step(ctl, ram)
    assert ctl.success
    assert not ctl.failed
    assert act.reason == "red_candle_natural"


# --- 0x1A: the searched clear as a rung ---------------------------------


def _room1a_ram(**fields: int) -> np.ndarray:
    """0x1A in live play with one full-HP goriya east of Link."""
    ram = _ram(screen=ROOM_1A, mode=PLAY_MODE, x=100, y=100, candle=0, **fields)
    ram[ADDR_LINK_X + 1] = 108
    ram[ADDR_LINK_Y + 1] = 100
    ram[ADDR_OBJ_TYPE + 1] = GORIYA_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 48
    return ram


class _StubEnv:
    """Enough of an env to construct a ``Rollout``. Never stepped."""

    class _Em:
        def get_state(self):  # pragma: no cover - a budgeted-out solver
            raise AssertionError("a budgeted-out solver must not touch the core")

    em = _Em()

    def get_ram(self):  # pragma: no cover - same
        raise AssertionError("a budgeted-out solver must not read RAM")


def test_room1a_rung_order_puts_the_search_above_the_position_table() -> None:
    """Structural. Precedence is a number, not a source line."""
    assert ROOM1A_RUNG_SOLVER < ROOM1A_RUNG_SCRIPTED
    names = [r.name for r in Room1ACandleController().clear_arbiter.rungs]
    assert names == ["room1a_solver", "room1a_scripted"]


def test_room1a_ladder_is_the_old_chain_with_nothing_bound() -> None:
    """Behavioural. The solver is off by default, so every frame is scripted."""
    ctl = Room1ACandleController()
    act = _step(ctl, _room1a_ram())
    assert not ctl.solver_clear
    assert act.reason in ("goriya_slash", "goriya_face")
    assert ctl.clear_arbiter.census() == {"room1a_solver": 0, "room1a_scripted": 1}
    assert ctl.report()["solver"] is None


def test_room1a_solver_past_its_room_budget_drops_to_the_scripted_clear() -> None:
    """Behavioural. The card's drop-out, wired: a bound solver whose room
    budget is spent declines every frame, and the position table drives the
    room exactly as it does today."""
    ctl = Room1ACandleController()
    solver = ctl.attach_solver(_StubEnv(), room_budget=0)
    assert ctl.solver_clear
    act = _step(ctl, _room1a_ram())
    assert act.reason in ("goriya_slash", "goriya_face")
    assert ctl.clear_arbiter.census() == {"room1a_solver": 0, "room1a_scripted": 1}
    assert solver.declines["room_budget"] == 1
    assert solver.report()["rollouts"] == 0
    # The latch the push phase reads has to stay honest whoever owns the frame.
    assert ctl.saw_goriya


def test_room1a_solver_declines_the_push_and_the_cellar() -> None:
    """Behavioural. A latching phase is a gate inside the rung, not a priority:
    the block push and the stairs walk have no enemy for a search to plan
    against."""
    ctl = Room1ACandleController()
    solver = ctl.attach_solver(_StubEnv(), room_budget=0)
    ctl.saw_goriya = True
    _step(ctl, _ram(screen=ROOM_1A, mode=PLAY_MODE, x=96, y=180, candle=0))
    _step(ctl, _ram(screen=ROOM_4A, mode=PASSAGE_MODE, x=96, y=93, candle=0))
    assert solver.declines == {}
    assert ctl.clear_arbiter.census()["room1a_solver"] == 0


def test_room1a_unkillable_drops_the_block_and_the_sealed_centre() -> None:
    """Behavioural. The ``0x68`` pushable block sits in the object table with
    the wave, and a goriya inside the sealed diamond cross can never be cut —
    a goal that waits for either never fires."""
    block = ZeldaObject(slot=1, type_id=0x68, x=96, y=136, facing=0, hp=1, state=0)
    sealed = ZeldaObject(
        slot=2, type_id=GORIYA_OBJECT_TYPE, x=128, y=144, facing=0, hp=48, state=0
    )
    reachable = ZeldaObject(
        slot=3, type_id=GORIYA_OBJECT_TYPE, x=64, y=100, facing=0, hp=48, state=0
    )
    assert room1a_unkillable(block)
    assert room1a_unkillable(sealed)
    assert not room1a_unkillable(reachable)


def test_room1a_alphabet_has_the_swings_and_no_standing() -> None:
    """Behavioural. Passivity is this room's measured failure: slots 4/5 reach
    the sealed centre if they are not engaged early."""
    labels = [a.label for a in room1a_segments()]
    assert [x for x in labels if x.startswith("swing_")]
    assert not [x for x in labels if x.startswith("stand")]


def test_attach_solver_passes_the_budget_to_the_constructor() -> None:
    """Behavioural, and the dataclass trap: a default cannot be overridden
    after the class exists, so every knob goes through ``__init__``."""
    ctl = Room1ACandleController()
    solver = ctl.attach_solver(_StubEnv(), room_budget=7)
    assert solver.room_budget == 7
    assert solver.objective.unreachable is room1a_unkillable
    assert solver.reason_prefix == "candle_solve"
