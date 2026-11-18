"""Saddle shipping: horse reads, stands, the park plan and the park state machine (no ROM)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from harvest.core.npc_catalog import GOBJ_INITIALIZED, GOBJ_STRUCT_BASE, GOBJ_STRUCT_STRIDE
from harvest.core.ram_catalog import field_spec, live_wram_base
from harvest.tasks import horse
from harvest.tasks.harvest_task import HarvestTask
from harvest.tasks.nav import make_action

from day_plan_test_helpers import make_navigation_ram, set_player_pos

BIN = (11, 30)
HORSE = (17, 20)


def _set(ram: np.ndarray, key: str, value: int) -> None:
    spec = field_spec(key)
    base = spec.address + (live_wram_base(ram) if spec.live_offset else 0)
    for i in range({"u8": 1, "u16": 2, "u24": 3}[spec.kind]):
        ram[base + i] = (value >> (8 * i)) & 0xFF


def _put_horse(ram: np.ndarray, px: int, py: int, sprite: int = 0x0200, slot: int = 1) -> None:
    base = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    for offset, value in ((0, GOBJ_INITIALIZED), (2, sprite), (8, px), (10, py)):
        ram[base + offset] = value & 0xFF
        ram[base + offset + 1] = value >> 8


def _world_ram(**fields) -> np.ndarray:
    ram = np.zeros(0x24000, dtype=np.uint8)
    defaults = dict(tilemap=0, horse_age=horse.HORSE_ADULT_AGE, hour=10, minute=0, player_state=1, input_lock=1)
    for key, value in (defaults | fields).items():
        _set(ram, key, value)
    _put_horse(ram, 273, 320)
    return ram


@pytest.fixture(autouse=True)
def _saddle_on(monkeypatch):
    monkeypatch.setenv("HARVEST_SADDLE", "1")


def test_the_horse_is_found_by_its_sprite_on_the_farm():
    ram = _world_ram()
    assert horse.farm_horse(ram) == HORSE
    assert horse.saddle_usable(ram) == HORSE


@pytest.mark.parametrize(
    "fields",
    [
        dict(horse_age=0x10),  # still a foal
        dict(hour=17),  # the ROM pays nothing from 17:00
        dict(hour=16, minute=58),  # a throw takes two game minutes
        dict(player_state=0x11),  # riding
        dict(tilemap=0x27),  # indoors
    ],
)
def test_no_saddle_shipping_when_the_rom_would_not_pay(fields):
    assert horse.saddle_usable(_world_ram(**fields)) is None


def test_saddle_shipping_is_opt_in(monkeypatch):
    monkeypatch.delenv("HARVEST_SADDLE", raising=False)
    assert horse.saddle_usable(_world_ram()) is None  # off until an A/B shows a gain
    monkeypatch.setenv("HARVEST_SADDLE", "0")
    assert horse.saddle_usable(_world_ram()) is None


def test_a_dog_or_a_hen_is_not_the_horse():
    ram = _world_ram()
    _put_horse(ram, 273, 320, sprite=0x01E1)  # dog frame
    assert horse.farm_horse(ram) is None
    _put_horse(ram, 273, 320, sprite=0x0212)  # first hen
    assert horse.farm_horse(ram) is None


def test_saddle_bit_and_riding_read_the_right_bits():
    ram = _world_ram()
    assert not horse.saddle_ready(ram)
    _set(ram, "event_flags_1f5c", 0x0010)
    assert horse.saddle_ready(ram)
    assert not horse.is_riding(ram)
    _set(ram, "player_state", 0x11)
    assert horse.is_riding(ram)


def test_stands_run_north_west_south_east_and_skip_closed_tiles():
    ram = _world_ram()
    everything = lambda tile: True
    assert horse.horse_stands(ram, HORSE, open_tile=everything) == [
        ((17, 19), "down"), ((16, 20), "right"), ((17, 21), "up"), ((18, 20), "left"),
    ]
    no_north = lambda tile: tile != (17, 19)
    assert horse.horse_stands(ram, HORSE, open_tile=no_north)[0] == ((16, 20), "right")


# --- the park plan ---------------------------------------------------------

EAST_RING = [(40 + dx, 32 + dy) for dx in (0, 1, 2) for dy in (0, 1, 2)]  # nine stands 30+ tiles from the bin
WEST_RING = [(8 + dx, 34 + dy) for dx in (0, 1, 2) for dy in (0, 1)]  # six stands 5-8 tiles from the bin
open_everywhere = lambda tile: 0 <= tile[0] < 64 and 0 <= tile[1] < 64


def test_far_east_crops_are_worth_parking_the_horse_beside():
    plan = horse.plan_park(EAST_RING, bin_stand=BIN, here=(8, 24), horse=HORSE, open_tile=open_everywhere)
    assert plan is not None
    px, py = plan.park
    assert 36 <= px <= 46 and 29 <= py <= 36
    assert plan.saving_tiles >= horse.PARK_MIN_SAVING_TILES
    assert plan.saving_tiles >= horse.PARK_MIN_RATIO * plan.cost_tiles


def test_crops_near_the_bin_are_not_worth_a_ride():
    assert horse.plan_park(WEST_RING, bin_stand=BIN, here=(8, 24), horse=HORSE, open_tile=open_everywhere) is None


def test_a_handful_of_far_crops_does_not_pay_for_the_ride():
    assert horse.plan_park(EAST_RING[:3], bin_stand=BIN, here=(8, 24), horse=HORSE, open_tile=open_everywhere) is None


def test_the_park_tile_and_the_tile_south_of_it_must_both_be_open():
    # The farmer steps off one tile south of the horse, so a park tile with a wall below is out.
    walled_below = lambda tile: open_everywhere(tile) and tile[1] < 33
    plan = horse.plan_park(EAST_RING, bin_stand=BIN, here=(8, 24), horse=HORSE, open_tile=walled_below)
    assert plan is not None and plan.park[1] < 32


def test_a_far_away_farmer_and_a_far_away_park_raise_the_cost():
    near = horse.plan_park(EAST_RING, bin_stand=BIN, here=(18, 21), horse=HORSE, open_tile=open_everywhere)
    far = horse.plan_park(EAST_RING, bin_stand=BIN, here=(60, 60), horse=HORSE, open_tile=open_everywhere)
    assert near is not None
    assert far is None or far.cost_tiles > near.cost_tiles


# --- the park state machine ------------------------------------------------


def _arrived(_ram, _goal):
    return None, False


def _run(park: horse.ParkHorse, ram: np.ndarray, script, walk=_arrived, frames: int = 2000):
    """Step the machine; ``script(frame, ram, action)`` plays the game's part. Returns (outcome, actions)."""
    actions = []
    for frame in range(frames):
        action, outcome = park.step(ram, walk)
        actions.append(action)
        if outcome is not None:
            return outcome, actions
        script(frame, ram, action)
    raise AssertionError("park never finished")


def _a_down(action) -> bool:
    return action is not None and bool(action[8])  # SNES pad index of A (snes_action order B,Y,Sel,Start,U,D,L,R,A)


def _snes_a_index() -> int:
    probe = make_action(a=True)
    return int(np.flatnonzero(probe)[0])


def _make_script(*, raise_bit_after=4, mount_after_a=True, dismount_after_a=True):
    state = {"a_frames": 0, "mounted": False}
    a_index = _snes_a_index()

    def script(frame, ram, action):
        if frame >= raise_bit_after and not state["mounted"]:
            _set(ram, "event_flags_1f5c", 0x0010)
        if action is not None and action[a_index]:
            state["a_frames"] += 1
            if not state["mounted"] and mount_after_a and state["a_frames"] >= 2:
                state["mounted"] = True
                _set(ram, "player_state", 0x11)
            elif state["mounted"] and dismount_after_a and state["a_frames"] >= 4:
                state["mounted"] = False
                _set(ram, "player_state", 0x01)
    return script, state


def _plan() -> horse.ParkPlan:
    return horse.ParkPlan(park=(41, 33), saving_tiles=400, cost_tiles=60)


def test_park_walks_to_the_horse_faces_it_mounts_rides_and_dismounts():
    ram = _world_ram()
    script, state = _make_script()
    outcome, actions = _run(horse.ParkHorse(_plan()), ram, script)
    assert outcome == "done"
    assert not horse.is_riding(ram)
    assert state["a_frames"] >= 4


def test_park_never_holds_a_longer_than_a_pulse():
    ram = _world_ram()
    script, _ = _make_script()
    _outcome, actions = _run(horse.ParkHorse(_plan()), ram, script)
    a_index = _snes_a_index()
    longest = run = 0
    for action in actions:
        run = run + 1 if action is not None and action[a_index] else 0
        longest = max(longest, run)
    # A filter that rewrites actions must never hold a button; the machine's own pulses are 3 frames.
    assert 1 <= longest <= 3


def test_park_gives_up_on_foot_when_the_horse_never_raises_the_bit():
    ram = _world_ram()
    outcome, _ = _run(horse.ParkHorse(_plan()), ram, lambda frame, ram, action: None)
    assert outcome.startswith("abandon")
    assert not horse.is_riding(ram)


def test_park_gives_up_when_there_is_no_horse_out():
    ram = _world_ram(tilemap=0x27)
    outcome, _ = _run(horse.ParkHorse(_plan()), ram, lambda *_: None)
    assert outcome == "abandon: no horse on the farm"


def test_a_rider_who_cannot_reach_the_park_tile_dismounts_before_giving_up():
    ram = _world_ram()
    script, _state = _make_script()
    stuck = lambda ram, goal: (make_action(right=True), True) if horse.is_riding(ram) else (None, False)
    outcome, _ = _run(horse.ParkHorse(_plan()), ram, script, walk=stuck, frames=4000)
    assert outcome in ("done", "abandon: park timeout") or outcome.startswith("abandon")
    assert not horse.is_riding(ram)


# --- the harvest task hooks --------------------------------------------------


def _harvest_with_horse(**fields):
    ram = make_navigation_ram(current_tile=(20, 26))
    for key, value in (dict(horse_age=horse.HORSE_ADULT_AGE, hour=10, minute=0, player_state=1, input_lock=1) | fields).items():
        _set(ram, key, value)
    _put_horse(ram, 273, 320)
    task = HarvestTask()
    task._navigator.update(ram)
    return task, ram


def test_ship_options_offer_the_horse_stands_beside_the_bin_stands():
    task, ram = _harvest_with_horse()
    task._reset_ship_options(ram)
    stands = [stand for stand, _face in task._ship_options]
    assert BIN in stands
    assert set(task._horse_stands) <= set(stands) and task._horse_stands
    assert task._horse_at == HORSE


def test_a_horse_throw_waits_for_the_bit_then_throws():
    task, ram = _harvest_with_horse()
    _set(ram, "player_state", 0x03)  # carrying
    _set(ram, "held_item", 0x10)
    task._reset_ship_options(ram)
    stand = next(iter(task._horse_stands))
    task._ship_options = [(stand, task._horse_stands[stand])]
    task._phase, task._horse_at = "horse_face", HORSE
    world = type("W", (), {"ram": ram})()
    a_index = _snes_a_index()
    for _ in range(10):  # bit down: face the horse, never press A
        result = task._step_horse_face(world)
        assert not result.action.action[a_index]
    assert task._phase == "horse_face"
    _set(ram, "event_flags_1f5c", 0x0010)
    task._step_horse_face(world)
    assert task._phase == "ship_verify"
    assert [bool(a[a_index]) for a in list(task._action_queue)[:5]] == [False, False, False, True, True]


def test_a_horse_that_never_raises_the_bit_hands_the_route_back_to_the_bin():
    task, ram = _harvest_with_horse()
    _set(ram, "player_state", 0x03)
    task._reset_ship_options(ram)
    stand = next(iter(task._horse_stands))
    task._ship_options = [(stand, task._horse_stands[stand])]
    task._phase, task._horse_at = "horse_face", HORSE
    world = type("W", (), {"ram": ram})()
    for _ in range(horse.FACE_FRAMES + 2):
        task._step_horse_face(world)
    assert task._saddle_off and task._phase == "ship_nav" and not task._horse_stands
    task._reset_ship_options(ram)
    assert not task._horse_stands and BIN in [s for s, _ in task._ship_options]


def test_a_rider_in_the_harvest_gets_off_with_a_pulse_not_a_hold():
    task, ram = _harvest_with_horse(player_state=0x11)
    task._phase = "select"
    world = type("W", (), {"ram": ram})()
    a_index = _snes_a_index()
    pressed = [bool(task.step(world).action.action[a_index]) for _ in range(120)]
    assert any(pressed) and not all(pressed)
    run = longest = 0
    for flag in pressed:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    assert longest <= 3


def test_the_ship_window_counts_the_horse_stand_when_it_is_nearer_than_the_bin():
    task, ram = _harvest_with_horse()
    far_pick = (40, 33)
    near_horse = task._window_ship_stand(ram, (18, 22))
    assert near_horse != BIN  # the horse tile is beside a pick near it
    assert task._window_ship_stand(ram, (11, 33)) == BIN  # but the bin wins beside the bin
    task._saddle_off = True
    assert task._window_ship_stand(ram, (18, 22)) == (11, 29)  # the nearer of the bin's east stands
    assert far_pick  # (kept: documents the far ring used by the plan tests)


def test_the_park_tile_keeps_clear_of_crops_when_it_costs_little():
    # The ring's own neighbours are watering stands later in the day; a horse parked on one
    # stalls the watering walk. Nothing within the ring's neighbourhood: pick the clear tile.
    ring_neighbourhood = lambda tile: 39 <= tile[0] <= 43 and 31 <= tile[1] <= 35
    plan = horse.plan_park(
        EAST_RING, bin_stand=BIN, here=(8, 24), horse=HORSE,
        open_tile=open_everywhere, crop_adjacent=ring_neighbourhood,
    )
    assert plan is not None
    assert not ring_neighbourhood(plan.park) and not ring_neighbourhood((plan.park[0], plan.park[1] + 1))


def test_a_short_afternoon_does_not_pay_for_the_ride():
    # 16:00 with the bin closing at 17:00: after fetching and riding the horse there is time
    # for fewer than PARK_MIN_CROPS crops, so the far crops stay on the plant for tomorrow.
    late = horse.plan_park(EAST_RING, bin_stand=BIN, here=(8, 24), horse=HORSE, open_tile=open_everywhere, minutes_left=60)
    early = horse.plan_park(EAST_RING, bin_stand=BIN, here=(8, 24), horse=HORSE, open_tile=open_everywhere, minutes_left=240)
    assert late is None and early is not None


def test_the_harvest_parks_only_when_the_near_crops_are_done_and_the_day_is_young():
    task, ram = _harvest_with_horse()
    far, near = (41, 33), (12, 33)
    mk = lambda stand: type("S", (), {"stand": stand})()
    task._steps = [mk(near), mk(far)]
    assert not task._park_due(ram)  # a near crop is still on the list: the bin is cheaper
    task._steps = [mk(far), mk((42, 33))]
    assert task._park_due(ram)
    _set(ram, "hour", 15)
    assert not task._park_due(ram)  # past 14:30 there is no afternoon left to pay back the ride
    _set(ram, "hour", 10)
    task._park_tried = True
    assert not task._park_due(ram)  # one look per route


def test_a_horse_stand_that_cannot_be_reached_falls_back_to_the_bin():
    # ROM: the horse wandered to (40,40) and the walk to it burned 4,000 frames until the
    # bin-unreachable abort tossed the crop. A horse leg gets 600 steps (150 blocked), then the bin.
    task, ram = _harvest_with_horse()
    _set(ram, "player_state", 0x03)
    _set(ram, "held_item", 0x10)
    task._reset_ship_options(ram)
    stand = next(iter(task._horse_stands))
    task._ship_options = [(stand, task._horse_stands[stand])]
    task._phase = "ship_nav"
    task._ship_nav_steps = horse.HORSE_NAV_STEPS
    world = type("W", (), {"ram": ram})()
    task.step(world)
    assert task._saddle_off and not task._horse_stands
    task._reset_ship_options(ram)
    assert BIN in [s for s, _ in task._ship_options]
