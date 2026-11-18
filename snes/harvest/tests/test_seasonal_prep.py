"""Seasonal spending, pasture progress, and mushroom menu contracts."""

import numpy as np
import pytest
from retro_harness import TaskStatus, WorldState

from tests.layout_markers import south_pasture_only
from harvest.core.animal_status import CHICKEN_SLOT_BASE, CHICKEN_SLOT_SIZE
from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import ADDR_MAP, MATURE_GRASS_TILES
from harvest.maps.farm_pond import POCKET_PLANT_CENTERS, pocket_plant_targets
from harvest.planner import day_plan_status
from harvest.planner import livestock_econ as econ
from harvest.planner.day_phase_catalog import pasture_day_phases
from harvest.planner.livestock_econ import animal_shop_errand
from harvest.planner.day_phase_types import DayPlannerPolicy, day_planner_policy_for_season
from harvest.planner.day_plan_phases import build_outdoor_day_phases_from_ram
from harvest.tasks.grass_planter import (
    MIN_BAG_TILES, PASTURE_BOUNDS, GrassPlantTask, plan_stands, reachable_tiles, stand_squares,
)
from harvest.tasks.mountain_berry import MountainBerryTask
from harvest.tasks.nav import make_action


def ram(**fields):
    result = np.zeros(0x20000, dtype=np.uint8)
    for name, value in fields.items():
        spec = field_spec(name)
        value = spec.to_storage(value)
        for shift in range(spec.width):
            result[spec.address + shift] = (value >> (8 * shift)) & 255
    return result


def tile(result, x, y, value):
    result[ADDR_MAP + y * 64 + x] = value


def test_animal_schedule_buys_the_hen_first_then_cows():
    ready = dict(season=1, day=13, weekday=2, planted_grass=32, money=5300, stored_grass=40)
    assert animal_shop_errand(ram(**ready)) == "chicken"
    assert animal_shop_errand(ram(**(ready | dict(eve_hearts=250)))) == "chicken"
    assert animal_shop_errand(ram(**(ready | dict(season=2)))) == "chicken"
    for change in (
        dict(planted_grass=15),
        dict(money=1299),
        dict(season=3),
        dict(weekday=0),
        dict(season=0),
        dict(stored_grass=3),  # fewer than four days of feed for the hen
    ):
        assert animal_shop_errand(ram(**(ready | change))) is None
    assert animal_shop_errand(ram(**(ready | dict(stored_grass=4)))) == "chicken"
    # The hen needs only her own 16 tiles and 1000G plus the 300G floor.
    assert animal_shop_errand(ram(**(ready | dict(money=1300, planted_grass=16)))) == "chicken"
    assert animal_shop_errand(ram(**(ready | dict(money=1300, planted_grass=15)))) is None
    # With the hen owned the cow is next: 5000G + 300G floor, 16*(1+0+1) tiles.
    hen = ready | dict(num_chickens=1)
    assert animal_shop_errand(ram(**hen)) == "cow"
    assert animal_shop_errand(ram(**(hen | dict(money=5290)))) is None
    assert animal_shop_errand(ram(**(hen | dict(planted_grass=31)))) is None
    # Each hatched chick raises the shop's grass bar for the next cow.
    flock = hen | dict(num_chickens=4)
    assert animal_shop_errand(ram(**flock)) is None
    assert animal_shop_errand(ram(**(flock | dict(planted_grass=80)))) == "cow"
    assert animal_shop_errand(ram(**(flock | dict(planted_grass=80, stored_grass=19)))) is None
    # A second cow needs 16*(1+1+1) tiles. Two cows are the target.
    assert animal_shop_errand(ram(**(hen | dict(num_cows=1)))) is None
    assert animal_shop_errand(ram(**(hen | dict(num_cows=1, planted_grass=48)))) == "cow"
    assert animal_shop_errand(ram(**(hen | dict(num_cows=2, planted_grass=80)))) is None
    # The caller can ask for one species only.
    assert animal_shop_errand(ram(**ready), chickens=False) == "cow"
    assert animal_shop_errand(ram(**hen), cows=False) is None
    rainy = ram(**ready)
    rainy[0x0196] = 2
    assert animal_shop_errand(rainy) is None
    bought = ram(**ready)
    bought[0x11F5A] = 0x20
    assert animal_shop_errand(bought) is None


def test_pasture_reserve_holds_seed_bags_then_the_hen_then_a_funded_cow():
    base = dict(season=1, day=9, weekday=2, money=3000)
    # Blank farm RAM: every ring site is bare, so one corn bag (300G) per ring.
    rings = len(POCKET_PLANT_CENTERS)
    assert econ.pasture_reserve_g(ram(**base)) == 300 + 300 * rings + 1000
    assert econ.pasture_reserve_g(ram(**(base | dict(corn_seeds=rings)))) == 300 + 1000
    assert econ.pasture_reserve_g(ram(**(base | dict(corn_seeds=rings, num_chickens=1)))) == 300
    # No seed fund outside Summer, nor once a corn bag can no longer harvest in it.
    assert econ.pasture_reserve_g(ram(**(base | dict(season=2)))) == 300 + 1000
    last = econ.SUMMER_LAST_CORN_BAG_DAY
    assert econ.pasture_reserve_g(ram(**(base | dict(day=last)))) == 300 + 300 * rings + 1000
    assert econ.pasture_reserve_g(ram(**(base | dict(day=last + 1)))) == 300 + 1000
    late = dict(season=1, day=28, weekday=2, money=810, num_chickens=1, planted_grass=25)
    assert econ.grass_bags_to_buy(810, 0, 25, reserve=econ.pasture_reserve_g(ram(**late)), target=48) == 1
    # A wallet that already covers the cow keeps it covered.
    rich = base | dict(corn_seeds=rings, num_chickens=1, money=5600)
    assert econ.pasture_reserve_g(ram(**rich)) == 300 + 5000
    assert econ.pasture_reserve_g(ram(**(rich | dict(num_cows=2)))) == 300


def test_pasture_buys_needed_seeds_with_low_reserve():
    # With 2830G and 1 seed, funds afford the 3 needed bags (leaving > 300G)
    # (seed bags for every ring already in stock, so only the hen fund is held)
    rings = len(POCKET_PLANT_CENTERS)
    data = ram(season=1, day=2, weekday=2, grass_seeds=1, planted_grass=0, money=2830, corn_seeds=rings)
    phases = pasture_day_phases(data)
    assert [phase.phase for phase in phases] == ["BUY_GRASS_SEEDS", "PLANT_PASTURE"]
    # Poor morning (< 800G) only plants held stock without buying
    poor = ram(season=1, day=2, weekday=2, grass_seeds=1, planted_grass=0, money=680, corn_seeds=rings)
    assert [phase.phase for phase in pasture_day_phases(poor)] == ["PLANT_PASTURE"]
    # Rings still waiting on corn bags own the early-Summer shop trip: no grass errand.
    waiting = ram(season=1, day=2, weekday=2, grass_seeds=1, planted_grass=0, money=2830)
    assert pasture_day_phases(waiting) == []
    # A Fall bag never raises planted_grass (the ROM counts it before Fall only).
    fall = ram(season=2, day=18, weekday=2, grass_seeds=1, planted_grass=0, money=680)
    assert pasture_day_phases(fall) == []
    # Summer D1 is a shop day for grass.
    first = ram(season=1, day=1, weekday=2, grass_seeds=1, planted_grass=0, money=2830, corn_seeds=rings)
    assert [phase.phase for phase in pasture_day_phases(first)] == ["BUY_GRASS_SEEDS", "PLANT_PASTURE"]


def test_pasture_cuts_hay_for_a_short_herd_and_grows_with_the_flock():
    x0, y0, _x1, _y1 = PASTURE_BOUNDS
    base = dict(season=1, day=12, weekday=2, planted_grass=48, stored_grass=0,
                num_chickens=1, money=2000)

    def with_mature(count, **fields):
        data = ram(**(base | fields))
        for i in range(count):
            tile(data, x0 + i % 6, y0 + i // 6, 0x79)
        return [phase.phase for phase in pasture_day_phases(data)]

    # One ripe tile is enough when the herd has under four days of feed.
    assert with_mature(1) == ["CUT_FODDER"]
    # A stocked barn waits for a worthwhile cut of eight tiles.
    assert with_mature(3, stored_grass=20) == []
    assert with_mature(8, stored_grass=20) == ["CUT_FODDER"]
    # Sixteen tiles per animal (the shop gate) plus two spare, up to the cap.
    assert econ.pasture_target_tiles(ram()) == 32
    assert econ.pasture_target_tiles(ram(num_chickens=1)) == 48
    assert econ.pasture_target_tiles(ram(num_chickens=6, num_cows=1)) == 144
    assert econ.pasture_target_tiles(ram(num_chickens=12, num_cows=2)) == econ.PASTURE_MAX_TILES
    assert econ.pasture_feeds_more(ram(planted_grass=18, num_chickens=1), reserve_first_cow=False)
    assert not econ.pasture_feeds_more(ram(planted_grass=17, num_chickens=1), reserve_first_cow=False)
    assert with_mature(0, planted_grass=33, num_chickens=6, num_cows=1, money=3000,
                       corn_seeds=len(POCKET_PLANT_CENTERS))[:1] == ["BUY_GRASS_SEEDS"]
    # Fall stocks the rest of Fall, Winter and Spring's regrowth for each animal: D10 with 2.
    fall = dict(season=2, day=10, stored_grass=100, num_chickens=2, planted_grass=48)
    assert econ.hay_cut_target(ram(**fall)) == 2 * (20 + 31 + econ.SPRING_HAY_DAYS)
    assert with_mature(8, **fall) == ["CUT_FODDER"]
    assert with_mature(3, **fall) == []


def test_late_summer_planting_beats_a_hay_cut_unless_the_herd_is_short():
    x0, y0, _x1, _y1 = PASTURE_BOUNDS
    # Summer D28: 25 tiles planted, 810G, one hen, 29 hay, ten ripe tiles.
    base = dict(season=1, day=28, weekday=2, planted_grass=25, stored_grass=29,
                num_chickens=1, money=810)

    def names(**fields):
        data = ram(**(base | fields))
        for i in range(10):
            tile(data, x0 + i % 6, y0 + i // 6, 0x79)
        return [phase.phase for phase in pasture_day_phases(data)]

    assert names() == ["BUY_GRASS_SEEDS", "PLANT_PASTURE"]
    assert names(stored_grass=2) == ["CUT_FODDER"]  # a hen with two days of hay cuts first
    assert names(money=500) == ["CUT_FODDER"]  # nothing to plant or buy: cut the ripe tiles
    assert names(season=2, day=10, planted_grass=25, stored_grass=29, money=3000) == ["CUT_FODDER"]


def test_hatching_preserves_grass_required_for_first_cow():
    # v2 reached Fall with 35 grass and four chickens; each chick had raised
    # the vendor's purchase threshold, leaving the first cow unreachable.
    assert not econ.pasture_feeds_more(ram(planted_grass=36, num_chickens=1))
    assert not econ.pasture_feeds_more(ram(planted_grass=47, num_chickens=1))
    assert econ.pasture_feeds_more(ram(planted_grass=48, num_chickens=1))


def test_hatch_gate_counts_chicks_and_incubating_eggs_toward_two_hens():
    base = dict(season=1, day=10, num_cows=1, planted_grass=40)

    def flock(adults=0, chicks=0, eggs=0, **fields):
        data = ram(**(base | fields | dict(num_chickens=adults + chicks)))
        for slot, status in enumerate([0x09] * adults + [0x05] * chicks + [0x03] * eggs):
            data[CHICKEN_SLOT_BASE + slot * CHICKEN_SLOT_SIZE] = status
        return data

    assert econ.hatch_allowed(flock(adults=1))
    assert not econ.hatch_allowed(flock(adults=1, planted_grass=26))
    assert not econ.hatch_allowed(flock(adults=1, chicks=1))
    assert not econ.hatch_allowed(flock(adults=1, eggs=1))
    # Breeding surplus hens creates recurring care and sale trips.
    for year in (0, 1, 2):
        for season in range(4):
            assert not econ.hatch_allowed(flock(adults=2, year=year, season=season))
            assert econ.kept_adults(flock(adults=11, year=year, season=season)) == 2
    assert econ.FLOCK_CAP == econ.HENS_KEPT == 2
    # The second hen still reserves the first cow's shop pasture gate.
    assert not econ.hatch_allowed(flock(adults=1, num_cows=0, planted_grass=47))
    assert econ.hatch_allowed(flock(adults=1, num_cows=0, planted_grass=48))


# Summer D1 of a natural run (PR_Y1_Summer_D1), x=12..25, y=32..45: weeds 03,
# stones 04, lumber 05, stumps 09-0C and boulders 0D-10 across the south field.
NATURAL_SOUTH_FIELD = (
    "01 03 01 01 01 01 01 01 05 03 03 03 01 01",
    "01 01 01 01 01 01 01 01 03 01 01 01 01 01",
    "01 01 01 04 01 01 03 05 01 01 0D 0E 01 01",
    "01 01 03 03 01 03 01 01 01 01 0F 10 01 01",
    "04 01 01 01 03 05 01 01 03 01 01 03 03 01",
    "01 01 01 01 03 01 09 0A 01 01 01 04 01 01",
    "01 03 03 03 03 01 0B 0C 01 01 04 01 01 03",
    "03 03 01 03 01 03 01 01 01 01 01 01 03 03",
    "01 01 01 03 03 03 03 01 01 01 01 01 01 01",
    "09 0A 01 03 01 03 03 03 01 01 01 01 01 01",
    "0B 0C 01 03 03 01 01 01 0D 0E 01 01 09 0A",
    "01 01 03 03 01 01 01 01 0F 10 01 01 0B 0C",
    "01 04 01 03 03 02 01 01 01 01 01 0D 0E 01",
    "03 01 01 02 03 03 01 01 03 01 01 0F 10 01",
)


def natural_farm(**fields):
    data = ram(tilemap=0, player_x=16 * 16 + 8, player_y=33 * 16 + 8, **fields)
    for y, row in enumerate(NATURAL_SOUTH_FIELD, start=32):
        for x, value in enumerate(row.split(), start=12):
            tile(data, x, y, int(value, 16))
    return data


def test_bag_ready_skips_unseedable_squares_instead_of_rejecting_the_chunk():
    """The ROM seeds only hoed squares; a weed or stone in the 3x3 just shrinks the yield."""
    task = GrassPlantTask(bounds=PASTURE_BOUNDS, no_go_rects=[], bag_limit=2)
    data = ram()

    def chunk(*ids):
        for i, value in enumerate(ids):
            tile(data, 13 + i % 3, 33 + i // 3, value)
        return task._chunk_bag_ready(data, (13, 33))

    assert chunk(7, 7, 7, 7, 7, 7, 7, 3, 7)          # one weed: was rejected, 8 tiles
    assert chunk(7, 7, 7, 7, 8, 7, 4, 3, 5)          # stone, weed, lumber: 6 tiles
    assert not chunk(7, 7, 7, 7, 7, 4, 3, 5, 9)      # 5 hoed squares: not worth the bag
    assert not chunk(7, 7, 7, 7, 4, 7, 7, 3, 7)      # centre is a stone: stand on soil
    assert not chunk(3, 5, 7, 3, 1, 9, 3, 1, 11)     # the (16,36) chunk from the failing log


@south_pasture_only
def test_plan_stands_lifts_debris_but_skips_stumps_and_boulders():
    data = natural_farm()
    task = GrassPlantTask(bounds=PASTURE_BOUNDS, no_go_rects=[], bag_limit=4)
    task.reset(WorldState(frame=0, ram=ram(grass_seeds=4), info={}))
    task._navigator.update(data)
    reach = reachable_tiles(data, task._pathfinder, task._navigator.current_tile)
    stands = plan_stands(data, PASTURE_BOUNDS, 6, reach)
    assert len(stands) >= 5
    assert stands[0] == (16, 34)                       # the free bag goes at the anchor
    x0, y0, x1, y1 = PASTURE_BOUNDS
    for cx, cy in stands:
        assert x0 < cx < x1 and y0 < cy < y1
        soil, lifts = stand_squares(data, (cx, cy), reach)
        assert soil + lifts >= MIN_BAG_TILES
    for i, (ax, ay) in enumerate(stands):
        for bx, by in stands[:i]:
            assert max(abs(ax - bx), abs(ay - by)) >= 3   # no square is sown twice
    # Four bags of this jungle still cover the 32-tile shop gate once lifted.
    assert sum(sum(stand_squares(data, c, reach)) for c in stands[:4]) >= 32


def test_reach_walks_through_weed_walls_but_not_past_the_step_limit():
    data = ram(tilemap=0, player_x=16 * 16 + 8, player_y=33 * 16 + 8)
    for x in range(10, 30):
        for y in range(30, 46):
            tile(data, x, y, 0x01)
    for y in range(30, 46):
        tile(data, 20, y, 0x03)                       # a weed wall east of the start
    task = GrassPlantTask(bounds=PASTURE_BOUNDS, no_go_rects=[], bag_limit=1)
    task._navigator.update(data)
    reach = reachable_tiles(data, task._pathfinder, task._navigator.current_tile)
    assert (21, 34) in reach and (20, 34) in reach
    assert (10, 45) not in reach or abs(10 - 16) + abs(45 - 33) <= 30


@south_pasture_only
def test_scan_hoes_the_stand_centre_before_the_ring_and_never_under_his_feet():
    data = ram(tilemap=0, player_x=17 * 16 + 8, player_y=34 * 16 + 8, grass_seeds=1, input_lock=1)
    for y in range(32, 37):
        for x in range(14, 20):
            tile(data, x, y, 0x01)
    task = GrassPlantTask(bounds=PASTURE_BOUNDS, no_go_rects=[], bag_limit=1)
    world = WorldState(frame=0, ram=data, info={})
    task.reset(world)
    task._phase = "till"
    task._chunk_order = [(15, 33)]
    task._chunk_index = 0
    task._navigator.update(data)
    assert task._navigator.current_tile == (17, 34)
    task._handle_scan(data)
    assert task._target_tile == (16, 34)              # centre, though he is nearer the east side
    tile(data, 16, 34, 0x07)
    task._handle_scan(data)
    assert task._target_tile != (17, 34)              # (17,34) is under his feet


def _plant_task(data, *, centre_tile, ring_tile=0x07):
    """A grass task one square south of a ready 3x3 stand, in the plant phase."""
    for y in range(33, 36):
        for x in range(13, 16):
            tile(data, x, y, ring_tile)
    tile(data, 14, 34, centre_tile)
    task = GrassPlantTask(bounds=PASTURE_BOUNDS, no_go_rects=[], bag_limit=1)
    task.reset(WorldState(frame=0, ram=data, info={}))
    task._phase = "plant"
    task._chunk_order, task._chunk_index = [(13, 33), (40, 40)], 0
    task._navigator.update(data)
    return task


@south_pasture_only
def test_rained_on_centre_is_a_plantable_stand_not_a_scan_loop():
    """Rain turns a hoed 0x07 into 0x08. e2e t2/t3 spun 6M frames on Summer D20-21 because only
    0x07 counted as a plantable target, so the walk to a rained-on centre was always 'finished'."""
    data = ram(tilemap=0, player_x=14 * 16 + 8, player_y=36 * 16 + 8, grass_seeds=1, input_lock=1)
    task = _plant_task(data, centre_tile=0x08)
    assert 0x08 in task._target_ids()
    world = WorldState(frame=0, ram=data, info={})
    task.step(world)                                   # scan: walk to the centre
    assert task._state == "navigate" and task._target_tile == (14, 34)
    task.step(world)                                   # navigate keeps the target (still plantable)
    assert task._state == "navigate" and task._bounces == 0


@south_pasture_only
def test_scan_bounces_without_a_tool_use_fail_the_chunk_instead_of_spinning():
    """Whatever makes a target invalid, a dozen scan round trips with no swing give the chunk up."""
    data = ram(tilemap=0, player_x=14 * 16 + 8, player_y=36 * 16 + 8, grass_seeds=1, input_lock=1)
    task = _plant_task(data, centre_tile=0x08)
    task._chunk_bag_ready = lambda _ram, _origin: True       # force scan to keep choosing this stand
    tile(data, 14, 34, 0x01)                                  # ...though the centre is plain grass ground
    world = WorldState(frame=0, ram=data, info={})
    for _ in range(80):
        task.step(world)
        if task._chunk_index:
            break
    assert task._chunk_index == 1                             # gave up on (13,33) and moved on
    assert task._failures == 1 and (14, 34) in task._failed_tiles


def test_plant_verify_waits_for_the_ninth_square_then_moves_on():
    data = ram(tilemap=0, player_x=16 * 16 + 8, player_y=34 * 16 + 8, grass_seeds=3,
               planted_grass=10, input_lock=1)
    task = GrassPlantTask(bounds=PASTURE_BOUNDS, no_go_rects=[], bag_limit=2)
    task.reset(WorldState(frame=0, ram=data, info={}))
    task._phase, task._target_tile = "plant", (16, 34)
    task._chunk_order, task._chunk_index = [(15, 33), (18, 33)], 0
    task._bag_seeds, task._bag_grass, task._verify_wait = 3, 10, 0
    data[field_spec("planted_grass").address] = 14     # four squares seeded so far
    assert task._handle_verify(data) is None and task._chunk_index == 0
    assert task._verify_wait == 1                      # counter not spent yet: keep waiting
    data[field_spec("grass_seeds").address] = 2        # the ninth square spends the bag
    data[field_spec("planted_grass").address] = 19
    assert task._handle_verify(data) is None
    assert task._chunk_index == 1 and task.planted_count == 9
    # A bag that vanishes without growth (Fall seeds do not count) is a failure.
    task._phase, task._bag_seeds, task._bag_grass = "plant", 2, 19
    data[field_spec("grass_seeds").address] = 1
    assert task._handle_verify(data).status == TaskStatus.FAILURE


def test_toss_faces_prefers_open_ground():
    data = ram(tilemap=0, player_x=16 * 16 + 8, player_y=34 * 16 + 8)
    for y in range(30, 40):
        for x in range(12, 22):
            tile(data, x, y, 0x01)
    for dx, dy in ((0, 1), (-1, 0)):
        tile(data, 16 + dx, 34 + dy, 0x03)             # weeds below and left
    task = GrassPlantTask(bounds=PASTURE_BOUNDS, no_go_rects=[])
    task._navigator.update(data)
    faces = task._toss_faces(data)
    assert faces[-2:] == ["down", "left"] or set(faces[-2:]) == {"down", "left"}
    assert faces[0] in ("right", "up")


def test_pasture_planter_keeps_sowing_past_the_cow_gate():
    """s1_ring_fix: PLANT_PASTURE "succeeded" on 16 mornings at 35 tiles with 2-5 bags in stock.

    The planter's own target was its default of 32, so it returned at once while
    ``pasture_target_tiles`` wanted 48-80. The phase must carry the economy target.
    """
    from harvest.core.tile_catalog import ADDR_INPUT_LOCK
    from harvest.planner.day_phase_registry import TaskBuildContext, _build_grass

    data = ram(season=1, day=17, weekday=2, grass_seeds=2, planted_grass=35, num_chickens=1,
               money=2000, corn_seeds=len(POCKET_PLANT_CENTERS))
    plant = [phase for phase in pasture_day_phases(data) if phase.phase == "PLANT_PASTURE"]
    assert plant and plant[0].params["target_tiles"] == econ.pasture_target_tiles(data) == 48

    x0, y0, x1, y1 = PASTURE_BOUNDS
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            tile(data, x, y, 0x07)
    data[ADDR_INPUT_LOCK] = 1
    world = WorldState(frame=0, ram=data, info={})
    sowing = _build_grass(TaskBuildContext(), plant[0], world).tasks[-1]
    assert sowing.target_tiles == 48
    sowing.reset(world)
    assert sowing.step(world).status == TaskStatus.RUNNING  # a bag is sown, not "complete"
    # The old default is the stall: 35 >= 32 ends the task before any bag is used.
    stalled = GrassPlantTask(bounds=PASTURE_BOUNDS, no_go_rects=[], target_tiles=32, bag_limit=2)
    stalled.reset(world)
    assert stalled.step(world).status == TaskStatus.SUCCESS


def test_pasture_plan_is_one_nav_and_no_farm_clearer():
    """The old FarmClearer pass wandered to the town and left stones behind."""
    from harvest.planner.day_phase_registry import TaskBuildContext, _build_grass
    from harvest.planner.day_phase_types import PhaseSpec
    from harvest.tasks.farm_clearer import FarmClearer
    from harvest.tasks.skills import BestEffortSkill

    world = WorldState(frame=0, ram=ram(season=1, day=3, grass_seeds=2), info={})
    task = _build_grass(TaskBuildContext(), PhaseSpec("PLANT_PASTURE", "grass"), world)
    assert not [t for t in task.tasks if isinstance(t, (FarmClearer, BestEffortSkill))]
    assert isinstance(task.tasks[-1], GrassPlantTask)


@pytest.mark.parametrize("planted_grass", [0, 64])
def test_till_field_limits_verified_hoeing_independently_of_pasture(planted_grass):
    from harvest.planner.day_phase_catalog import TILL_FIELD_PHASE
    from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
    from harvest.tasks.grass_planter import Phase

    data = ram(planted_grass=planted_grass, input_lock=1, player_x=264, player_y=552)
    x0, y0, x1, y1 = PASTURE_BOUNDS
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            tile(data, x, y, 0x01)
    world = WorldState(frame=0, ram=data, info={})
    task = build_phase_task(TaskBuildContext(), TILL_FIELD_PHASE, world).tasks[-1]
    task.reset(world)
    assert len(task._chunk_order) * 9 >= 32  # no grass seed bags are required
    assert task.step(world).status == TaskStatus.RUNNING
    task.reset(world)
    for index in range(32):
        target = (x0 + index % 10, y0 + index // 10)
        tile(data, *target, 0x07)
        task._phase = Phase.TILL
        task._target_tile = target
        task._handle_verify(data)
    result = task.step(world)
    assert result.status == TaskStatus.SUCCESS
    assert "tilled=32" in result.reason


def test_till_field_reports_short_hoeing_even_with_an_existing_pasture():
    from harvest.planner.day_phase_catalog import TILL_FIELD_PHASE
    from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task

    world = WorldState(frame=0, ram=ram(planted_grass=64, input_lock=1), info={})
    task = build_phase_task(TaskBuildContext(), TILL_FIELD_PHASE, world).tasks[-1]
    task.reset(world)
    result = task.step(world)
    assert result.status == TaskStatus.FAILURE
    assert "tilling short" in result.reason


@pytest.mark.rom
def test_till_field_rom_stops_after_32_hoed_tiles_with_an_existing_pasture():
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.day_phase_catalog import TILL_FIELD_PHASE
    from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
    from harvest.planner.tasks.inventory import ExitToFarmTask
    from harvest.runtime.retro_setup import make_harvest_env
    from retro_harness import AuditedEnv, AuditCapabilities

    env = AuditedEnv(make_harvest_env("LN_cowfix_pin_fall_d21", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.till_field_test"))
    try:
        env.reset()
        before = {key: int(read_ram_value(env.get_ram(), key))
                  for key in ("planted_grass", "grass_seeds")}
        assert before["planted_grass"] >= 32
        world = WorldState(frame=0, ram=env.get_ram(), info={})
        work = build_phase_task(TaskBuildContext(), TILL_FIELD_PHASE, world)
        for task in (ExitToFarmTask(), work):
            task.reset(WorldState(frame=0, ram=env.get_ram(), info={}))
            for frame in range(16000):
                result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}))
                if result.status != TaskStatus.RUNNING:
                    break
                env.step(result.action.action if result.action else make_action())
            assert result.status == TaskStatus.SUCCESS, result.reason
        assert work.tasks[-1].tilled_count == 32
        assert {key: int(read_ram_value(env.get_ram(), key)) for key in before} == before
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


def test_exit_coop_waits_out_the_farm_fade_in():
    """The pad is dead until $0025 reaches 15; the settle counts after that.

    A count from the tilemap flip (85 frames) expired ~70 frames before the
    farmer even landed: test_coop_exit_waits_for_the_farm_fade_in.
    """
    import numpy as np

    from harvest.planner.day_phase_chicken import COOP_EXIT_INPUT_LOCK_FRAMES, EXIT_COOP_PHASE
    from harvest.planner.tasks.transitions import screen_faded_in

    assert EXIT_COOP_PHASE.params["settle_frames"] == COOP_EXIT_INPUT_LOCK_FRAMES <= 10
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[0x0025] = 14
    assert not screen_faded_in(ram)
    ram[0x0025] = 15
    assert screen_faded_in(ram)


def test_grass_waypoint_lands_inside_pasture_bounds():
    from harvest.planner.day_phase_registry import TaskBuildContext, _build_grass
    from harvest.planner.day_phase_types import PhaseSpec
    from harvest.planner.tasks.navigation import MultiMapNavTask

    world = WorldState(frame=0, ram=ram(season=1, day=25, grass_seeds=4), info={})
    task = _build_grass(TaskBuildContext(), PhaseSpec("PLANT_PASTURE", "grass"), world)
    navs = [child for child in task.tasks if isinstance(child, MultiMapNavTask)]
    assert navs
    px, py = navs[-1].waypoints[-1].target_px
    x0, y0, x1, y1 = PASTURE_BOUNDS
    assert x0 <= px // 16 <= x1 and y0 <= py // 16 <= y1
    assert (px, py) != (312, 536)


@south_pasture_only
def test_pasture_approach_pin_is_not_arrival_on_the_way_to_264_520():
    """(249, 526) is the weed-row stall. Radius 8 must not call it (264, 520)."""
    from harvest.maps.map_config import Waypoint
    from harvest.planner.day_phase_registry import TaskBuildContext, _build_grass
    from harvest.planner.day_phase_types import PhaseSpec
    from harvest.planner.tasks.navigation import WATER_STAND_ARRIVE_PX, MultiMapNavTask

    assert WATER_STAND_ARRIVE_PX == 23
    world = WorldState(frame=0, ram=ram(season=1, day=25, grass_seeds=4), info={})
    task = _build_grass(TaskBuildContext(), PhaseSpec("PLANT_PASTURE", "grass"), world)
    navs = [child for child in task.tasks if isinstance(child, MultiMapNavTask)]
    points = [wp.target_px for wp in navs[-1].waypoints]
    assert (312, 520) in points
    assert navs[-1].allow_tiles == frozenset({(11, 26), (11, 27), (11, 28)})
    assert points[points.index((312, 520)) + 1] == (296, 536)
    assert (264, 520) not in points
    probe = MultiMapNavTask(waypoints=[Waypoint(0x00, (264, 520), radius=8)])
    probe._navigator.current_pos.x = 249
    probe._navigator.current_pos.y = 526
    assert probe._at_wp_target(probe.waypoints[0]) is False
    hop = navs[-1].waypoints[points.index((312, 520)) + 1]
    navs[-1]._navigator.current_pos.x = 249
    navs[-1]._navigator.current_pos.y = 526
    assert navs[-1]._at_wp_target(hop) is False


def test_held_seeds_plant_on_summer_d25_without_a_shop_trip():
    data = ram(season=1, day=25, weekday=6, grass_seeds=4, planted_grass=0, money=3010, num_cows=0)
    assert [phase.phase for phase in pasture_day_phases(data)] == ["PLANT_PASTURE"]


def test_cut_fodder_targets_the_roms_sickle_tile():
    assert MATURE_GRASS_TILES == {0x79}
    data = ram(season=1, day=14, stored_grass=0, planted_grass=32)
    x0, y0, _x1, _y1 = PASTURE_BOUNDS
    tile(data, x0, y0, 0x78)
    assert pasture_day_phases(data) == []
    tile(data, x0, y0, 0x79)
    assert [p.phase for p in pasture_day_phases(data)] == ["CUT_FODDER"]


def test_summer_keeps_regrowing_plots_without_buying_a_bag_for_one_hole():
    data = ram(season=1)
    for cx, cy in POCKET_PLANT_CENTERS:
        for y in range(cy - 1, cy + 2):
            for x in range(cx - 1, cx + 2):
                tile(data, x, y, 0x3A if (x, y) != (cx, cy) else 0x01)
    cx, cy = POCKET_PLANT_CENTERS[0]
    tile(data, cx - 1, cy - 1, 0x01)
    assert pocket_plant_targets(data) == ()
    data[field_spec("season").address] = 0
    # One corner hole walls the sowing stand in, and the bag would spend
    # seven uses on growing cells: the ring waits in Spring too.
    assert pocket_plant_targets(data) == ()
    tile(data, cx, cy - 1, 0x01)
    tile(data, cx + 1, cy - 1, 0x01)
    assert pocket_plant_targets(data) == (POCKET_PLANT_CENTERS[0],)


def test_mushroom_uses_live_choices_and_actual_row_not_text_offset():
    data = ram(tilemap=0x10, input_lock=2, dialog_text_id=0x0366,
               dialog_menu_cursor=1, dialog_choice_result=0)
    task = MountainBerryTask()
    task._eat_phase = "wait"
    task._eat_wait = 95  # next frame is a pulse
    world = WorldState(frame=0, ram=data, info={})
    assert np.array_equal(task._handle_eat_box(world, lock=2).action.action, make_action(a=True))
    data[0x019B] = 0x10
    task._eat_wait = 95
    assert np.array_equal(task._handle_eat_box(world, lock=2).action.action, make_action(down=True))
    data[field_spec("dialog_choice_result").address] = 1
    data[field_spec("dialog_menu_cursor").address] = 0
    task._eat_wait = 95
    assert np.array_equal(task._handle_eat_box(world, lock=2).action.action, make_action(a=True))


def test_fall_schedules_shipping_before_home_and_summer_uses_pocket_clearing():
    summer = day_planner_policy_for_season(1)
    assert not summer.include_field_clear and summer.include_planting
    data = ram(season=2, day=1, weekday=2, hour=6, tilemap=0, input_lock=1)
    policy = day_planner_policy_for_season(2, DayPlannerPolicy())
    names = [p.phase for p in build_outdoor_day_phases_from_ram(data, policy=policy)]
    assert "MUSHROOM_GATHER" in names
    assert "CROP_WATER" not in names


def _day_phase_names(**fields):
    from harvest.planner.day_plan_phases import build_day_phases_from_ram

    base = dict(season=1, day=14, weekday=2, hour=6, tilemap=0, money=2000,
                num_cows=1, num_chickens=1, stamina=100, planted_grass=32, input_lock=1)
    return [p.phase for p in build_day_phases_from_ram(ram(**(base | fields)))]


def test_cow_tools_trip_waits_for_a_cow_the_gold_and_an_open_store():
    assert "BUY_COW_TOOLS" in _day_phase_names()
    assert "BUY_COW_TOOLS" not in _day_phase_names(num_cows=0)
    assert "BUY_COW_TOOLS" not in _day_phase_names(money=(econ.SHOP_PRICE_G['brush'] + econ.WALLET_RESERVE_G) - 10)
    assert "BUY_COW_TOOLS" in _day_phase_names(money=(econ.SHOP_PRICE_G['brush'] + econ.WALLET_RESERVE_G))
    assert "BUY_COW_TOOLS" not in _day_phase_names(weekday=0)
    assert "BUY_COW_TOOLS" not in _day_phase_names(hour=14)


def test_cow_tools_trip_waits_for_the_milker_price_once_the_brush_is_owned():
    from harvest.tasks.buy_seeds import TOOL_SHED_ROW_ADDR

    brush_owned = dict(money=(econ.SHOP_PRICE_G['brush'] + econ.WALLET_RESERVE_G) + 200)

    def names(**fields):
        from harvest.planner.day_plan_phases import build_day_phases_from_ram

        base = dict(season=1, day=14, weekday=2, hour=6, tilemap=0, num_cows=1, num_chickens=1,
                    stamina=100, planted_grass=32, input_lock=1)
        data = ram(**(base | fields))
        data[TOOL_SHED_ROW_ADDR] = 0x40  # brush on the shed shelf
        return [p.phase for p in build_day_phases_from_ram(data)]

    assert "BUY_COW_TOOLS" not in names(**brush_owned)
    assert "BUY_COW_TOOLS" not in names(money=econ.SHOP_PRICE_G['milker'] + econ.WALLET_RESERVE_G - 10)
    assert "BUY_COW_TOOLS" in names(money=econ.SHOP_PRICE_G['milker'] + econ.WALLET_RESERVE_G)
    # Fall is the last window before Winter shuts the town trips: no reserve then.
    assert "BUY_COW_TOOLS" not in names(season=2, money=econ.SHOP_PRICE_G['milker'] - 10)
    assert "BUY_COW_TOOLS" in names(season=2, money=econ.SHOP_PRICE_G['milker'])


def test_cow_tools_trip_does_not_share_a_morning_with_an_animal_purchase():
    # Hen and one cow owned, cow #2 funded: the shop errand is the cow, the store waits.
    names = _day_phase_names(money=5600, planted_grass=48, stored_grass=40)
    assert "BUY_COW_VENDOR" in names
    assert "BUY_COW_TOOLS" not in names


def test_coop_ships_eggs_while_the_pasture_cannot_feed_another_animal():
    from harvest.planner.day_plan_phases import build_day_phases_from_ram

    def coop_mode(planted, *, num_chickens=1):
        data = ram(season=1, day=14, weekday=2, hour=6, tilemap=0, money=500, num_chickens=num_chickens,
                   planted_grass=planted, stored_grass=20, stamina=100, input_lock=1)
        # fed_chickens_n 0 with hens owned makes the chores due.
        for phase in build_day_phases_from_ram(data):
            if phase.phase == "COOP_CHORES":
                return phase.params.get("egg_mode")
        return "missing"

    # Below the two-bird cap, reserve the vendor's grass for a hatch and the first cow.
    assert coop_mode(48) == "auto"
    assert coop_mode(47) == "ship"
    assert coop_mode(20) == "ship"
    # At or above the two-bird cap, always ship.
    assert coop_mode(80, num_chickens=2) == "ship"
    assert coop_mode(80, num_chickens=3) == "ship"


def test_grass_stands_preserve_crop_footprints_and_working_lanes():
    from harvest.maps.farm_pond import crop_working_tiles

    data = ram()
    data[ADDR_MAP:ADDR_MAP + 64 * 64] = 1
    reserved = crop_working_tiles()
    stands = plan_stands(data, (2, 25, 28, 48), 20, reach=None)
    assert stands
    assert all((cx + dx, cy + dy) not in reserved
               for cx, cy in stands for dx in (-1, 0, 1) for dy in (-1, 0, 1))
