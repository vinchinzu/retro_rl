"""Seasonal spending, pasture progress, and mushroom menu contracts."""

import numpy as np
from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import ADDR_MAP, MATURE_GRASS_TILES
from harvest.maps.farm_pond import POCKET_PLANT_CENTERS, pocket_plant_targets
from harvest.planner.day_phase_catalog import pasture_day_phases, summer_animal_species
from harvest.planner.day_phase_types import DayPlannerPolicy, day_planner_policy_for_season
from harvest.planner.day_plan_phases import build_outdoor_day_phases_from_ram
from harvest.tasks.grass_planter import GrassPlantTask, PASTURE_BOUNDS
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


def test_animal_schedule_needs_summer_date_feed_reserve_and_one_of_each():
    ready = dict(season=1, day=14, weekday=2, planted_grass=32,
                 stored_grass=16, money=5300)
    assert summer_animal_species(ram(**ready)) == "cow"
    for change in (dict(day=13), dict(season=2), dict(planted_grass=31),
                   dict(stored_grass=15), dict(money=5290)):
        assert summer_animal_species(ram(**(ready | change))) is None
    assert summer_animal_species(ram(**(ready | dict(num_cows=1, money=1300)))) == "chicken"
    assert summer_animal_species(ram(**(ready | dict(num_cows=1, num_chickens=1)))) is None
    rainy = ram(**ready)
    rainy[0x0196] = 2
    assert summer_animal_species(rainy) is None
    bought = ram(**ready)
    bought[0x11F5A] = 0x20
    assert summer_animal_species(bought) is None


def test_pasture_spends_bags_on_empty_chunks_before_repairing_one_hole():
    data = ram(season=1, day=4, weekday=6, grass_seeds=2, planted_grass=17,
               input_lock=1)
    x0, y0, x1, y1 = PASTURE_BOUNDS
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            tile(data, x, y, 0x01 if y >= y0 + 3 else 0x70)
    tile(data, x0, y0, 0x01)
    task = GrassPlantTask(bounds=PASTURE_BOUNDS, no_go_rects=[], bag_limit=2)
    task.reset(WorldState(frame=0, ram=data, info={}))
    assert task._chunk_order[0][1] == y0 + 3
    assert [p.phase for p in pasture_day_phases(data)] == ["PLANT_PASTURE"]
    data[field_spec("grass_seeds").address] = 0
    data[field_spec("planted_grass").address] = 34
    result = task.step(WorldState(frame=1, ram=data, info={}))
    assert result.status == TaskStatus.SUCCESS


def test_cut_fodder_targets_the_roms_sickle_tile():
    assert MATURE_GRASS_TILES == {0x79}
    data = ram(season=1, day=14, stored_grass=0, planted_grass=32)
    tile(data, 13, 33, 0x78)
    assert pasture_day_phases(data) == []
    tile(data, 13, 33, 0x79)
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
