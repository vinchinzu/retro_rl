"""Year 3 Summer buys both crops on one shop trip, so every free site is sown the day it can be.

r41..r47 (2026-11-17) bought 4-6 corn bags on Summer D1 (a Friday: the shop shuts Sat and Sun) and the tomato bags on
D5-D6, so half the twelve sites stood bare for a week and Y3 shipped 232-366 crops against the 380 it needs.
Each test below goes red on the code before ``joint_summer_bags`` and the shop task's ``then`` list.
"""
from __future__ import annotations

import pytest

from harvest.planner import run_strategy as rs
from harvest.tasks.buy_seeds import SEED_BAG_PRICE, SEED_STAND_X, BuyShelfItemTask
from retro_harness import TaskStatus
from tests.day_plan_test_helpers import make_date_world, set_money, set_player_pos
from tests.test_run_strategy import set_field, set_tile, world

# Y1 + Y2 Summers delivered corn 247 and tomato 365 in every r4x run, so Y3 owes corn 249 and tomato 131.
SHIPPED = {"corn": 247, "tomato": 365}


@pytest.fixture(autouse=True)
def score_herd_with_joint_buy(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score_herd")
    monkeypatch.setenv(rs.Y3_JOINT_BUY_ENV, "1")


def _summer(day=1, shipped=None, year=2):
    w = world(season=1, day=day, shipped=SHIPPED if shipped is None else shipped)
    set_field(w.ram, "year", year)
    return w.ram


def test_the_switch_is_on_by_default_and_zero_turns_it_off(monkeypatch):
    monkeypatch.delenv(rs.Y3_JOINT_BUY_ENV)
    assert rs.joint_summer_bags(_summer(), 12, {})
    monkeypatch.setenv(rs.Y3_JOINT_BUY_ENV, "0")
    assert rs.joint_summer_bags(_summer(), 12, {}) == {}


def test_twelve_free_sites_split_two_to_one_by_what_each_crop_still_owes():
    bags = rs.joint_summer_bags(_summer(), 12, {})
    assert sum(bags.values()) == 12 and bags["corn"] > bags["tomato"] >= 3, bags
    assert bags == {"corn": 8, "tomato": 4}


def test_nine_free_sites_are_all_sown_and_each_crop_gets_one():
    """r41/r43 had 9 free sites on D1 (three still held an unripe Spring crop): all nine take a bag today."""
    bags = rs.joint_summer_bags(_summer(), 9, {})
    assert sum(bags.values()) == 9 and min(bags.values()) >= 2
    assert bags["corn"] > bags["tomato"]


def test_held_bags_come_off_the_trip():
    bags = rs.joint_summer_bags(_summer(), 12, {"corn_seeds": 3, "tomato_seeds": 1})
    assert bags == {"corn": 5, "tomato": 3}


def test_a_standing_ring_counts_toward_its_crops_share_of_the_twelve():
    ram = _summer(day=4)
    for x in range(10, 14):                                    # four corn rings (32 tiles) already stand
        for y in range(40, 48):
            set_tile(ram, x, y, 0x3A)
    bags = rs.joint_summer_bags(ram, 8, {})
    assert sum(bags.values()) == 8 and bags["corn"] <= 4 and bags["tomato"] >= 4, bags


def test_one_crop_left_is_the_single_crop_paths_business():
    assert rs.joint_summer_bags(_summer(shipped={"corn": 247, "tomato": 496}), 12, {}) == {}   # tomatoes are done
    assert rs.joint_summer_bags(_summer(shipped={"corn": 496, "tomato": 365}), 12, {}) == {}   # corn is done


def test_a_crop_that_cannot_ripen_before_the_credits_gets_no_bag():
    assert rs.summer_cycles("corn", 1) == 5 and rs.summer_cycles("tomato", 1) == 6
    assert rs.summer_cycles("corn", 18) == 0 and rs.summer_cycles("tomato", 18) >= 1
    assert rs.joint_summer_bags(_summer(day=18), 12, {}) == {}      # only tomato can still ripen: single-crop path


def test_only_year_three_summer_changes():
    assert rs.joint_summer_bags(_summer(year=1), 12, {}) == {}
    spring = world(season=0, day=3, shipped=SHIPPED).ram
    set_field(spring, "year", 2)
    assert rs.joint_summer_bags(spring, 12, {}) == {}


def test_other_strategies_never_split():
    import os

    os.environ[rs.ENV_VAR] = "cash"
    assert rs.joint_summer_bags(_summer(), 12, {}) == {}


# ── the shop task ─────────────────────────────────────────────────────────────────────────────────────────────────

def _shop_task(then):
    w = make_date_world(0x1C, season=1, day=1)
    set_money(w.ram, 30_000)
    set_player_pos(w.ram, 153, 342)
    task = BuyShelfItemTask(stock_field="corn_seeds", bags=8, then=then)
    task.reset(w)
    return w, task


def test_the_task_walks_on_to_the_next_bag_when_the_first_crop_is_bought():
    w, task = _shop_task((("tomato_seeds", 4),))
    assert task.stand_px[0] == SEED_STAND_X["corn_seeds"] == 152
    task._seen_shop, task._bought, task._buy_attempts = True, True, 40
    set_field(w.ram, "corn_seeds", 8)
    set_money(w.ram, 30_000 - 8 * SEED_BAG_PRICE["corn_seeds"])
    w.ram[0x22] = 0x1C
    task.step(w)
    assert task.stock_field == "tomato_seeds" and task.bags == 4
    assert not task._bought and task._buy_attempts == 0 and task.bag_price == SEED_BAG_PRICE["tomato_seeds"]
    assert task.stand_px[0] == SEED_STAND_X["tomato_seeds"] == 168
    walk = task._align_to_stand(w).action.action
    assert walk[7] and not walk[6]                          # RIGHT along the shelf row: 152 to 168


def test_the_receipt_names_every_crop_bought_and_the_last_crop_closes_the_task():
    w, task = _shop_task((("tomato_seeds", 4),))
    task._seen_shop, task._bought, task._buy_attempts = True, True, 40
    set_field(w.ram, "corn_seeds", 8)
    set_money(w.ram, 30_000 - 8 * 300)
    w.ram[0x22] = 0x1C
    task.step(w)
    set_field(w.ram, "tomato_seeds", 4)
    set_money(w.ram, 30_000 - 12 * 300)
    task._bought = True
    reason = task._success(w.ram).reason
    assert reason.startswith("bought corn_seeds 0->8 ") and " then tomato_seeds 0->4 " in reason, reason


def test_a_follow_on_crop_that_will_not_close_leaves_with_the_first_instead_of_failing():
    w, task = _shop_task((("tomato_seeds", 4),))
    task._seen_shop, task._bought = True, True
    set_field(w.ram, "corn_seeds", 8)
    set_money(w.ram, 30_000 - 2400)
    w.ram[0x22] = 0x1C
    task.step(w)
    assert task.stock_field == "tomato_seeds"
    task._buy_attempts = 28 * 4
    result = task._queue_buy(w, lock=1)
    assert result.status == TaskStatus.RUNNING and task._bought and not task._then
    assert "corn_seeds" in task._seed_receipts[0]


def test_a_single_crop_trip_is_unchanged():
    w, task = _shop_task(())
    assert task._then == [] and task.stand_px[0] == 152
    task._seen_shop, task._bought, task._buy_attempts = True, True, 3
    w.ram[0x22] = 0x1C
    task.step(w)
    assert task.stock_field == "corn_seeds" and task._bought


def test_the_day_plan_builds_one_trip_for_both_crops(monkeypatch):
    """The BUY_SEEDS phase names corn; the task it builds also carries the tomato bags for the other free sites."""
    from harvest.maps import farm_pond
    from harvest.planner.day_phase_catalog import buy_seeds_phase
    from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task

    ram = _summer()
    monkeypatch.setattr(farm_pond, "pocket_plant_targets", lambda *a, **k: [(0, 0)] * 12)
    set_money(ram, 90_000)
    w = world(season=1, day=1, shipped=SHIPPED)
    w.ram[:] = ram
    task = build_phase_task(TaskBuildContext(), buy_seeds_phase(seed_type="corn"), w)
    assert (task.stock_field, task.bags, task.then) == ("corn_seeds", 8, (("tomato_seeds", 4),))
    monkeypatch.setenv(rs.Y3_JOINT_BUY_ENV, "0")
    single = build_phase_task(TaskBuildContext(), buy_seeds_phase(seed_type="corn"), w)
    assert single.then == ()
