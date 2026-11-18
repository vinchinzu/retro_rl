"""Run strategies (`harvest.planner.run_strategy`): cash stays the default and changes nothing,
score stops each crop at the ROM's 511 shipped window and balances the crop kinds."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from harvest.core.ram_catalog import field_spec, ram_index
from harvest.core.tile_catalog import ADDR_MAP, ADDR_TILEMAP as ADDR_TILEMAP_BYTE
from harvest.maps.farm_pond import POCKET_PLANT_CENTERS, pocket_plant_targets
from harvest.planner import livestock_econ as econ
from harvest.planner import run_strategy as rs
from harvest.planner.crop_planner import resolve_seed_type_for_date
from harvest.planner.world_probe import WorldProbe
from harvest.tasks.crop_geometry import plot_tiles
from harvest.tasks.harvest_task import HarvestTask
from tests.day_plan_test_helpers import make_date_world

SPRING, SUMMER, FALL = 0, 1, 2
CORN, TOMATO, POTATO, TURNIP = 0x52, 0x38, 0x60, 0x6E
SHIPPED_FIELD = {"corn": "shipped_corn", "tomato": "shipped_tomatoes",
                 "potato": "shipped_potatoes", "turnip": "shipped_turnips"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(rs.ENV_VAR, raising=False)


def use(monkeypatch, name: str) -> rs.RunStrategy:
    monkeypatch.setenv(rs.ENV_VAR, name)
    return rs.active()


def set_field(ram: np.ndarray, name: str, value: int) -> None:
    spec = field_spec(name)
    stored = spec.to_storage(value)
    idx = ram_index(ram, spec.address, live_offset=spec.live_offset)
    for shift in range(spec.width):
        ram[idx + shift] = (stored >> (8 * shift)) & 0xFF


def set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * 64 + tx] = tile_id


def world(*, season=SUMMER, day=10, shipped=None):
    w = make_date_world(0x00, season=season, day=day)
    for crop, count in (shipped or {}).items():
        set_field(w.ram, SHIPPED_FIELD[crop], count)
    return w


# ── selection ────────────────────────────────────────────────────────

def test_cash_is_the_default_and_overrides_nothing():
    cash = rs.active()
    assert cash.name == "cash" and cash is rs.CASH
    assert cash.ship_cap is None and not cash.balance_crops
    assert (cash.cow_target_later, cash.cow_target_fall, cash.hens_kept) == (None, None, None)


def test_env_selects_and_unknown_names_are_rejected(monkeypatch):
    assert use(monkeypatch, "SCORE").name == "score"
    assert rs.select("score_herd").name == "score_herd"
    with pytest.raises(ValueError, match="unknown strategy"):
        rs.get("cheat")


def test_every_preset_is_findable_by_its_name():
    assert all(rs.get(name).name == name for name in rs.PRESETS)


def test_only_perfect_dispatches_the_milestone_campaign():
    assert [name for name, s in rs.PRESETS.items() if s.perfect_rancher] == ["perfect"]
    assert rs.get("perfect").hens_kept == rs.get("perfect").cow_target_later == 12     # the ROM caps


# ── counters ─────────────────────────────────────────────────────────

def test_crop_points_follow_the_nine_bit_counter():
    assert [rs.crop_points(n) for n in (0, 15, 16, 495, 496, 511, 512, 811)] == [0, 0, 1, 30, 31, 31, 0, 18]


def test_points_lost_to_wrap_match_the_v3n_scorecard():
    # v3n shipped 811 corn (18 points) and 671 potatoes (9): 13 + 22 points left on the table.
    assert rs.points_lost_to_wrap({"corn": 811, "potato": 671, "tomato": 0, "turnip": 0}) == 35
    assert rs.points_lost_to_wrap({"corn": 511, "potato": 300}) == 0


# ── the cap ──────────────────────────────────────────────────────────

def ripe_field(w, crop_tile: int, count: int, *, x0=20, y=34):
    tiles = [(x0 + i, y) for i in range(count)]
    for tx, ty in tiles:
        set_tile(w.ram, tx, ty, crop_tile)
    return tiles


def test_cash_keeps_every_ripe_tile(monkeypatch):
    use(monkeypatch, "cash")
    w = world(shipped={"corn": 600})
    tiles = ripe_field(w, CORN, 5)
    assert rs.within_ship_cap(w.ram, tiles) == (tiles, [])


def test_the_cap_is_the_window_floor_with_fifteen_shipments_of_slack():
    assert rs.SCORE.ship_cap == rs.SHIP_WINDOW_LOW == 496
    assert rs.crop_points(rs.SCORE.ship_cap) == rs.SHIP_FULL_POINTS
    assert rs.SHIP_WINDOW_HIGH - rs.SCORE.ship_cap == 15       # a stray throw or two cannot wrap it


def test_score_keeps_only_the_room_each_crop_has_left_nearest_first(monkeypatch):
    use(monkeypatch, "score")
    w = world(shipped={"corn": 494, "tomato": 100})
    corn = ripe_field(w, CORN, 5, x0=12)
    tomato = ripe_field(w, TOMATO, 3, x0=30)
    kept, held = rs.within_ship_cap(w.ram, corn + tomato)
    assert sorted(kept) == sorted(corn[:2] + tomato)      # corn room 2, nearest to the bin (11,30) first
    assert sorted(held) == sorted(corn[2:])
    assert rs.crop_kinds(w.ram, held) == ["corn"]


def test_a_cap_never_reorders_the_tiles_it_keeps(monkeypatch):
    use(monkeypatch, "score")
    w = world(shipped={"corn": 100})
    tiles = [(40, 34), (12, 28), (30, 30), (11, 31)]        # deliberately not nearest-first
    for tile in tiles:
        set_tile(w.ram, *tile, CORN)
    assert rs.within_ship_cap(w.ram, tiles) == (tiles, [])
    set_field(w.ram, "shipped_corn", 494)                    # room 2: the two nearest stay, in caller order
    kept, held = rs.within_ship_cap(w.ram, tiles)
    assert kept == [(12, 28), (11, 31)] and held == [(40, 34), (30, 30)]


def test_score_holds_every_tile_of_a_capped_crop_and_never_a_non_crop(monkeypatch):
    use(monkeypatch, "score")
    w = world(shipped={"potato": 496})
    potatoes = ripe_field(w, POTATO, 4)
    set_tile(w.ram, 40, 34, 0x03)                          # a weed: not a crop, never held
    kept, held = rs.within_ship_cap(w.ram, potatoes + [(40, 34)])
    assert kept == [(40, 34)] and sorted(held) == sorted(potatoes)


def test_harvest_task_leaves_capped_tiles_on_the_plant(monkeypatch):
    use(monkeypatch, "score")
    w = world(shipped={"corn": 496})
    tiles = ripe_field(w, CORN, 3)
    task = HarvestTask(state_name=None)
    task.reset(w)
    assert task._initial_target_count == 0                 # nothing to pick: the route ends SUCCESS
    assert not task._steps
    use(monkeypatch, "cash")
    task.reset(w)
    assert task._initial_target_count == len(tiles)


def test_world_probe_sees_no_harvest_when_every_ripe_tile_is_capped(monkeypatch):
    w = world(shipped={"corn": 496})
    ripe_field(w, CORN, 3)
    use(monkeypatch, "cash")
    assert WorldProbe.from_inputs(ram=w.ram).has_harvestable_crops()
    use(monkeypatch, "score")
    assert not WorldProbe.from_inputs(ram=w.ram).has_harvestable_crops()
    ripe_field(w, TOMATO, 1, x0=40)                         # tomatoes still have room
    assert WorldProbe.from_inputs(ram=w.ram).has_harvestable_crops()


# ── seeds ────────────────────────────────────────────────────────────

def test_cash_ranks_by_room_but_the_planner_override_pins_the_calendar_default(monkeypatch):
    # The rank itself already prefers the crop with room left; it is the planner's calendar
    # override (multi_day_planner) that kept Summer on corn in cash runs. Cash is untouched.
    use(monkeypatch, "cash")
    assert resolve_seed_type_for_date(SUMMER, 3, shipped={"corn": 0, "tomato": 0}) == "corn"
    assert resolve_seed_type_for_date(SUMMER, 3, shipped={"corn": 300, "tomato": 0}) == "tomato"
    assert resolve_seed_type_for_date(SPRING, 3, shipped={"potato": 511, "turnip": 0}) == "turnip"


def test_score_moves_summer_to_tomatoes_and_stops_sowing_a_capped_crop(monkeypatch):
    use(monkeypatch, "score")
    assert resolve_seed_type_for_date(SUMMER, 3, shipped={"corn": 0, "tomato": 0}) == "corn"
    assert resolve_seed_type_for_date(SUMMER, 3, shipped={"corn": 250, "tomato": 0}) == "tomato"
    assert resolve_seed_type_for_date(SUMMER, 3, shipped={"corn": 496, "tomato": 100}) == "tomato"
    assert resolve_seed_type_for_date(SUMMER, 3, shipped={"corn": 496, "tomato": 496}) is None


def test_score_balances_turnips_after_the_shop_stand_is_verified(monkeypatch):
    use(monkeypatch, "score")
    assert "turnip" in rs.BUYABLE_CROPS
    assert resolve_seed_type_for_date(SPRING, 3, shipped={"potato": 400, "turnip": 0}) == "turnip"
    assert resolve_seed_type_for_date(SPRING, 3, shipped={"potato": 496, "turnip": 0}) == "turnip"


def test_a_pick_without_shipped_totals_is_not_filtered(monkeypatch):
    # Callers that only ask "is a bag affordable/plantable today" pass no totals.
    use(monkeypatch, "score")
    assert resolve_seed_type_for_date(SPRING, 3) == "potato"


# ── rings ────────────────────────────────────────────────────────────

def ripe_potato_ring(w):
    center = POCKET_PLANT_CENTERS[0]
    for other in POCKET_PLANT_CENTERS:
        for tx, ty in plot_tiles(other, include_center=False):
            set_tile(w.ram, tx, ty, 0x54)
    for tx, ty in plot_tiles(center, include_center=False):
        set_tile(w.ram, tx, ty, POTATO)
    return center


def test_a_capped_potato_ring_is_not_replanted_but_a_cash_ring_is(monkeypatch):
    w = world(season=SPRING, day=20, shipped={"potato": 496})
    center = ripe_potato_ring(w)
    use(monkeypatch, "cash")
    assert pocket_plant_targets(w.ram, replant_ripe=True) == (center,)
    use(monkeypatch, "score")
    assert pocket_plant_targets(w.ram, replant_ripe=True) == ()
    set_field(w.ram, "shipped_potatoes", 100)               # room again: the ring replants
    assert pocket_plant_targets(w.ram, replant_ripe=True) == (center,)


# ── the day plan names one crop ──────────────────────────────────────

def day_task(monkeypatch, name, *, season, day, shipped, year=0):
    from harvest.planner.day_plan import MultiDayPlannerTask
    from tests.test_day_plan_calendar_events import _make_ram

    use(monkeypatch, name)
    ram = _make_ram(season=season, day=day, money=5000)
    set_field(ram, "year", year)
    for crop, count in shipped.items():
        set_field(ram, SHIPPED_FIELD[crop], count)
    return MultiDayPlannerTask()._build_day_task(SimpleNamespace(ram=ram))


def test_the_shop_buys_the_crop_the_plan_names(monkeypatch):
    # s2 labelled 42 Summer days "tomato" while the policy still bought corn: nothing was planted.
    task = day_task(monkeypatch, "score", season=SUMMER, day=3, shipped={"corn": 300, "tomato": 0})
    assert task.seed_type == "tomato" and task.policy.summer_crop == "tomato"
    cash = day_task(monkeypatch, "cash", season=SUMMER, day=3, shipped={"corn": 300, "tomato": 0})
    assert cash.seed_type == "corn" and cash.policy.summer_crop == "corn"      # the calendar default stays


def test_spring_shop_and_plan_agree_on_turnips(monkeypatch):
    from harvest.planner.day_phase_berry import _berry_run_phases

    task = day_task(monkeypatch, "score", season=SPRING, day=10, year=1, shipped={"potato": 400, "turnip": 0})
    phases = _berry_run_phases(is_sunday=False, hour=6, has_seeds=False, policy=task.policy,
                              season=SPRING, day=10, money=5000, has_plant_capacity=True)
    buy = next(p for p in phases if p.phase == "BUY_SEEDS")
    assert task.seed_type == buy.params["seed_type"] == "turnip"
    assert buy.params["stock_field"] == "turnip_seeds"


def test_first_year_funds_the_farm_with_potatoes_before_balancing_spring_score(monkeypatch):
    early = day_task(monkeypatch, "score", season=SPRING, day=3, shipped={"potato": 8, "turnip": 0})
    later = day_task(monkeypatch, "score", season=SPRING, day=3, year=1, shipped={"potato": 8, "turnip": 0})
    assert early.seed_type == early.policy.spring_crop == "potato"
    assert later.seed_type == later.policy.spring_crop == "turnip"


def test_full_potato_commitment_leaves_room_for_turnips(monkeypatch):
    use(monkeypatch, "score")
    w = world(season=SPRING, day=14, shipped={"potato": 496, "turnip": 0})
    assert rs.one_shot_sowing_ok(w.ram)
    assert pocket_plant_targets(w.ram)


def test_standing_turnips_are_reserved_against_their_own_cap(monkeypatch):
    from harvest.planner.day_plan_status import resolve_seed_type_from_ram

    use(monkeypatch, "score")
    w = world(season=SPRING, day=14, shipped={"potato": 496, "turnip": 440})
    for i in range(56):
        set_tile(w.ram, 10 + i % 40, 40 + i // 40, 0x66)
    assert not rs.one_shot_sowing_ok(w.ram)
    assert resolve_seed_type_from_ram(w.ram) is None


def test_indoor_seed_choice_reserves_the_persistent_field(monkeypatch):
    from harvest.core.harvest_state import FARM_MAP_ARRAY_ADDR
    from harvest.core.ram_catalog import live_wram_base
    from harvest.planner.day_plan_status import resolve_seed_type_from_ram
    from harvest.runtime.rom_tools import wram_offset

    use(monkeypatch, "score")
    w = world(season=SPRING, day=14, shipped={"potato": 496, "turnip": 440})
    w.ram[ADDR_TILEMAP_BYTE] = 0x16
    base = live_wram_base(w.ram) + wram_offset(FARM_MAP_ARRAY_ADDR)
    for i in range(56):
        w.ram[base + (40 + i // 40) * 64 + 10 + i % 40] = 0x66
    assert resolve_seed_type_from_ram(w.ram) is None


def test_a_fully_capped_spring_sows_nothing_but_late_summer_is_left_alone(monkeypatch):
    capped = day_task(monkeypatch, "score", season=SPRING, day=10, shipped={"potato": 496, "turnip": 496})
    assert not capped.policy.include_planting
    open_ = day_task(monkeypatch, "score", season=SPRING, day=10, shipped={"potato": 100})
    assert open_.policy.include_planting
    # No crop can mature after Summer D20 whatever the strategy says; that is the calendar, not the cap.
    late = day_task(monkeypatch, "score", season=SUMMER, day=25, shipped={"corn": 100})
    assert late.policy.include_planting
    cash = day_task(monkeypatch, "cash", season=SPRING, day=10, shipped={"potato": 700})
    assert cash.policy.include_planting


def test_spring_stops_sowing_once_shipped_plus_standing_potatoes_reach_the_cap(monkeypatch):
    # e2e s4: the cap left ~55 ripe potato tiles standing, the rings stayed full, Y3 Summer planted no corn.
    w = world(season=SPRING, day=14, shipped={"potato": 440, "turnip": 496})  # only potatoes have room
    use(monkeypatch, "score")
    assert len(pocket_plant_targets(w.ram)) == len(POCKET_PLANT_CENTERS)
    for i in range(56):                                               # standing + shipped now fill the window
        set_tile(w.ram, 10 + i % 40, 40 + i // 40, 0x54)
    assert pocket_plant_targets(w.ram) == ()
    assert not rs.one_shot_sowing_ok(w.ram)
    use(monkeypatch, "cash")
    assert len(pocket_plant_targets(w.ram)) > 0                       # cash sows as before


def test_spring_can_use_turnips_when_potatoes_are_full_and_summer_is_unaffected(monkeypatch):
    use(monkeypatch, "score")
    w = world(season=SPRING, day=14, shipped={"potato": 490})
    assert rs.one_shot_sowing_ok(w.ram)                                # turnips have room
    set_field(w.ram, "shipped_turnips", 496)
    assert rs.one_shot_sowing_ok(w.ram)                                # one last bag can fill the six-crop deficit
    set_field(w.ram, "shipped_potatoes", 496)
    assert not rs.one_shot_sowing_ok(w.ram)
    summer = world(season=SUMMER, day=3, shipped={"potato": 490})
    assert rs.one_shot_sowing_ok(summer.ram)                           # corn and tomato regrow: not a one-shot cap
    indoors = world(season=SPRING, day=14, shipped={"potato": 490})
    indoors.ram[ADDR_TILEMAP_BYTE] = 0x04                              # town: use the persistent farm array
    assert rs.one_shot_sowing_ok(indoors.ram)


# ── herd ─────────────────────────────────────────────────────────────

def econ_ram(**fields):
    ram = np.zeros(0x20000, dtype=np.uint8)
    for name, value in fields.items():
        set_field(ram, name, value)
    return ram


def herd_ram(**extra):
    base = dict(year=1, season=2, day=5, weekday=2, num_cows=4, num_chickens=2, money=150_000,
                planted_grass=170, stored_grass=40)
    return econ_ram(**(base | extra))


def test_a_bigger_herd_needs_the_winter_hay_a_fall_cow_requires(monkeypatch):
    # e2e h5 bought cows 5-8 into a Winter the stock could not feed: slots 0-2 went 255 -> 8.
    thin = herd_ram(stored_grass=40)
    stocked = herd_ram(stored_grass=700)
    use(monkeypatch, "cash")
    assert econ.cow_target(thin) == econ.COW_TARGET_LATER == econ.cow_target(stocked)   # cash is unchanged
    use(monkeypatch, "score_herd")
    assert econ.cow_target(thin) == 4                       # no more until the hay is in
    assert econ.cow_target(stocked) == 12
    assert econ.cow_target(herd_ram(season=1, stored_grass=40)) == 12       # the gate is a Fall gate


def test_late_cow_purchases_may_not_dip_the_wallet_under_the_money_term(monkeypatch):
    base = dict(year=2, season=0, day=3, weekday=2, num_cows=4, num_chickens=2, planted_grass=170,
                stored_grass=400, money=112_000)
    use(monkeypatch, "score_herd")
    assert econ.livestock_block(econ_ram(**base), "cow") == "cash_floor"             # 112,000 - 5,000 < 110,000
    assert econ.livestock_block(econ_ram(**(base | dict(money=130_000))), "cow") != "cash_floor"
    assert econ.livestock_block(econ_ram(**(base | dict(year=1, money=112_000))), "cow") != "cash_floor"   # Year 2: money is not yet the limit
    use(monkeypatch, "cash")
    assert econ.livestock_block(econ_ram(**base), "cow") != "cash_floor"


def test_herd_targets_move_only_under_score_herd(monkeypatch):
    later = econ_ram(year=1, season=1, day=10, num_cows=4)
    flock = econ_ram(year=1, season=1, day=10, num_chickens=2)
    for name in ("cash", "score"):
        use(monkeypatch, name)
        assert econ.cow_target(later) == econ.COW_TARGET_LATER
        assert econ.kept_adults(flock) == econ.HENS_KEPT == 2
    use(monkeypatch, "score_herd")
    assert econ.cow_target(later) == 12 > econ.COW_TARGET_LATER
    # Hens share the 16-tile-per-animal pasture gate with cows: 6 of them blocked cow 5 (e2e h2).
    assert econ.kept_adults(flock) == econ.HENS_KEPT == 2
    use(monkeypatch, "perfect")
    assert econ.kept_adults(flock) == 12


def test_full_herd_pasture_covers_the_last_cow_and_ending_flock(monkeypatch):
    use(monkeypatch, "score_herd")
    growing = herd_ram(season=0, day=10, num_cows=11, planted_grass=256, stored_grass=400)
    assert econ.next_livestock(growing) == "cow"
    assert econ.pasture_target_tiles(growing) >= econ.purchase_grass_required(11, 2, "cow")
    full = herd_ram(year=2, season=1, day=1, num_cows=12, num_chickens=12, planted_grass=400, stored_grass=400)
    assert econ.next_livestock(full) is None
    assert econ.pasture_target_tiles(full) >= econ.GRASS_PER_ANIMAL * 24


def test_score_flock_waits_for_cow_care_window_and_still_needs_feed(monkeypatch):
    base = dict(year=2, season=0, day=24, num_chickens=2, planted_grass=240)
    for strategy in ("score", "score_herd"):
        use(monkeypatch, strategy)
        assert econ.kept_adults(econ_ram(**base)) == 2
        assert not econ.hatch_allowed(econ_ram(**base))
        ready = econ_ram(**(base | dict(day=25)))
        assert econ.kept_adults(ready) == 12
        assert econ.hatch_allowed(ready)
        assert not econ.hatch_allowed(econ_ram(**(base | dict(day=25, planted_grass=0))))
        assert not econ.hatch_allowed(econ_ram(**(base | dict(day=25, num_chickens=12))))
    use(monkeypatch, "cash")
    assert econ.kept_adults(econ_ram(**(base | dict(day=25)))) == 2


@pytest.mark.rom
def test_score_flock_incubates_an_egg_from_a_natural_final_spring_pin(monkeypatch):
    from pathlib import Path
    from harvest.core.animal_status import is_incubating
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE
    from harvest.planner.day_phase_chicken import NAV_TO_COOP_PHASE, ENTER_COOP_PHASE, EXIT_COOP_PHASE
    from harvest.planner.day_plan import DayPlanTask
    from harvest.planner.day_plan_phases import build_day_phases_from_ram
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import make_action
    from retro_harness import AuditedEnv, AuditCapabilities, TaskStatus, WorldState

    pin = "PR_e2es4_Y3_Spring_D11"
    if not (Path(__file__).resolve().parents[1] / "custom_integrations" / "HarvestMoon-Snes" / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is absent")
    use(monkeypatch, "score")
    # The pin is Y3 Spring D11 (index 250); the final flock now starts at ``final_flock_start_index`` (264, D25), so a
    # D11 morning correctly ships its eggs. The env override is the supported way to start the flock on this pin.
    monkeypatch.setenv("HARVEST_FINAL_FLOCK_START", "250")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.final_flock"))
    try:
        env.reset()
        assert not is_incubating(env.get_ram())
        before = read_ram_value(env.get_ram(), "egg_available")
        chores = next(p for p in build_day_phases_from_ram(env.get_ram()) if p.phase == "COOP_CHORES")
        assert chores.params["egg_mode"] == "auto"
        plan = DayPlanTask(phase_sequence=[EXIT_TO_FARM_PHASE, NAV_TO_COOP_PHASE, ENTER_COOP_PHASE, chores, EXIT_COOP_PHASE])
        plan.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        for frame in range(12_000):
            result = plan.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
        assert result.status == TaskStatus.SUCCESS, result.reason
        assert is_incubating(env.get_ram())
        assert read_ram_value(env.get_ram(), "egg_available") < before
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


def test_final_flock_starts_late_enough_to_spare_the_hay() -> None:
    """One hen hatches every 3 days and a chick eats nothing for 8: twelve must arrive near the end, not by Summer D12.

    Year 3 runs started the flock at index 240/250, had twelve hens at 281 and ran out of hay on Summer D23, which made
    the cows sick (player happiness -200, cow points -11). Twelve hens still have to be in by the tally at index 299.
    """
    hatches = 10                      # twelve hens from two
    assert rs.SCORE.final_flock_start_index >= 262
    first_hatch = rs.SCORE.final_flock_start_index + 2
    assert first_hatch + 3 * (hatches - 1) <= 298, "ten hatches at one per three days must land before the tally"


def herd_with_children(*, hay=40):
    from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE, field_spec, live_wram_base

    ram = np.zeros(0x20000, dtype=np.uint8)
    for slot, (status, timer) in enumerate(((0x09, 0), (0x09, 0), (0x05, 3), (0x05, 18))):
        ram[COW_SLOT_BASE + slot * COW_SLOT_SIZE] = status
        ram[COW_SLOT_BASE + slot * COW_SLOT_SIZE + 2] = 0x27
        ram[COW_SLOT_BASE + slot * COW_SLOT_SIZE + 3] = timer
    for name, value in (("num_cows", 4), ("num_chickens", 6), ("stored_grass", hay)):
        spec = field_spec(name)
        base = spec.address + (live_wram_base(ram) if spec.live_offset else 0)
        ram[base] = value & 0xFF
        if spec.kind != "u8":
            ram[base + 1] = value >> 8
    return ram


def test_a_herd_strategy_counts_only_mouths_the_rom_feeds(monkeypatch):
    from harvest.planner import livestock_econ as econ

    monkeypatch.setattr(rs, "active", lambda: rs.get("score_herd"))
    # Two adults, the child a week from grown, not the new calf, and only the kept pair of six hens.
    assert econ.hay_eaters(herd_with_children()) == 3 + econ.HENS_KEPT


def test_cash_still_counts_every_animal(monkeypatch):
    from harvest.planner import livestock_econ as econ

    monkeypatch.setattr(rs, "active", lambda: rs.get("cash"))
    assert econ.hay_eaters(herd_with_children()) == 10


def test_a_short_loft_goes_to_the_cows_before_a_surplus_hen():
    from harvest.planner import livestock_econ as econ

    assert econ.hens_to_feed(herd_with_children(hay=5)) == econ.HENS_KEPT
    assert econ.hens_to_feed(herd_with_children(hay=16)) == 6  # two adults' reserve held back, six hens eat


def test_year_three_reaches_the_seed_shop_before_it_forages(monkeypatch):
    """y3e Summer D1: two forage loops ran to 14:00, the shop trip left at 15:13 and reached a closed door at 17:05.
    Saturday and Sunday are shop-closed and Monday went to grass seed, so corn went in on D7 and tomatoes never."""
    from harvest.planner.day_phase_berry import _berry_run_phases

    task = day_task(monkeypatch, "score_herd", season=SUMMER, day=1, year=2, shipped={"corn": 281, "tomato": 470})
    seeds_needed = dict(is_sunday=False, hour=10, has_seeds=False, policy=task.policy, season=SUMMER, day=1,
                        money=300_000, has_plant_capacity=True)

    third_year = [p.phase for p in _berry_run_phases(**seeds_needed, year=2)]
    assert third_year[:3] == ["BUY_SEEDS_WINDOW", "NAV_FARM_EXIT", "BUY_SEEDS"]
    assert "MOUNTAIN_BERRY" not in third_year
    window = next(p for p in _berry_run_phases(**seeds_needed, year=2) if p.phase == "BUY_SEEDS_WINDOW")
    assert window.params["latest_hour"] <= 14   # the walk is ~1h50 and the door shuts at 17:00

    # the first two years still forage first: their cash and timeline are the proven power-on path
    second_year = [p.phase for p in _berry_run_phases(**seeds_needed, year=1)]
    assert "MOUNTAIN_BERRY" in second_year


def test_summer_starts_the_unstarted_crop_after_the_first_one_is_in_the_ground(monkeypatch):
    """corn standing, tomatoes (26 short of the cap) not started: the next shop trip is for tomatoes, in Year 3 only."""
    from harvest.planner.day_plan_status import resolve_seed_type_from_ram

    use(monkeypatch, "score_herd")
    w = world(season=SUMMER, day=4, shipped={"corn": 281, "tomato": 470})
    for center in POCKET_PLANT_CENTERS[:3]:
        for x, y in plot_tiles(center):
            set_tile(w.ram, x, y, CORN)
    set_field(w.ram, "year", 2)
    assert resolve_seed_type_from_ram(w.ram) == "tomato"
    set_field(w.ram, "year", 1)
    assert resolve_seed_type_from_ram(w.ram) == "corn"


def test_a_busy_morning_counts_only_the_ripe_tiles_the_harvest_will_pick(monkeypatch):
    """Tiles left on the plant at the ship cap are not work: fixD's Y3 Summer D22-D29 "busy morning (80 ripe)"
    was capped corn and tomatoes, and it blocked every cover trip of the last week (development 42, not 49)."""
    use(monkeypatch, "score_herd")
    w = world(season=SUMMER, day=26, shipped={"corn": 496, "tomato": 490})
    ripe_field(w, CORN, 40, x0=12, y=34)                    # capped corn stays on the plant
    tomato = ripe_field(w, TOMATO, 10, x0=12, y=36)         # room for six more tomatoes
    probe = WorldProbe.from_inputs(ram=w.ram)
    assert probe.ripe_crop_count() == 6
    assert len(tomato) == 10
    set_field(w.ram, "shipped_tomatoes", 496)               # nothing ripe is worth picking any more
    assert WorldProbe.from_inputs(ram=w.ram).ripe_crop_count() == 0
    use(monkeypatch, "cash")                                # an uncapped strategy picks every ripe tile
    set_field(w.ram, "shipped_corn", 100)
    assert WorldProbe.from_inputs(ram=w.ram).ripe_crop_count() == 50
