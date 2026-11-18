"""CowChoresTask: what the chores owe, in what order, and how a pass ends. The ROM side is ``barn_fuzz``."""
from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from cow_test_helpers import ADULT, BABY, CHILD, PREGNANT, SICK, make_barn_ram, make_world, set_cow, set_player_px

from harvest.core.animal_status import (
    COW_DAILY_BRUSHED_FLAG,
    COW_DAILY_MILKED_FLAG,
    COW_DAILY_TALKED_FLAG,
    ITEM_FODDER,
)
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP
from harvest.planner.day_phase_cow import splice_barn_reset
from harvest.tasks.cow_geometry import BLOCKED_REASON
from harvest.tasks.cow_task import (
    ADDR_HELD_ITEM,
    BRUSH,
    BRUSH_TOOL_ID,
    FODDER,
    JOB_FRAMES,
    MAX_STRIKES,
    MEDICINE,
    MEDICINE_TOOL_ID,
    MILK,
    MILKER_TOOL_ID,
    SHIP,
    TALK,
    TROUGH,
    CowChoresTask,
    Job,
)
from retro_harness import TaskStatus

CARED = COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG
TOOLS = dict(tool_selected=BRUSH_TOOL_ID, tool_backpack=MILKER_TOOL_ID)
MILK_BOTTLE = 0x16
SETTLED = 60


def task_on(ram, **kwargs) -> CowChoresTask:
    task = CowChoresTask(**kwargs)
    task.reset(make_world(ram))
    return task


def job(task, ram) -> Job | None:
    return task._next_job(ram)


# ── what is owed, in what order ───────────────────────────────────────


def test_fodder_comes_before_any_cow_care():
    ram = make_barn_ram(cows=3, hay=40, **TOOLS)
    assert job(task_on(ram), ram).kind == FODDER


def test_a_child_or_a_calf_is_owed_no_trough_because_the_night_pass_never_tests_it():
    ram = make_barn_ram(cows=2, status=CHILD, hay=40, **TOOLS)
    set_cow(ram, 2, status=BABY)
    task = task_on(ram)
    assert task._required_spots(ram) == ()
    assert job(task, ram).kind == TALK


def test_only_barn_adults_take_hay_in_a_mixed_herd():
    ram = make_barn_ram(cows=3, status=ADULT, hay=40, **TOOLS)
    set_cow(ram, 1, status=CHILD)
    assert {spot.flag for spot in task_on(ram)._required_spots(ram)} == {0x0001, 0x0004}


def test_a_pregnant_cow_is_fed_at_her_own_trough_not_her_slot_bit():
    ram = make_barn_ram(cows=1, status=PREGNANT, hay=40, **TOOLS)
    assert [spot.flag for spot in task_on(ram)._required_spots(ram)] == [0x1000]


def test_a_gap_in_the_slots_still_feeds_the_right_bits():
    ram = make_barn_ram(cows=0, hay=40, **TOOLS)
    set_cow(ram, 3)
    set_cow(ram, 11)
    assert {spot.flag for spot in task_on(ram)._required_spots(ram)} == {0x0008, 0x0800}


def test_fodder_in_hand_goes_to_the_empty_trough_nearest_the_crossing():
    ram = make_barn_ram(cows=12, fed_flags=0x0004, hay=40, held_item=ITEM_FODDER, **TOOLS)
    picked = job(task_on(ram), ram)
    assert picked.kind == TROUGH
    assert picked.spot.flag == 0x0100  # row y=213 is the closest to y=184, and its right trough is full


def test_milk_in_hand_is_shipped_before_anything_else():
    ram = make_barn_ram(cows=2, hay=40, held_item=MILK_BOTTLE, **TOOLS)
    assert job(task_on(ram), ram).kind == SHIP


def test_no_hay_goes_straight_to_care():
    ram = make_barn_ram(cows=2, hay=0, **TOOLS)
    assert job(task_on(ram), ram).kind == TALK


def test_talk_then_brush_on_the_nearest_cow_and_milk_last():
    ram = make_barn_ram(cows=0, hay=0, player_px=(180, 280), **TOOLS)
    set_cow(ram, 0, (168, 278))
    set_cow(ram, 5, (168, 86))
    task = task_on(ram)
    first = job(task, ram)
    assert (first.kind, first.slot) == (TALK, 0)
    set_cow(ram, 0, (168, 278), flags=COW_DAILY_TALKED_FLAG)
    second = job(task, ram)
    assert (second.kind, second.slot) == (BRUSH, 0)
    set_cow(ram, 0, (168, 278), flags=CARED)
    assert job(task, ram).slot == 5
    set_cow(ram, 5, (168, 86), flags=CARED)
    last = job(task, ram)
    assert (last.kind, last.slot) == (MILK, 0)


def test_the_next_cow_is_the_nearest_by_the_walk_not_by_the_straight_line():
    """From the centre aisle the east cow is 58 px away as the crow flies and ~250 by the map: the west cow is nearer."""
    ram = make_barn_ram(cows=0, hay=0, player_px=(112, 120), **TOOLS)
    set_cow(ram, 0, (168, 118))
    set_cow(ram, 6, (56, 86))
    assert job(task_on(ram), ram).slot == 6


def test_the_pass_works_the_tool_in_hand_first_and_swaps_once():
    both = dict(cows=0, hay=0, player_px=(180, 280))
    brush_first = make_barn_ram(tool_selected=BRUSH_TOOL_ID, tool_backpack=MILKER_TOOL_ID, **both)
    milker_first = make_barn_ram(tool_selected=MILKER_TOOL_ID, tool_backpack=BRUSH_TOOL_ID, **both)
    for ram in (brush_first, milker_first):
        set_cow(ram, 0, (168, 278))
        set_cow(ram, 5, (168, 86))
    assert job(task_on(brush_first), brush_first).kind == TALK
    picked = job(task_on(milker_first), milker_first)
    assert (picked.kind, picked.slot) == (MILK, 0)
    # The milker's pass finishes before the brush's: no cow is walked to twice for a swap of 69 frames.
    for slot in (0, 5):
        set_cow(milker_first, slot, (168, 278 if slot == 0 else 86), flags=COW_DAILY_MILKED_FLAG)
    assert job(task_on(milker_first), milker_first).kind == TALK


def test_a_cow_on_the_bin_stand_holds_back_the_milk_pass_but_not_the_care_pass():
    from harvest.tasks.cow_geometry import BARN_SHIP_BIN_PX

    ram = make_barn_ram(cows=0, hay=0, tool_selected=MILKER_TOOL_ID, tool_backpack=BRUSH_TOOL_ID, player_px=(180, 280))
    set_cow(ram, 0, BARN_SHIP_BIN_PX, flags=CARED)
    set_cow(ram, 1, (168, 246))
    picked = job(task_on(ram), ram)
    assert (picked.kind, picked.slot) == (TALK, 1)
    set_cow(ram, 1, (168, 246), flags=CARED | COW_DAILY_MILKED_FLAG)
    assert job(task_on(ram), ram) is None  # only the milk is left, and it has nowhere to go


def test_a_bottle_waits_off_the_bin_stand_while_a_cow_is_on_it():
    """Fuzz seed 20 nudged a cow parked on the bin for 1,300 frames, struck out, and lost the bottle to the reset."""
    from harvest.tasks.cow_geometry import BARN_SHIP_BIN_PX, BIN_HOLD_PX

    ram = make_barn_ram(cows=1, hay=0, held_item=MILK_BOTTLE, player_px=BIN_HOLD_PX, **TOOLS)
    set_cow(ram, 0, (BARN_SHIP_BIN_PX[0] - 2, BARN_SHIP_BIN_PX[1] - 4), flags=CARED | COW_DAILY_MILKED_FLAG)
    task = task_on(ram)
    world = make_world(ram)
    for _ in range(40):
        assert task.step(world).status == TaskStatus.RUNNING
    assert task._job.kind == SHIP and task._job.stage == "walk" and task._job.presses == 0
    assert task._job_stand(ram, task._job) == (BIN_HOLD_PX, 4)
    set_cow(ram, 0, (128, 250), flags=CARED | COW_DAILY_MILKED_FLAG)  # she wandered off: the stand is the goal again
    assert task._job_stand(ram, task._job)[0] == BARN_SHIP_BIN_PX


def test_the_fodder_press_stands_within_one_pixel_of_the_dispenser():
    """A run stopped 2 px short (x=214) pressed out of reach of the dispenser: every third pickup was a 60-frame miss."""
    ram = make_barn_ram(cows=1, hay=40, **TOOLS)
    task = task_on(ram)
    from harvest.tasks.cow_geometry import FODDER_STAND_PX

    assert task._job_stand(ram, Job(FODDER)) == (FODDER_STAND_PX, 1)


def test_a_stand_that_missed_is_the_last_one_tried_again():
    ram = make_barn_ram(cows=0, hay=0, player_px=(150, 100), **TOOLS)
    set_cow(ram, 0, (30, 100))
    task = task_on(ram)
    job_ = Job(MILK, slot=0)
    first = task._pick_cow_stand(ram, job_, (30, 100))
    task._tried[job_.key] = [first]
    second = task._pick_cow_stand(ram, job_, (30, 100))
    assert second != first


def test_a_cow_whose_job_struck_out_waits_while_other_cows_are_owed():
    ram = make_barn_ram(cows=0, hay=0, player_px=(185, 270), **TOOLS)
    set_cow(ram, 0, (168, 278))
    set_cow(ram, 5, (168, 86))
    task = task_on(ram)
    task._step_count = 100
    task._job_failed(ram, Job(TALK, slot=0), "test")
    assert job(task, ram).slot == 5
    set_cow(ram, 5, (168, 86), flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG)
    assert job(task, ram).slot == 0  # nobody else is owed: she is tried again before the cooldown is over
    task._step_count = 100 + 10_000
    assert not task._cooling(0)


def test_a_bottle_with_no_milked_flag_is_shipped_not_pressed_again():
    """Seen on the ROM: a cow pinned to the west wall gave a bottle from her east side but her flag never set."""
    ram = make_barn_ram(cows=1, hay=0, **TOOLS)
    set_cow(ram, 0, (14, 121), flags=CARED)
    task = task_on(ram)
    task._job = Job(MILK, slot=0, stage="verify", presses=1, wait=1_000, settled=SETTLED, stand=((31, 121), "left"), held_before=0)
    ram[ADDR_HELD_ITEM] = MILK_BOTTLE
    task.step(make_world(ram))
    assert task._strikes[(MILK, 0)] == 1 and task._tried[(MILK, 0)] == [((31, 121), "left")]
    assert task._next_job(ram).kind == SHIP


def test_a_cow_at_the_happiness_cap_owes_no_talk_or_brush_only_when_asked():
    ram = make_barn_ram(cows=2, hay=0, **TOOLS)
    set_cow(ram, 0, happiness=255)
    set_cow(ram, 1, happiness=254)
    assert task_on(ram)._needs(ram, 0) == [TALK, BRUSH, MILK]
    capped = task_on(ram, skip_capped=True)
    assert capped._needs(ram, 0) == [MILK]
    assert capped._needs(ram, 1) == [TALK, BRUSH, MILK]
    receipt = capped.care_receipt(ram)
    assert receipt[0]["talked"] and receipt[0]["brushed"] and not receipt[0]["complete"]  # milk still owed


def test_the_cap_switch_is_read_from_the_environment(monkeypatch):
    ram = make_barn_ram(cows=1, hay=0, **TOOLS)
    set_cow(ram, 0, happiness=255)
    monkeypatch.delenv("HARVEST_BARN_SKIP_CAPPED", raising=False)
    assert TALK in task_on(ram)._needs(ram, 0)
    monkeypatch.setenv("HARVEST_BARN_SKIP_CAPPED", "1")
    assert TALK not in task_on(ram)._needs(ram, 0)


def test_a_finished_pass_with_capped_cows_reports_talk_and_brush_as_done():
    ram = done_ram()
    for slot in range(2):
        set_cow(ram, slot, flags=COW_DAILY_MILKED_FLAG, happiness=255)
    task = task_on(ram, skip_capped=True)
    result = task.step(make_world(ram))
    assert result.status == TaskStatus.SUCCESS and "talk=True brush=True" in result.reason


def test_milk_is_owed_only_by_a_healthy_adult():
    ram = make_barn_ram(cows=0, hay=0, **TOOLS)
    set_cow(ram, 0, status=CHILD, flags=CARED)
    set_cow(ram, 1, status=SICK, flags=CARED)
    task = task_on(ram)
    assert job(task, ram) is None


def test_a_cranky_cow_owes_no_milk_because_the_rom_refuses_status_0x70():
    ram = make_barn_ram(cows=0, hay=0, **TOOLS)
    set_cow(ram, 0, status=0x19, flags=CARED)
    assert job(task_on(ram), ram) is None


def test_an_empty_stamina_bar_owes_no_brush_because_the_swing_never_happens():
    from harvest.core.tile_catalog import ADDR_STAMINA

    ram = make_barn_ram(cows=1, hay=0, **TOOLS)
    set_cow(ram, 0, flags=COW_DAILY_TALKED_FLAG | COW_DAILY_MILKED_FLAG)
    ram[ADDR_STAMINA] = 0
    assert job(task_on(ram), ram) is None


def test_no_milk_is_owed_from_17_00_because_the_bin_destroys_it():
    from harvest.core.ram_catalog import field_spec, live_wram_base

    ram = make_barn_ram(cows=1, hay=0, **TOOLS)
    set_cow(ram, 0, flags=CARED)
    assert job(task_on(ram), ram).kind == MILK
    spec = field_spec("hour")
    ram[spec.address + (live_wram_base(ram) if spec.live_offset else 0)] = 17
    assert job(task_on(ram), ram) is None


def test_a_brush_or_milker_left_in_the_shed_owes_nothing():
    ram = make_barn_ram(cows=1, hay=0)
    set_cow(ram, 0, flags=COW_DAILY_TALKED_FLAG)
    assert job(task_on(ram), ram) is None


def test_the_cure_runs_before_talk_and_only_with_a_bottle_in_the_pair():
    ram = make_barn_ram(cows=2, hay=0, tool_selected=MEDICINE_TOOL_ID, tool_backpack=MILKER_TOOL_ID)
    set_cow(ram, 1, status=SICK)
    picked = job(task_on(ram), ram)
    assert (picked.kind, picked.slot) == (MEDICINE, 1)
    ram = make_barn_ram(cows=2, hay=0, **TOOLS)
    set_cow(ram, 1, status=SICK)
    assert job(task_on(ram), ram).kind == TALK


def test_medicine_is_never_aimed_at_a_healthy_cow_or_a_sick_calf():
    ram = make_barn_ram(cows=1, hay=0, tool_selected=MEDICINE_TOOL_ID, tool_backpack=0)
    set_cow(ram, 0, flags=CARED)
    assert job(task_on(ram), ram) is None
    set_cow(ram, 0, status=0x03 | 0x20, flags=CARED)
    assert job(task_on(ram), ram) is None


def test_the_cure_phase_owes_nothing_but_the_cure():
    ram = make_barn_ram(cows=2, hay=40, tool_selected=MEDICINE_TOOL_ID, tool_backpack=0)
    assert job(task_on(ram, talk=False, brush=False, milk=False, feed=False), ram) is None
    set_cow(ram, 0, status=SICK)
    assert job(task_on(ram, talk=False, brush=False, milk=False, feed=False), ram).kind == MEDICINE


def test_a_cow_that_walked_out_of_the_door_is_not_chased_around_the_barn():
    from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE

    ram = make_barn_ram(cows=2, hay=0, **TOOLS)
    ram[COW_SLOT_BASE + 2] = 0  # Spring: her home map is the season number, 0
    assert task_on(ram)._barn_slots(ram) == [1]


# ── how a pass ends ───────────────────────────────────────────────────


def done_ram(cows: int = 2, **kwargs):
    ram = make_barn_ram(cows=cows, fed_flags=(1 << cows) - 1, **{**TOOLS, **kwargs})
    for slot in range(cows):
        set_cow(ram, slot, flags=CARED | COW_DAILY_MILKED_FLAG, happiness=100 + slot)
    return ram


def test_a_finished_pass_reports_the_receipt_the_bench_parses():
    ram = done_ram()
    task = task_on(ram)
    result = task.step(make_world(ram))
    assert result.status == TaskStatus.SUCCESS
    assert "milk=0 ship=0 fed=0 flags=0x0003" in result.reason
    assert "happy=0:100,1:101" in result.reason
    assert "talk=True brush=True" in result.reason


def test_an_empty_loft_is_a_note_not_a_failure():
    ram = done_ram(hay=0)
    ram_flags = make_barn_ram(cows=2, hay=0, **TOOLS)
    for slot in range(2):
        set_cow(ram_flags, slot, flags=CARED | COW_DAILY_MILKED_FLAG)
    result = task_on(ram_flags).step(make_world(ram_flags))
    assert result.status == TaskStatus.SUCCESS
    assert "hay_short=1" in result.reason


def test_a_care_job_struck_out_ends_the_pass_with_the_reset_reason():
    ram = done_ram()
    set_cow(ram, 1, flags=COW_DAILY_BRUSHED_FLAG | COW_DAILY_MILKED_FLAG)
    task = task_on(ram)
    world = make_world(ram)
    task._strikes[(TALK, 1)] = MAX_STRIKES
    result = task.step(world)
    assert result.status == TaskStatus.FAILURE
    assert "incomplete cow care {1: ['talk']}" == result.reason


def test_a_strike_moves_on_to_the_next_job():
    ram = make_barn_ram(cows=2, hay=0, **TOOLS)
    task = task_on(ram)
    world = make_world(ram)
    task.step(world)
    first = task._job
    task._job_failed(ram, first, "test")
    assert task._job is None and task._strikes[first.key] == 1
    task._strikes[first.key] = MAX_STRIKES
    assert job(task, ram).key != first.key


def test_a_job_over_budget_takes_a_strike_not_the_barn():
    ram = make_barn_ram(cows=1, hay=0, **TOOLS)
    task = task_on(ram)
    world = make_world(ram)
    task.step(world)
    task._job.frames = JOB_FRAMES[TALK]
    result = task.step(world)
    assert result.status == TaskStatus.RUNNING
    assert task._strikes[(TALK, 0)] == 1


def test_a_milk_swing_started_before_the_deadline_finishes_with_its_bottle():
    from harvest.tasks.cow_task import ADDR_PLAYER_ACTION

    ram = make_barn_ram(cows=1, hay=0, **TOOLS)
    set_cow(ram, 0, flags=CARED)
    task = task_on(ram, shift_frames=10)
    task._step_count = 5
    task._job = Job(MILK, slot=0, stage="verify", frames=JOB_FRAMES[MILK], presses=1)
    ram[ADDR_PLAYER_ACTION] = 0x0A  # tool animation, bottle not in hand yet
    world = make_world(ram)
    for _ in range(20):
        assert task.step(world).status == TaskStatus.RUNNING
        assert task._job is not None and task._job.kind == MILK
    set_cow(ram, 0, flags=CARED | COW_DAILY_MILKED_FLAG)
    ram[ADDR_PLAYER_ACTION] = 0
    ram[ADDR_HELD_ITEM] = MILK_BOTTLE
    for _ in range(40):
        result = task.step(world)
        assert result.status == TaskStatus.RUNNING
        if task._job is not None and task._job.kind == SHIP:
            break
    assert task._job.kind == SHIP
    assert not task._strikes.get((MILK, 0))


def test_milk_that_cannot_reach_the_bin_asks_for_a_fresh_barn():
    ram = make_barn_ram(cows=1, hay=0, held_item=MILK_BOTTLE, **TOOLS)
    task = task_on(ram)
    world = make_world(ram)
    task.step(world)
    task._strikes[(SHIP, None)] = MAX_STRIKES - 1
    result = task._job_failed(ram, task._job, "test")
    assert result.status == TaskStatus.FAILURE and BLOCKED_REASON in result.reason


def test_the_reasons_a_pass_ends_with_are_the_ones_the_day_plan_answers():
    class Schedule:
        spliced: list = []

        def splice_at(self, index, phases):
            self.spliced = list(phases)

    for reason in (f"{BLOCKED_REASON} at (1, 2)", "incomplete cow care {1: ['talk']}", "missing required cow feed flags=0x0001"):
        schedule = Schedule()
        assert splice_barn_reset(schedule, 0, object(), reason)
        assert len(schedule.spliced) == 3


def test_over_the_frame_budget_the_day_plan_can_still_reset():
    ram = make_barn_ram(cows=1, hay=0, **TOOLS)
    task = task_on(ram, timeout=5)
    world = make_world(ram)
    task._step_count = 6
    result = task.step(world)
    assert result.status == TaskStatus.FAILURE and BLOCKED_REASON in result.reason


def test_a_long_pass_walks_out_before_the_herd_can_reach_the_door():
    from harvest.tasks.cow_geometry import BARN_SHIFT_REASON

    ram = make_barn_ram(cows=2, hay=0, **TOOLS)
    task = task_on(ram, shift_frames=10)
    task._step_count = 11
    result = task.step(make_world(ram))
    assert result.status == TaskStatus.FAILURE and BARN_SHIFT_REASON in result.reason

    class Schedule:
        spliced: list = []

        def splice_at(self, index, phases):
            self.spliced = list(phases)

    schedule = Schedule()
    assert splice_barn_reset(schedule, 0, object(), result.reason) and len(schedule.spliced) == 3


def test_the_shift_waits_for_empty_hands():
    ram = make_barn_ram(cows=2, hay=0, held_item=MILK_BOTTLE, **TOOLS)
    task = task_on(ram, shift_frames=10)
    task._step_count = 11
    assert task.step(make_world(ram)).status == TaskStatus.RUNNING


def test_a_finished_pass_is_never_held_back_by_the_shift():
    ram = done_ram()
    task = task_on(ram, shift_frames=10)
    task._step_count = 11
    assert task.step(make_world(ram)).status == TaskStatus.SUCCESS


def test_a_cow_near_the_door_ends_the_shift_before_she_reaches_it():
    from harvest.tasks.cow_geometry import BARN_SHIFT_REASON

    ram = make_barn_ram(cows=2, hay=0, **TOOLS)
    set_cow(ram, 0, (150, 340))
    result = task_on(ram).step(make_world(ram))
    assert result.status == TaskStatus.FAILURE and BARN_SHIFT_REASON in result.reason


def test_a_cow_in_the_hall_away_from_the_door_costs_no_reset():
    """The hall rows lead nowhere: only a cow within BARN_DOOR_DANGER_PX of the door tile is worth ~650 frames."""
    ram = make_barn_ram(cows=2, hay=0, **TOOLS)
    for spot in ((185, 300), (60, 320), (40, 340)):
        set_cow(ram, 0, spot)
        assert task_on(ram).step(make_world(ram)).status == TaskStatus.RUNNING, spot


def test_a_cow_walking_at_the_door_in_the_middle_of_a_job_ends_the_pass_at_once():
    """Fuzz seed 78: a cow went from 57 px to the door tile in 350 frames, between two job boundaries."""
    from harvest.tasks.cow_geometry import BARN_SHIFT_REASON

    ram = make_barn_ram(cows=2, hay=0, player_px=(100, 80), **TOOLS)
    task = task_on(ram)
    world = make_world(ram)
    assert task.step(world).status == TaskStatus.RUNNING and task._job is not None and task._job.stage == "walk"
    set_cow(ram, 1, (150, 330))
    result = task.step(world)
    assert result.status == TaskStatus.FAILURE and BARN_SHIFT_REASON in result.reason


def test_a_cow_near_the_door_waits_for_a_bottle_to_be_shipped_first():
    ram = make_barn_ram(cows=2, hay=0, held_item=MILK_BOTTLE, player_px=(100, 200), **TOOLS)
    set_cow(ram, 1, (150, 330))
    task = task_on(ram)
    world = make_world(ram)
    for _ in range(5):
        assert task.step(world).status == TaskStatus.RUNNING
    assert task._job.kind == SHIP


def test_the_shift_leaves_a_pregnant_cow_alone_and_never_runs_with_hands_full():
    ram = make_barn_ram(cows=2, hay=0, **TOOLS)
    set_cow(ram, 0, (136, 340), status=PREGNANT)
    assert task_on(ram).step(make_world(ram)).status == TaskStatus.RUNNING
    set_cow(ram, 0, (136, 340))
    ram[ADDR_HELD_ITEM] = MILK_BOTTLE
    assert task_on(ram).step(make_world(ram)).status == TaskStatus.RUNNING


def test_the_cure_only_pass_never_asks_for_a_shift():
    """A cow in the hall at 1f failed COW_CURE in a fuzz herd seed, and the cure phase has no reset to answer it."""
    ram = make_barn_ram(cows=2, hay=0, tool_selected=MEDICINE_TOOL_ID, tool_backpack=0)
    set_cow(ram, 0, (136, 340), status=SICK)
    task = task_on(ram, talk=False, brush=False, milk=False, feed=False)
    assert task.step(make_world(ram)).status == TaskStatus.RUNNING
    assert task._job.kind == MEDICINE


def test_a_task_outside_the_barn_is_blocked():
    ram = make_barn_ram(cows=1)
    task = task_on(ram)
    ram[ADDR_TILEMAP] = 0x00
    assert task.step(make_world(ram)).status == TaskStatus.BLOCKED
    assert not task.can_start(make_world(ram))


# ── a press and its receipt ───────────────────────────────────────────


def test_open_text_is_paged_with_taps_and_no_job_runs():
    ram = make_barn_ram(cows=1, hay=0, **TOOLS)
    task = task_on(ram)
    ram[ADDR_INPUT_LOCK] = 2
    world = make_world(ram)
    pressed = [int(task.step(world).action.action[8]) for _ in range(24)]
    assert task._job is None
    assert 0 < sum(pressed) < 24


def run_until_press(task, ram, limit=200):
    world = make_world(ram)
    for _ in range(limit):
        task.step(world)
        if task._job is not None and task._job.stage == "verify":
            return
    raise AssertionError("never pressed")


def test_standing_on_a_cow_presses_and_the_flag_is_the_receipt():
    ram = make_barn_ram(cows=1, hay=0, player_px=(185, 278), **TOOLS)
    task = task_on(ram)
    run_until_press(task, ram)
    assert task._job.kind == TALK and task._job.presses == 1
    set_cow(ram, 0, flags=COW_DAILY_TALKED_FLAG)
    world = make_world(ram)
    for _ in range(40):
        task.step(world)
        if task._job is None or task._job.kind != TALK:
            break
    assert task.happiness_deltas and task.happiness_deltas[0]["kind"] == TALK


def test_a_press_with_no_receipt_walks_back_and_tries_another_stand():
    ram = make_barn_ram(cows=1, hay=0, player_px=(185, 278), **TOOLS)
    task = task_on(ram)
    run_until_press(task, ram)
    first = task._job.stand
    world = make_world(ram)
    for _ in range(200):
        task.step(world)
        if task._job.stage == "walk" and task._job.stand not in (None, first):
            break
    assert task._tried[task._job.key] == [first]
    assert task._job.stand != first, "the same stand missed once; the next press comes from another side"


def test_a_wrong_tool_is_swapped_with_x_before_walking():
    ram = make_barn_ram(cows=1, hay=0, tool_selected=MILKER_TOOL_ID, tool_backpack=BRUSH_TOOL_ID, player_px=(185, 278))
    set_cow(ram, 0, flags=COW_DAILY_TALKED_FLAG | COW_DAILY_MILKED_FLAG)
    task = task_on(ram)
    world = make_world(ram)
    actions = [task.step(world).action.action for _ in range(8)]
    assert task._job.kind == BRUSH and task._job.stage == "walk"
    assert sum(int(action[9]) for action in actions) >= 1  # X


def test_a_stand_with_a_cow_on_it_moves_the_fodder_press_to_the_north_stand():
    from harvest.tasks.cow_geometry import FODDER_NORTH_STAND_PX, FODDER_STAND_PX

    ram = make_barn_ram(cows=1, hay=40, **TOOLS)
    task = task_on(ram)
    assert task._fodder_stand(ram) == FODDER_STAND_PX
    set_cow(ram, 0, FODDER_STAND_PX)
    assert task._fodder_stand(ram) == FODDER_NORTH_STAND_PX


def test_a_dose_is_aimed_from_a_stand_no_other_cow_can_take_it_from():
    from harvest.tasks.cow_geometry import press_reaches

    ram = make_barn_ram(cows=0, hay=0, tool_selected=MEDICINE_TOOL_ID, tool_backpack=0, player_px=(200, 240))
    set_cow(ram, 0, (200, 210), status=SICK)
    set_cow(ram, 1, (200, 222))
    task = task_on(ram)
    stand, face = task._pick_cow_stand(ram, Job(MEDICINE, slot=0), (200, 210))
    assert press_reaches(stand, face, (200, 210))
    assert not press_reaches(stand, face, (200, 222))


def test_a_stall_cow_is_reached_along_her_lane():
    from harvest.tasks.cow_geometry import cow_stands

    assert [face for _stand, face in cow_stands((168, 86))] == ["left", "right"]
    assert "left" in [face for _stand, face in cow_stands((56, 278))] or "right" in [face for _stand, face in cow_stands((56, 278))]


def test_held_item_stays_in_the_hand_until_the_bin_takes_it():
    from cow_test_helpers import write_u24
    from harvest.tasks.harvest_task import ADDR_SHIPPING_MONEY

    ram = make_barn_ram(cows=1, hay=0, held_item=MILK_BOTTLE, player_px=(56, 348), **TOOLS)
    task = task_on(ram)
    run_until_press(task, ram)
    assert task._job.kind == SHIP
    ram[ADDR_HELD_ITEM] = 0
    assert not task._receipt_landed(ram, task._job)
    write_u24(ram, ADDR_SHIPPING_MONEY, 25)
    world = make_world(ram)
    for _ in range(80):
        task.step(world)
    assert task.milk_shipped_count == 1


def test_the_bin_accepts_a_near_stand_when_a_cow_blocks_the_far_corner():
    ram = make_barn_ram(cows=1, hay=0, held_item=MILK_BOTTLE, player_px=(59, 348), **TOOLS)
    task = task_on(ram)
    assert task._job_stand(ram, Job(SHIP)) == ((59, 348), 0)
    run_until_press(task, ram)
    assert task._job.kind == SHIP


def test_empty_hands_do_not_prove_a_trough_was_filled():
    from cow_test_helpers import write_u16
    from harvest.core.animal_status import ADDR_FED_COWS_FLAGS
    from harvest.tasks.cow_geometry import COW_FEED_SPOTS

    ram = make_barn_ram(cows=1, hay=0, **TOOLS)
    task = task_on(ram)
    feeding = Job(TROUGH, spot=COW_FEED_SPOTS[0])
    assert not task._receipt_landed(ram, feeding)
    write_u16(ram, ADDR_FED_COWS_FLAGS, feeding.spot.flag)
    assert task._receipt_landed(ram, feeding)


def test_exhausted_barn_retries_leave_the_rest_of_the_day_runnable():
    from harvest.planner.day_phase_cow import COW_CHORES_PHASE, MAX_BARN_RESETS
    from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec
    from harvest.planner.day_plan import DayPlanTask
    from retro_harness import TaskResult, WorldState

    ran = []

    class Phase:
        def __init__(self, name):
            self.name = name

        def reset(self, world):
            pass

        def can_start(self, world):
            return True

        def step(self, world):
            ran.append(self.name)
            return TaskResult(status=TaskStatus.FAILURE if self.name == "COW_CHORES" else TaskStatus.SUCCESS,
                              reason=f"{BLOCKED_REASON}: pinned" if self.name == "COW_CHORES" else "fields worked")

    class Plan(DayPlanTask):
        def _make_task(self, spec, world):
            return Phase(spec.phase)

    plan = Plan(phase_sequence=[COW_CHORES_PHASE, PhaseSpec("FIELDS", "sentinel")],
                policy=DayPlannerPolicy(include_end_day=False))
    world = WorldState(frame=0, ram=make_barn_ram(cows=1), info={}, obs=None)
    plan.reset(world)
    plan._barn_resets = MAX_BARN_RESETS
    for frame in range(10):
        world = WorldState(frame=frame, ram=world.ram, info={}, obs=None)
        result = plan.step(world)
        if result.status != TaskStatus.RUNNING:
            break
    assert result.status == TaskStatus.SUCCESS
    assert ran == ["COW_CHORES", "FIELDS"]
    assert any(row["phase"] == "COW_CHORES" and row["status"] == "skipped" for row in plan.phase_results)


def test_barn_door_exit_never_pulses_run_into_a_cow():
    from harvest.planner.day_phase_cow import EXIT_BARN_PHASE
    from harvest.planner.tasks.transitions import DirectionalTransitionTask

    ram = make_barn_ram(cows=1, player_px=(128, 356))
    set_cow(ram, 0, (128, 369))
    task = DirectionalTransitionTask(**EXIT_BARN_PHASE.params)
    task.reset(make_world(ram))
    task._navigator.update(ram)
    for frame in range(12):
        task._step_count = frame
        assert not task._barn_exit_action(ram)[0]


def test_cows_outside_reads_the_season_number_she_left_as_home():
    from harvest.core.animal_status import cows_outside
    from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE

    ram = make_barn_ram(cows=3, hay=0)
    assert cows_outside(ram) == []
    ram[COW_SLOT_BASE + COW_SLOT_SIZE + 2] = 1  # Summer
    assert cows_outside(ram) == [1]


def test_a_cow_outside_puts_the_roundup_in_the_morning_plan():
    from harvest.core.ram_catalog import COW_SLOT_BASE
    from harvest.planner.day_phase_catalog import cow_day_phases

    ram = make_barn_ram(cows=2, hay=40, **TOOLS)
    names = [spec.phase for spec in cow_day_phases(ram)]
    assert "COW_ROUNDUP" not in names
    ram[COW_SLOT_BASE + 2] = 0
    names = [spec.phase for spec in cow_day_phases(ram)]
    assert names.index("EXIT_TO_FARM") < names.index("COW_ROUNDUP") < names.index("NAV_TO_BARN")


def test_roundup_takes_her_to_the_path_row_then_the_door_column_then_north():
    from harvest.tasks.cow_task import ROUNDUP_DOOR_PX, ROUNDUP_ROW_Y, CowRoundupTask

    task = CowRoundupTask()
    assert task._plan((500, 450))[1] == "up"
    assert task._plan((500, ROUNDUP_ROW_Y))[1] == "left"
    assert task._plan((100, ROUNDUP_ROW_Y))[1] == "right"
    stand, push = task._plan((ROUNDUP_DOOR_PX[0], ROUNDUP_ROW_Y))
    assert push == "up" and stand[1] > ROUNDUP_ROW_Y


def test_a_pinned_cow_tries_the_other_leg():
    from harvest.tasks.cow_task import CowRoundupTask

    task = CowRoundupTask()
    assert task._plan((500, 450))[1] == "up"
    task._flip = True
    assert task._plan((500, 450))[1] == "left"
