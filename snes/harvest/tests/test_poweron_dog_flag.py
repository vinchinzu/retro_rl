"""Dog pickup flag (game_state 0x0800) guards and recovery (rr-bu62.19 / .21)."""

import numpy as np

from harvest.planner.tasks import home_recover as hr
from harvest.tasks.nav import make_action

A = hr._A_INDEX
B = hr._B_INDEX


def _ram(gs=0x4401, player=(100, 100), dog=(110, 100), tilemap=0, held=0, hour=6, day=16):
    from harvest.core.ram_catalog import field_spec, live_wram_base

    ram = np.zeros(0x24000, dtype=np.uint8)
    base = live_wram_base(ram)
    # Live get_ram() is 0x24000 and mirrors WRAM at +0x4000. The stuck bit
    # is present in both windows (measured on LN_poweron_D27_bed).
    ram[0xD2], ram[0xD3] = gs & 0xFF, (gs >> 8) & 0xFF
    if base:
        ram[base + 0xD2], ram[base + 0xD3] = gs & 0xFF, (gs >> 8) & 0xFF
    fields = (
        ("input_lock", 1),
        ("tilemap", tilemap),
        ("player_x", player[0]),
        ("player_y", player[1]),
        ("hour", hour),
        ("minute", 0),
        ("day", day),
        ("season", 0),
        ("year", 0),
        ("held_item", held),
    )
    for name, val in fields:
        spec = field_spec(name)
        idx = spec.address + (base if spec.live_offset else 0)
        ram[idx] = val & 0xFF
        if spec.kind == "u16":
            ram[idx + 1] = (val >> 8) & 0xFF
    idx = hr.DOG_POS_ADDR + base
    ram[idx], ram[idx + 1] = dog[0] & 0xFF, dog[0] >> 8
    ram[idx + 2], ram[idx + 3] = dog[1] & 0xFF, dog[1] >> 8
    return ram


def _sleep_planner(gs, tilemap=0x15):
    """Planner entering sleep on a fake morning, before any bed attempt."""
    from retro_harness import WorldState

    from harvest.planner.multi_day_planner import MultiDayPlannerTask

    world = WorldState(frame=0, ram=_ram(gs=gs, tilemap=tilemap, player=(72, 86)), info={}, obs=None)
    planner = MultiDayPlannerTask()
    planner.reset(world)
    planner._phase = "sleep"
    planner._current_task = None
    planner._dog_released_today = False
    result = planner.step(world)
    return planner, result


def test_stuck_flag_detection():
    # Stuck is bit 0x0800, with or without the 0x0080 warp/carry bit.
    assert hr.dog_pickup_stuck(_ram(gs=0x4C01))
    assert not hr.dog_pickup_stuck(_ram(gs=0x4401))
    assert hr.dog_pickup_stuck(_ram(gs=0x4C81))


def test_a_becomes_b_beside_dog():
    ram = _ram()
    out = hr.strip_dog_hug_press(make_action(a=True), ram)
    assert not out[A] and out[B]


def test_a_kept_when_dog_far_or_indoors():
    far = _ram(dog=(300, 300))
    assert hr.strip_dog_hug_press(make_action(a=True), far)[A]
    indoors = _ram(tilemap=0x15)
    assert hr.strip_dog_hug_press(make_action(a=True), indoors)[A]


def test_release_task_may_press_a_and_planner_obeys_the_flag():
    from types import SimpleNamespace

    from retro_harness import ActionResult, TaskResult, TaskStatus

    from harvest.planner.multi_day_planner import MultiDayPlannerTask

    assert hr.DogReleaseTask.allows_dog_a is True
    world = SimpleNamespace(ram=_ram())
    planner = MultiDayPlannerTask()
    pressed = TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(a=True)))
    planner._step_inner = lambda world: pressed
    planner._current_task = hr.DogReleaseTask()
    assert planner.step(world).action.action[A]
    planner._current_task = object()
    stripped = planner.step(world).action.action
    assert not stripped[A] and stripped[B]


def test_a_kept_while_text_box_open():
    ram = _ram()
    from harvest.core.ram_catalog import field_spec, live_wram_base

    spec = field_spec("input_lock")
    ram[spec.address + (live_wram_base(ram) if spec.live_offset else 0)] = 2
    assert hr.strip_dog_hug_press(make_action(a=True), ram)[A]


def test_stuck_morning_selects_dog_release_before_bed():
    """Sleep with bit 0x0800 (with or without 0x0080) picks DogReleaseTask first.

    A clear flag starts GoToSleepTask with zero bed attempts. Stripping A on a
    non-dog task is test_release_task_may_press_a_and_planner_obeys_the_flag.
    """
    from harvest.planner.tasks.home_sleep import GoToSleepTask

    for gs in (0x4C01, 0x4C81):
        planner, result = _sleep_planner(gs)
        assert isinstance(planner._current_task, hr.DogReleaseTask)
        assert planner._phase == "rescue"
        assert not isinstance(planner._current_task, GoToSleepTask)
        assert result.reason and result.reason.startswith("dog release")
        assert getattr(planner._current_task, "allows_dog_a", False) is True

    planner, result = _sleep_planner(0x4401)
    assert isinstance(planner._current_task, GoToSleepTask)
    assert planner._phase == "sleep"
    assert planner._current_task._sleep_attempts == 0
    assert result.reason is None or "dog release" not in result.reason

    # 0x0080 alone is a warp/carry bit, not the stuck pickup animation.
    planner, _result = _sleep_planner(0x4481)
    assert isinstance(planner._current_task, GoToSleepTask)
    assert planner._current_task._sleep_attempts == 0
