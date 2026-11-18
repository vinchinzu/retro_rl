"""Year 3 house exit: a farmer held in place by an NPC steps round it instead of timing out.

D r45 Summer D18 (round 3 preview): after a good gift the farmer stood at (327,156) under Eve, a child at (328,169) sat on the exit
column, the blind walk only presses down, ``EXIT_TO_FARM`` timed out after 2,200 frames, the barn feed check failed too and the
12 cows went unfed (cow happiness -25, ROM 949). Ten such days in 120 Y3 Summers of the previews, all right after ``GIVE_WIFE``.
"""
from __future__ import annotations

import numpy as np
import pytest

from harvest.planner.tasks import transitions as tr
from harvest.tasks.nav import make_action
from retro_harness import WorldState
from tests.day_plan_test_helpers import make_world, set_player_pos
from tests.test_run_strategy import set_field

UP, DOWN, LEFT, RIGHT = 4, 5, 6, 7


def _task(year=2):
    world = make_world(0x17)
    set_field(world.ram, "year", year)
    set_player_pos(world.ram, 327, 156)
    task = tr.ExitBuildingTask(target_tilemap=0x00)
    task.reset(world)
    task._navigator.update(world.ram)
    return task, world


def _frames(task, world, n):
    out = []
    for _ in range(n):
        task._navigator.update(world.ram)
        out.append(task._sidestep_if_blocked(world.ram, make_action(down=True, b=True)))
    return out


def test_a_held_farmer_sidesteps_left_then_right():
    task, world = _task()
    first = _frames(task, world, tr.HOUSE_EXIT_STILL_FRAMES + 1)
    assert all(a[DOWN] for a in first[:-1]) and first[-1][LEFT] and not first[-1][DOWN]
    side = _frames(task, world, tr.HOUSE_EXIT_SIDESTEP_FRAMES)
    assert all(a[LEFT] for a in side[: tr.HOUSE_EXIT_SIDESTEP_FRAMES - 1])
    resumed = _frames(task, world, 1)[0]
    assert resumed[DOWN]                                                # the walk resumes after the sidestep
    again = _frames(task, world, tr.HOUSE_EXIT_STILL_FRAMES + 1)
    assert any(a[RIGHT] for a in again) and not any(a[LEFT] for a in again)     # still held: the other side


def test_a_farmer_who_is_moving_never_sidesteps():
    task, world = _task()
    for y in range(156, 156 + 120):
        set_player_pos(world.ram, 327, y)
        task._navigator.update(world.ram)
        assert task._sidestep_if_blocked(world.ram, make_action(down=True))[DOWN]


@pytest.mark.parametrize("year", [0, 1])
def test_years_one_and_two_keep_their_frames(year):
    task, world = _task(year)
    assert all(a[DOWN] and not a[LEFT] for a in _frames(task, world, 200))
