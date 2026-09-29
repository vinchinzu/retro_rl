"""ExitBuildingTask must not blind-walk right into furniture from the bed (rr-bu62.20)."""

from types import SimpleNamespace

from harvest.planner.tasks.transitions import ExitBuildingTask
from harvest.tasks.nav import Point


def _pressed(action):
    from harvest.tasks.nav import make_action  # noqa: F401

    return action


def test_stalled_rightward_walk_steps_down():
    task = ExitBuildingTask()
    task._navigator = SimpleNamespace(current_pos=Point(89, 86))
    acts = [task._house_exit_action() for _ in range(25)]
    # Action array layout: find index that differs between right and down.
    from harvest.tasks.nav import make_action

    right, down = make_action(right=True, b=True), make_action(down=True)
    assert (acts[0] == right).all()
    assert (acts[-1] == down).all()


def test_moving_walk_keeps_going_right():
    task = ExitBuildingTask()
    from harvest.tasks.nav import make_action

    for x in range(70, 120, 2):
        task._navigator = SimpleNamespace(current_pos=Point(x, 86))
        assert (task._house_exit_action() == make_action(right=True, b=True)).all()
