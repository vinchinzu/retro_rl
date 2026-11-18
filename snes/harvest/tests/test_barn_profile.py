"""The barn profile's bookkeeping: labels, the walk split into moving and waiting, px walked, resets."""
from __future__ import annotations

from types import SimpleNamespace

from harvest.tools.barn_profile import breakdown, frame_label, labelled_frames, render


def profile_rows():
    rows = []
    x = 100
    for frame in range(10):  # a walk: 2 px every other frame, as the 12-cow barn plays it
        x += 2 if frame % 2 else 0
        rows.append((frame, "COW_CHORES", 0x27, "milk.walk", int(frame % 2 == 1), x, 200))
    rows += [(10 + i, "COW_CHORES", 0x27, "queue:milk", 0, x, 200) for i in range(4)]
    rows += [(14 + i, "COW_CHORES", 0x27, "milk.verify", 0, x, 200) for i in range(3)]
    rows += [(17, "COW_CHORES", 0x27, "plan", 0, x, 200), (18, "EXIT_BARN", 0x27, None, 0, x, 200), (19, "EXIT_BARN", 0x00, None, 0, 328, 360)]
    for frame in range(8):  # stuck behind a cow: pad down, farmer still
        rows.append((20 + frame, "COW_CHORES", 0x27, "ship.walk", 0, x, 200))
    return rows


def profile(rows=None):
    return {
        "frames": 28,
        "rows": rows or profile_rows(),
        "phase_results": [{"phase": "COW_CHORES", "status": "retry"}, {"phase": "COW_CHORES", "status": "success"}],
        "homes": {0: 0x27, 1: 0x27},
        "happy_after": {0: 255, 1: 250},
    }


def test_the_profile_splits_a_walk_into_moving_and_waiting_and_counts_the_pixels():
    report = breakdown(profile())
    assert report["stages"]["milk.walk.move"] + report["stages"].get("milk.walk.wait", 0) == 10
    assert report["stages"]["ship.walk.wait"] == 8
    assert report["stages"]["milk.queue"] == 4 and report["stages"]["milk.verify"] == 3 and report["stages"]["plan"] == 1
    assert report["px_walked"]["milk"] == 10  # five 2 px moves
    assert report["resets"] == 1 and report["cows_home"] and report["min_happiness"] == 250
    assert report["phases"]["EXIT_BARN"] == 2 and report["barn_frames"] == 28


def test_the_report_names_the_job_stage_with_the_most_frames_first():
    text = render([breakdown(profile())])
    assert "milk.walk.move" in text and "resets 1.00" in text
    assert text.index("milk.walk") < text.index("plan")


def test_frame_label_reads_the_task_state():
    job = SimpleNamespace(kind="talk", stage="step")
    idle = SimpleNamespace(_job=None, _action_queue=[], _walker=SimpleNamespace(_queue=[]))
    assert frame_label(idle) == "plan"
    assert frame_label(SimpleNamespace(_job=job, _action_queue=[], _walker=SimpleNamespace(_queue=[]))) == "talk.step"
    assert frame_label(SimpleNamespace(_job=job, _action_queue=[1], _walker=SimpleNamespace(_queue=[]))) == "queue:talk"
    walking = SimpleNamespace(kind="milk", stage="walk")
    assert frame_label(SimpleNamespace(_job=walking, _action_queue=[], _walker=SimpleNamespace(_queue=[1]))) == "milk.walk.nudge"


def test_the_patch_is_undone_on_exit():
    from harvest.tasks.cow_task import CowChoresTask

    before = CowChoresTask.step
    with labelled_frames({}):
        assert CowChoresTask.step is not before
    assert CowChoresTask.step is before
