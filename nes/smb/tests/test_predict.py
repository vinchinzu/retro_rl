"""SMB stepper claims + first-miss halt (no ROM)."""

from __future__ import annotations

from retro_harness.predict import first_miss_index
from smb.approx import press, rollout, step
from smb.observation import level1_start
from smb.predict import grade_player, grade_trajectory, halt_plan, predict_step


def test_predict_step_matches_stepper() -> None:
    player = level1_start()
    nxt, claim = predict_step(player, press("RIGHT"))
    rolled = step(player, press("RIGHT"))
    assert nxt.x == rolled.x
    assert grade_player(claim, rolled).ok


def test_grade_misses_wrong_x() -> None:
    player = level1_start()
    nxt, claim = predict_step(player, press("RIGHT"))
    fake = nxt.__class__(**{**nxt.to_dict(), "x": nxt.x + 8})
    grade = grade_player(claim, fake)
    assert not grade.ok
    assert any(part.startswith("x=") for part in grade.missed)


def test_halt_plan_on_pixel_miss() -> None:
    player = level1_start()
    predicted = rollout(player, [press("RIGHT")] * 8)
    observed_hit = list(predicted)
    observed_miss = rollout(player, [press("LEFT")] * 8)
    hit_grades = grade_trajectory(predicted, observed_hit)
    miss_grades = grade_trajectory(predicted, observed_miss)
    assert halt_plan(hit_grades) is False
    assert halt_plan(miss_grades) is True
    assert first_miss_index(miss_grades) == 1


def test_halt_plan_empty_grades() -> None:
    assert halt_plan([]) is False
