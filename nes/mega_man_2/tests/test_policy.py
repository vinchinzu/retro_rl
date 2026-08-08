from __future__ import annotations

from mega_man_2.policy import AirScreen1Policy
from retro_harness.nes import nes_action, nes_idle_action


def test_air_screen1_holds_right_and_jumps() -> None:
    pol = AirScreen1Policy(jump_period=50, jump_hold=12, shoot_period=40, shoot_hold=2)
    # frame 1 → i=0: jump window open
    t0 = pol.tick(frame=1, health=28, camera_x_screen=0)
    # i=0 hits both jump hold and shoot pulse windows
    assert t0.reason == "run_jump_shoot"
    assert list(t0.action) == list(nes_action("RIGHT", "A", "B"))

    # mid jump window still jumping
    t5 = pol.tick(frame=6, health=28, camera_x_screen=0)
    assert "run_jump" in t5.reason

    # after jump hold: walk only (frame 13 → i=12)
    t12 = pol.tick(frame=13, health=28, camera_x_screen=0)
    assert t12.reason == "run"
    assert list(t12.action) == list(nes_action("RIGHT"))


def test_air_screen1_clear_and_dead() -> None:
    pol = AirScreen1Policy()
    done = pol.tick(frame=10, health=26, camera_x_screen=1)
    assert done.reason == "clear_hold"
    assert list(done.action) == list(nes_idle_action())

    dead = pol.tick(frame=10, health=0, camera_x_screen=0)
    assert dead.reason == "dead"

    fallen = pol.tick(frame=10, health=20, camera_x_screen=0, fallen=True)
    assert fallen.reason == "dead"
