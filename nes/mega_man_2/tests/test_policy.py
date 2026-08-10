from __future__ import annotations

from mega_man_2.policy import AirManPolicy, AirScreen1Policy, HeatManPolicy
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


def test_air_man_level1_phases() -> None:
    pol = AirManPolicy(start="level1", target_camera_screen=2)
    # early hop
    early = pol.tick(frame=1, health=28, camera_x_screen=0)
    assert "jump" in early.reason
    # land jump window
    land = pol.tick(frame=217, health=26, camera_x_screen=0)
    assert land.reason.startswith("land_jump")
    # mid approach after land_frame
    mid = pol.tick(frame=302, health=26, camera_x_screen=1)  # i=301
    assert mid.reason in {"run", "run_jump", "run_shoot", "run_jump_shoot"}
    # gap jump: land_frame + gap_rel = 301 + 142 = 443 → frame 444
    gap = pol.tick(frame=444, health=24, camera_x_screen=1)
    assert gap.reason.startswith("gap_jump")
    # clear
    done = pol.tick(frame=500, health=22, camera_x_screen=2)
    assert done.reason == "clear_hold"


def test_air_man_landed_start() -> None:
    pol = AirManPolicy(start="landed", target_camera_screen=2)
    # relative gap at 142 → frame 143
    gap = pol.tick(frame=143, health=26, camera_x_screen=1)
    assert gap.reason.startswith("gap_jump")
    early = pol.tick(frame=10, health=26, camera_x_screen=1)
    assert early.reason in {"run", "run_jump", "run_shoot", "run_jump_shoot"}


def test_air_man_screen2_phases() -> None:
    pol = AirManPolicy(start="screen2", target_camera_screen=4)
    # before approach: walk only (frame 1 → i=0)
    walk = pol.tick(frame=1, health=22, camera_x_screen=2)
    assert walk.reason in {"run", "run_shoot"}
    # approach hop: frame 49 → i=48 opens jump window
    approach = pol.tick(frame=49, health=22, camera_x_screen=2)
    assert "jump" in approach.reason
    # fan hold window: frame 146 → i=145
    fan = pol.tick(frame=146, health=20, camera_x_screen=2)
    assert fan.reason.startswith("fan_hold")
    # late period after fan_end=180: frame 181 → i=180
    late = pol.tick(frame=181, health=20, camera_x_screen=3)
    assert late.reason.startswith("late_jump")
    # clear at target
    done = pol.tick(frame=500, health=16, camera_x_screen=4)
    assert done.reason == "clear_hold"


def test_heat_man_policy_jumps_and_clears() -> None:
    pol = HeatManPolicy(target_camera_screen=1)
    # frame 1 → i=0: jump + shoot windows open
    t0 = pol.tick(frame=1, health=28, camera_x_screen=0)
    assert t0.reason == "run_jump_shoot"
    assert list(t0.action) == list(nes_action("RIGHT", "A", "B"))
    # clear
    done = pol.tick(frame=50, health=24, camera_x_screen=1)
    assert done.reason == "clear_hold"
    assert list(done.action) == list(nes_idle_action())
    dead = pol.tick(frame=10, health=0, camera_x_screen=0)
    assert dead.reason == "dead"
