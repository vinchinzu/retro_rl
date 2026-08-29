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


def test_heat_man_screen2_phases() -> None:
    pol = HeatManPolicy(start="screen2", target_camera_screen=3)
    # mid window: frame 1 → i=0 under mid_period 60 / hold 14
    mid = pol.tick(frame=1, health=24, camera_x_screen=2)
    assert mid.reason.startswith("mid_jump")
    # late handoff after mid_until=260 → frame 261 → i=260
    late = pol.tick(frame=261, health=28, camera_x_screen=2)
    assert late.reason.startswith("late_jump")
    done = pol.tick(frame=400, health=28, camera_x_screen=3)
    assert done.reason == "clear_hold"


def test_heat_man_screen3_pillar_phase() -> None:
    pol = HeatManPolicy(start="screen3", target_camera_screen=4)
    # i=0: (0+10)%25=10 >= hold 10 → no jump
    walk = pol.tick(frame=1, health=28, camera_x_screen=3)
    assert walk.reason in {"run", "run_shoot"}
    # i=15: (15+10)%25=0 < 10 → jump
    hop = pol.tick(frame=16, health=28, camera_x_screen=3)
    assert hop.reason.startswith("pillar_jump")


def test_heat_man_start_for_state() -> None:
    assert HeatManPolicy.start_for_state("Heat1") == "early"
    assert HeatManPolicy.start_for_state("HeatScreen1") == "early"
    assert HeatManPolicy.start_for_state("HeatScreen2") == "screen2"
    assert HeatManPolicy.start_for_state("HeatScreen3_scr3_hp28") == "screen3"
    assert HeatManPolicy.start_for_state("HeatScreen4") == "screen4"
    assert HeatManPolicy.start_for_state("HeatScreen5Ground") == "screen5"
    assert HeatManPolicy.start_for_state("HeatScreen5") == "screen5"
    assert HeatManPolicy.start_for_state("HeatScreen6") == "screen5"
    assert HeatManPolicy.start_for_state("HeatScreen7") == "screen7"
    assert HeatManPolicy.start_for_state("HeatScreen7Mid") == "screen7"
    assert HeatManPolicy.start_for_state("HeatLadder") == "screen7"
    assert HeatManPolicy.start_for_state("HeatScreen8") == "screen8"
    assert HeatManPolicy.start_for_state("HeatScreen8Yoku") == "screen8"
    assert HeatManPolicy.start_for_state("HeatS8Left_88_148") == "screen8"


def test_heat_man_screen5_idle_then_j1() -> None:
    pol = HeatManPolicy(start="screen5", target_camera_screen=7)
    # idle frames release A for rising edge
    idle = pol.tick(frame=1, health=26, camera_x_screen=5, tile_feet=1)
    assert idle.reason == "s5_idle"
    assert list(idle.action) == list(nes_idle_action())
    idle2 = pol.tick(frame=2, health=26, camera_x_screen=5, tile_feet=1)
    assert idle2.reason == "s5_idle"
    # j1 window opens
    j1 = pol.tick(frame=3, health=26, camera_x_screen=5, tile_feet=1)
    assert j1.reason.startswith("s5_j1")
    assert list(j1.action) == list(nes_action("RIGHT", "A")) or list(
        j1.action
    ) == list(nes_action("RIGHT", "A", "B"))
    # clear hold
    done = pol.tick(frame=10, health=22, camera_x_screen=7, tile_feet=1)
    assert done.reason == "clear_hold"


def test_heat_man_screen7_left_then_climb() -> None:
    pol = HeatManPolicy(start="screen7", target_camera_screen=8)
    left = pol.tick(frame=1, health=22, camera_x_screen=7, tile_feet=1)
    assert left.reason == "s7_left_off"
    assert list(left.action) == list(nes_action("LEFT"))
    # after 12 LEFT frames, climb A+LEFT
    climb = pol.tick(frame=13, health=22, camera_x_screen=7, tile_feet=0)
    assert climb.reason == "s7_climb"
    assert list(climb.action) == list(nes_action("A", "LEFT"))
    # ladder forces DOWN
    lad = pol.tick(frame=50, health=18, camera_x_screen=7, tile_feet=2)
    assert lad.reason == "s7_ladder_down"
    assert list(lad.action) == list(nes_action("DOWN"))
    done = pol.tick(frame=10, health=18, camera_x_screen=8, tile_feet=1)
    assert done.reason == "clear_hold"


def test_heat_man_screen8_yoku_approach() -> None:
    pol = HeatManPolicy(start="screen8", target_camera_screen=99)
    # frames 1–187 → i=0..186 wait no-ceiling phase
    wait = pol.tick(frame=1, health=18, camera_x_screen=8, tile_feet=1)
    assert wait.reason == "s8_wait"
    assert list(wait.action) == list(nes_idle_action())
    wait_last = pol.tick(frame=187, health=18, camera_x_screen=8, tile_feet=1)
    assert wait_last.reason == "s8_wait"
    # frame 188 → i=187 LEFT approach
    left = pol.tick(frame=188, health=18, camera_x_screen=8, tile_feet=1)
    assert left.reason == "s8_approach"
    assert list(left.action) == list(nes_action("LEFT"))
    # frame 196 → i=195 idle release
    rel = pol.tick(frame=196, health=18, camera_x_screen=8, tile_feet=1)
    assert rel.reason == "s8_release"
    assert list(rel.action) == list(nes_idle_action())
    # frame 197 → i=196 A+LEFT first yoku jump
    hop = pol.tick(frame=197, health=18, camera_x_screen=8, tile_feet=1)
    assert hop.reason == "s8_yoku_jump"
    assert list(hop.action) == list(nes_action("A", "LEFT"))


def test_heat_man_screen8_catch_and_down() -> None:
    pol = HeatManPolicy(start="screen8", target_camera_screen=9)
    # wait 187 + LEFT8 + idle1 + A+LEFT14 + LEFT18 = 228; then gap 4
    # frame 229 → i=228 catch gap
    gap = pol.tick(frame=229, health=18, camera_x_screen=8, tile_feet=1)
    assert gap.reason == "s8_catch_gap"
    # frame 233 → i=232 A catch
    catch = pol.tick(frame=233, health=18, camera_x_screen=8, tile_feet=1)
    assert catch.reason == "s8_catch"
    assert list(catch.action) == list(nes_action("A"))
    # late DOWN window
    down = pol.tick(frame=400, health=18, camera_x_screen=8, tile_feet=2)
    assert down.reason == "s8_down"
    assert list(down.action) == list(nes_action("DOWN"))
    done = pol.tick(frame=400, health=18, camera_x_screen=9, tile_feet=2)
    assert done.reason == "clear_hold"
