"""Shot-guard decisions against measured turn poses and synthetic NES RAM."""

import numpy as np
import pytest

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.shot_guard import (
    GuardedController, LinkModel, ShotGuard, read_forecast, simulate,
)
from zelda_i.ram import read_snapshot


def room_ram(*, mode=5, level=9, x=128, y=141):
    ram = np.zeros(0x10000, dtype=np.uint8)
    ram[0x12], ram[0x10], ram[0xEB] = mode, level, 0x05
    ram[0x70], ram[0x84], ram[0x98] = x, y, 8
    return ram


def object_ram(ram, *, type_id, x=64, y=141, facing=1, state=0, rem=0):
    ram[0x350], ram[0x71], ram[0x85] = type_id, x, y
    ram[0x99], ram[0xAD], ram[0x395], ram[0x486] = facing, state, rem, 0x20
    return ram


@pytest.mark.parametrize("start, facing, press, end", [
    ((120, 152), "DOWN", "RIGHT", (120, 149)),
    ((120, 152), "DOWN", "LEFT", (120, 149)),
    ((120, 154), "UP", "RIGHT", (120, 157)),
    ((120, 145), "UP", "LEFT", (120, 141)),
    ((120, 145), "DOWN", "LEFT", (120, 149)),
    ((123, 141), "RIGHT", "UP", (120, 141)),
    ((125, 141), "LEFT", "DOWN", (128, 141)),
    ((124, 141), "LEFT", "UP", (120, 141)),
    ((124, 141), "RIGHT", "UP", (128, 141)),
])
def test_cross_axis_press_slides_to_measured_turn_line(start, facing, press, end):
    model = LinkModel(None, start)
    x, y = start
    for _ in range(4):
        x, y, facing = model.step(x, y, facing, press)
        if (x, y) == end:
            break
    assert (x, y) == end
    x, y, facing = model.step(x, y, facing, press)
    assert facing == press


@pytest.mark.parametrize("press", ["UP", "DOWN", "LEFT", "RIGHT"])
def test_dodge_stops_before_non_floor_node(press):
    model = LinkModel(frozenset({(120, 141)}), (120, 141))
    assert model.step(120, 141, "UP", press)[:2] == (120, 141)


@pytest.mark.parametrize("start", [(32, 93), (208, 93), (32, 189), (208, 189), (207, 188)])
@pytest.mark.parametrize("press", ["UP", "DOWN", "LEFT", "RIGHT"])
def test_dodge_never_enters_door_lane_from_room_interior(start, press):
    model = LinkModel(None, start)
    x, y = start
    facing = "UP"
    for _ in range(40):
        x, y, facing = model.step(x, y, facing, press)
        assert 32 <= x <= 208 and 93 <= y <= 189


def test_model_can_follow_predecessor_already_in_door_lane():
    model = LinkModel(None, (120, 205))
    assert model.step(120, 205, "UP", "UP") == (120, 203.5, "UP")


@pytest.mark.parametrize("counter, rem, facing, wy, lx, ly, hit", [
    (31, 0, 1, 141, 128, 141, 20),
    (0, 0, 1, 141, 128, 141, None),
    (31, 40, 1, 141, 128, 141, None),
    (31, 0, 2, 141, 128, 141, None),
    (31, 0, 1, 141, 128, 157, None),
    (31, 0, 4, 93, 64, 141, 14),
    (31, 0, 4, 93, 72, 141, None),
    (31, 0, 8, 93, 64, 141, None),
])
def test_blue_shot_requires_period_alignment_and_facing(counter, rem, facing, wy, lx, ly, hit):
    ram = object_ram(room_ram(x=lx, y=ly), type_id=0x23, y=wy, facing=facing, rem=rem)
    ram[0x15] = counter
    forecast = read_forecast(ram, 28, bodies=False, blue_bodies=False)
    result = simulate(forecast, LinkModel(None), (lx, ly), "UP", [None] * 28)
    assert result.first_hit == hit


@pytest.mark.parametrize("state, fire, hit", [(0xB0, 0, 20), (0xB4, 4, 24), (0xCF, 31, None)])
def test_red_countdown_predicts_fire_frame(state, fire, hit):
    ram = object_ram(room_ram(), type_id=0x24, state=state)
    forecast = read_forecast(ram, 32, bodies=False, blue_bodies=False)
    shot, = forecast.movers
    assert shot.at(fire - 1) is None
    assert shot.at(fire) == (64, 141)
    assert shot.at(fire + 1) == (67, 141)
    assert simulate(forecast, LinkModel(None), (128, 141), "UP", [None] * 32).first_hit == hit


@pytest.mark.parametrize("state", [0x3F, 0x7F, 0xAF, 0xD1, 0xFF])
def test_red_does_not_forecast_an_already_fired_or_unplaced_shot(state):
    ram = object_ram(room_ram(), type_id=0x24, state=state)
    assert read_forecast(ram, 32, bodies=False, blue_bodies=False).empty


@pytest.mark.parametrize("type_id, state, expected", [
    (0x58, 0x10, True), (0x59, 0x1F, True), (0x53, 0x11, True),
    (0x60, 0x10, False), (0x55, 0x10, False), (0x58, 0, False),
    (0x59, 0x20, False),
])
def test_forecast_excludes_drops_and_non_colliding_shot_states(type_id, state, expected):
    forecast = read_forecast(object_ram(room_ram(), type_id=type_id, state=state), 32)
    assert bool(forecast.movers) is expected


@pytest.mark.parametrize("mode, level", [(5, 0), (6, 9), (7, 9), (9, 9), (11, 9), (17, 9)])
def test_guard_passes_through_outside_dungeon_play(mode, level):
    ram = object_ram(room_ram(mode=mode, level=level), type_id=0x58, x=100, state=0x10)
    inner = FrameAction(nes_action("LEFT"), "inner")
    guard = ShotGuard()
    assert guard.filter(read_snapshot(ram), ram, inner) is inner
    assert guard.report()["overrides"] == 0


def test_guard_swaps_only_press_with_predicted_hit_and_preserves_ram():
    ram = object_ram(room_ram(), type_id=0x58, x=80, state=0x10)
    original = ram.copy()
    guard = ShotGuard()
    inner = FrameAction(nes_idle_action(), "hold_lane")
    dodge = guard.filter(read_snapshot(ram), ram, inner)
    assert dodge.reason.startswith("guard_") and dodge.action != inner.action
    safe_ram = object_ram(ram.copy(), type_id=0x58, x=80, y=189, state=0x10)
    assert guard.filter(read_snapshot(safe_ram), safe_ram, inner) is inner
    np.testing.assert_array_equal(ram, original)


def test_guard_keeps_dodging_until_inner_plan_is_safe_for_entire_horizon():
    ram = object_ram(room_ram(), type_id=0x58, x=80, state=0x10)
    inner = FrameAction(nes_idle_action(), "hold_lane")
    guard = ShotGuard()
    assert guard.filter(read_snapshot(ram), ram, inner).reason.startswith("guard_")
    ram[0x71] = 40  # Hit after the 20-frame trigger, inside the 32-frame horizon.
    assert ShotGuard().filter(read_snapshot(ram), ram, inner) is inner
    assert guard.filter(read_snapshot(ram), ram, inner).reason.startswith("guard_")
    ram[0x85] = 189
    assert guard.filter(read_snapshot(ram), ram, inner) is inner


@pytest.mark.parametrize("terminal", ["success", "failed"])
def test_completed_stage_keeps_its_terminal_action_even_in_danger(terminal):
    class Finished:
        success = False
        failed = False

        def step(self, snap):
            setattr(self, terminal, True)
            return FrameAction(nes_idle_action(), "arrived")

    class Env:
        def get_ram(self):
            return object_ram(room_ram(), type_id=0x58, x=100, state=0x10)

    controller = GuardedController(Finished())
    controller.bind_env(Env())
    assert controller.step(read_snapshot(Env().get_ram())).reason == "arrived"
    assert getattr(controller, terminal)


def _shot_guarded(controller):
    """The ShotGuard layer, under an optional ROM ``PolicyGuard`` wrapper."""
    from zelda_i.rollout import PolicyGuard

    return controller.inner if isinstance(controller, PolicyGuard) else controller


def test_level9_chapters_guard_dungeon_stages_and_delegate_reports():
    from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
    from zelda_i.level9.hops import (
        level9_credits_chapter, level9_patra_chapter, level9_silver_arrows_chapter,
    )
    from zelda_i.level9.natural_path import NaturalSilverArrowsController

    silver = level9_silver_arrows_chapter(handoff=MEASURED_POST_L8_HANDOFF)[0][1]
    assert isinstance(silver, GuardedController)
    assert isinstance(silver.inner, NaturalSilverArrowsController)
    assert any(hop.spec_id == "level9_patra_16" for hop in silver.inner._hops)
    assert all(not isinstance(hop, GuardedController) for hop in silver.inner._hops)
    assert silver.max_frames == sum(hop.max_frames for hop in silver.inner._hops)
    assert not silver.success and not silver.failed
    silver.inner.notes.append("delegated")
    assert silver.notes == ["delegated"]
    assert silver.report()["shot_guard"]["overrides"] == 0
    assert isinstance(_shot_guarded(level9_patra_chapter()[0][1]), GuardedController)

    credits = {name: controller for name, controller, _ in level9_credits_chapter()}
    for name in ("level9_final_patra", "level9_ganon"):
        assert isinstance(_shot_guarded(credits[name]), GuardedController)
        assert not isinstance(_shot_guarded(credits[name]).inner, GuardedController)
    # The pause-menu controller must receive its environment through the
    # same runner, and Ganon's bound environment must survive the wrapper.
    env = object()
    credits["level9_ganon"].bind_env(env)
    assert _shot_guarded(credits["level9_ganon"]).inner.env is env


def test_probe_guard_scope_matches_spine_without_double_wrapping():
    from zelda_i.scratch.l9_probe import steps

    guarded = steps()
    for name, controller, _ in guarded:  # The prefix hops and the join.
        sid = name.split("_", 1)[0]
        if not (sid in ("s14b", "s14r") or 8 <= int(sid[1:]) <= 25):
            continue
        assert isinstance(_shot_guarded(controller), GuardedController)
        assert not isinstance(_shot_guarded(controller).inner, GuardedController)
    by_name = {name: controller for name, controller, _ in guarded}
    assert "s14b_level9_patra_16" in by_name
    assert isinstance(by_name["s27_level9_final_patra"], GuardedController)
    assert isinstance(_shot_guarded(by_name["s29_level9_ganon"]), GuardedController)
    assert all(
        not isinstance(_shot_guarded(controller), GuardedController)
        for _, controller, _ in steps(guard=False)
    )
    # A wrapped probe stage still refuses a wrong-room Ganon contract and
    # keeps the refusal's idle input even when a magic shot is imminent.
    ram = object_ram(room_ram(), type_id=0x58, x=100, state=0x10)

    class Env:
        def get_ram(self):
            return ram

    ganon = by_name["s29_level9_ganon"]
    ganon.bind_env(Env())
    assert ganon.step(read_snapshot(ram)).action == nes_idle_action()
    assert ganon.failed and not ganon.success


@pytest.mark.rom
def test_guarded_silver_chapter_rom_reports_guard_and_earned_arrows():
    """Development evidence from an assisted dungeon-entry pin, never Clean."""
    from retro_harness.env import make_env, read_state_bytes, state_path
    from retro_harness.segment_runner import configure_headless
    from zelda_i.assist import UnlimitedHealthAssist
    from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
    from zelda_i.level9.hops import level9_silver_arrows_chapter
    from zelda_i.paths import GAME, GAME_DIR
    from zelda_i.route.chain import run_controller_stage

    pin = state_path(GAME_DIR, GAME, "C11Evalo5_level9_natural_silver_arrows")
    if not pin.is_file() or not (pin.parent / "rom.nes").is_file():
        pytest.skip("Zelda I ROM or assisted L9 entry pin missing")
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist()
    try:
        env.reset()
        env.em.set_state(read_state_bytes(pin))  # No idle before the real pose.
        start = read_snapshot(env.get_ram())
        name, controller, cap = level9_silver_arrows_chapter(handoff=MEASURED_POST_L8_HANDOFF)[0]
        _, result = run_controller_stage(
            env, None, name=name, controller=controller, max_frames=cap, assist=assist,
        )
        end = read_snapshot(env.get_ram())
        report = result.report()
    finally:
        env.close()
    assert result.success, report
    assert (end.level, end.screen, end.mode, end.arrows) == (9, 0x10, 5, 2)
    assert (end.ring, end.sword, end.triforce, end.max_bombs) == (
        start.ring, start.sword, start.triforce, start.max_bombs,
    )
    assert report["controller"]["shot_guard"]["overrides"] > 0
    assert report["controller"]["controller_memory_writes"] == 0
    assert report["hearts"]["damage"] > 0 and report["hits"]
    assert assist.telemetry.progression_writes == assist.telemetry.capacity_writes == 0
    assert assist.telemetry.deaths == 0
