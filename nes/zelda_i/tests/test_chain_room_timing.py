"""Integration tests: opt-in RoomTimer seam on chain stage runner (no ROM)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from zelda_i.chain import controller_stage_done, run_controller_stage
from zelda_i.ram import PLAY_MODE
from zelda_i.room_timer import RoomTimer, bottleneck_visits


class _HopEnv:
    """Synthetic env that scrolls 0x37 → 0x38 after a few settled frames."""

    def __init__(self) -> None:
        self.step_n = 0
        self._ram = bytearray(0x800)

    def get_ram(self):
        return bytes(self._ram)

    def step(self, _action):
        self.step_n += 1
        # Frames 1–3: settled on 0x37; 4–5: scroll; 6+: settled 0x38.
        if self.step_n <= 3:
            mode, screen, next_screen = PLAY_MODE, 0x37, 0
        elif self.step_n <= 5:
            mode, screen, next_screen = 6, 0x37, 0x38
        else:
            mode, screen, next_screen = PLAY_MODE, 0x38, 0
        # ADDR_* values are looked up via ram.read_snapshot — write full snapshot
        # through a FakeController that doesn't use RAM; we only need get_ram for
        # the timer observe path. Patch via monkey: use real addresses.
        from zelda_i import ram as ram_mod

        self._ram[ram_mod.ADDR_MODE] = mode
        self._ram[ram_mod.ADDR_LEVEL] = 0
        self._ram[ram_mod.ADDR_SCREEN] = screen
        self._ram[ram_mod.ADDR_NEXT_SCREEN] = next_screen
        self._ram[ram_mod.ADDR_SWORD] = 1
        self._ram[ram_mod.ADDR_TRIFORCE] = 1
        self._ram[ram_mod.ADDR_HEALTH] = 0x33
        return np.zeros((2, 2, 3), dtype=np.uint8), 0.0, False, False, {}


class _DoneOnScreenController:
    """Succeeds once screen is 0x38 (read from the env RAM we just stepped)."""

    def __init__(self, env: _HopEnv) -> None:
        self.env = env
        self.success = False
        self.phase = SimpleNamespace(name="HOP")
        self.frames = 0

    def step(self, snap):
        self.frames += 1
        # Controller sees pre-step snap; stage steps env afterward.
        if snap.screen == 0x38 and snap.mode == PLAY_MODE:
            self.success = True
            self.phase = SimpleNamespace(name="DONE")
        return SimpleNamespace(action=0)

    def report(self) -> dict:
        return {"success": self.success, "frames": self.frames}


def test_run_controller_stage_without_timer_unchanged() -> None:
    env = _HopEnv()
    # Seed RAM for first controller.step before any env.step.
    from zelda_i import ram as ram_mod

    env._ram[ram_mod.ADDR_MODE] = PLAY_MODE
    env._ram[ram_mod.ADDR_SCREEN] = 0x37
    env._ram[ram_mod.ADDR_SWORD] = 1
    env._ram[ram_mod.ADDR_TRIFORCE] = 1
    env._ram[ram_mod.ADDR_HEALTH] = 0x33

    controller = _DoneOnScreenController(env)
    obs, result = run_controller_stage(
        env, None, name="hop", controller=controller, max_frames=20
    )
    assert obs.shape == (2, 2, 3)
    assert result.success is True
    assert result.frames >= 6
    assert result.frame_base == 0
    assert result.end_frame == result.frames


def test_run_controller_stage_opt_in_timer_records_hop() -> None:
    env = _HopEnv()
    from zelda_i import ram as ram_mod

    env._ram[ram_mod.ADDR_MODE] = PLAY_MODE
    env._ram[ram_mod.ADDR_LEVEL] = 0
    env._ram[ram_mod.ADDR_SCREEN] = 0x37
    env._ram[ram_mod.ADDR_SWORD] = 1
    env._ram[ram_mod.ADDR_TRIFORCE] = 1
    env._ram[ram_mod.ADDR_HEALTH] = 0x33

    timer = RoomTimer()
    # Anchor open visit at frame_base before stage (mirrors runners).
    from zelda_i.ram import read_snapshot

    timer.observe(read_snapshot(env.get_ram()), frame=10)
    controller = _DoneOnScreenController(env)
    _obs, result = run_controller_stage(
        env,
        None,
        name="level2_prefix",
        controller=controller,
        max_frames=20,
        room_timer=timer,
        frame_base=10,
    )
    assert result.success is True
    assert result.frame_base == 10
    assert result.end_frame == 10 + result.frames
    assert len(timer.visits) == 1
    visit = timer.visits[0]
    assert visit.screen == 0x37
    assert visit.dest_screen == 0x38
    assert visit.entry_frame == 10
    assert visit.leave_frame == 10 + 4  # first scroll at local frame 4
    assert visit.exit_frame == 10 + 6
    assert visit.dwell_frames == 4
    assert visit.transition_frames == 2
    assert visit.location_frames == 6


def test_bottleneck_visits_ranks_by_location_frames() -> None:
    from zelda_i.room_timer import TimingSnapshot, run_offline

    samples = [
        TimingSnapshot(frame=0, mode=5, level=0, screen=0x37),
        TimingSnapshot(frame=5, mode=6, level=0, screen=0x37, next_screen=0x38),
        TimingSnapshot(frame=20, mode=5, level=0, screen=0x38),  # slow hop
        TimingSnapshot(frame=22, mode=6, level=0, screen=0x38, next_screen=0x48),
        TimingSnapshot(frame=30, mode=5, level=0, screen=0x48),  # faster
    ]
    report = run_offline(samples)
    top = bottleneck_visits(report["visits"], top_n=1)
    assert len(top) == 1
    assert top[0]["screen"] == 0x37
    assert top[0]["dest_screen"] == 0x38
    assert top[0]["location_frames"] == 20


def test_controller_stage_done_accepts_string_phase() -> None:
    """L3 west-key / north-chain controllers use phase str, not DungeonPhase."""
    door = SimpleNamespace(success=False, phase="door")
    assert not controller_stage_done(door)
    done = SimpleNamespace(success=True, phase="done")
    assert controller_stage_done(done)
    failed = SimpleNamespace(success=False, phase="failed")
    assert controller_stage_done(failed)
    enum_failed = SimpleNamespace(success=False, phase=SimpleNamespace(name="FAILED"))
    assert controller_stage_done(enum_failed)
    flagged = SimpleNamespace(success=False, failed=True, phase="door")
    assert controller_stage_done(flagged)


def test_run_controller_stage_string_phase_does_not_crash() -> None:
    """Spine --through level3 west_key used to AttributeError on phase.name."""
    env = _HopEnv()
    from zelda_i import ram as ram_mod

    env._ram[ram_mod.ADDR_MODE] = PLAY_MODE
    env._ram[ram_mod.ADDR_SCREEN] = 0x7C

    class _StringPhase:
        def __init__(self) -> None:
            self.success = False
            self.phase = "door"
            self.frames = 0

        def step(self, _snap):
            self.frames += 1
            if self.frames >= 2:
                self.success = True
                self.phase = "done"
            return SimpleNamespace(action=0)

        def report(self) -> dict:
            return {"success": self.success, "phase": self.phase, "frames": self.frames}

    controller = _StringPhase()
    _obs, result = run_controller_stage(
        env, None, name="west_key", controller=controller, max_frames=20
    )
    assert result.success is True
    assert result.frames == 2
    assert controller.phase == "done"
