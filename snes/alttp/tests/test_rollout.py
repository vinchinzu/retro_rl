"""Offline cover for ``alttp.rollout``. The CastleMain walk is ``@pytest.mark.rom``."""

from __future__ import annotations

import numpy as np
import pytest

from alttp.primitives import (
    SPRITE_HIT_TIMER,
    SPRITE_HP,
    SPRITE_STATE,
    SPRITE_X_HIGH,
    SPRITE_X_LOW,
    SPRITE_Y_HIGH,
    SPRITE_Y_LOW,
)
from alttp.ram import (
    LINK_HP,
    LINK_X,
    LINK_Y,
    MODULE,
    ROOM_ID,
    SUBMODULE,
    read_snapshot,
    wram_index,
)
from alttp.rollout import (
    POLICY_GUARD_HORIZON,
    ROM_CHECKED,
    PolicyGuard,
    Rollout,
    hold,
    press,
    restores,
    strike_if_safe,
    strike_soldier,
)
from retro_harness.actions import snes_action

LEFT = press("LEFT")
RIGHT = press("RIGHT")
UP = press("UP")
DOWN = press("DOWN")
IDLE = press()


def _ram(
    *,
    hp: int = 0x18,
    x: int = 100,
    y: int = 200,
    room: int = 0x72,
    sub: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    # Play mode, so PolicyGuard treats the fixture as a controllable frame.
    ram[MODULE] = 0x07
    ram[SUBMODULE] = sub & 0xFF
    ram[wram_index(LINK_HP)] = hp & 0xFF
    ram[LINK_X] = x & 0xFF
    ram[LINK_X + 1] = (x >> 8) & 0xFF
    ram[LINK_Y] = y & 0xFF
    ram[LINK_Y + 1] = (y >> 8) & 0xFF
    ram[ROOM_ID] = room & 0xFF
    ram[ROOM_ID + 1] = (room >> 8) & 0xFF
    return ram


class _FakeEmulator:
    """Steps a per-action RAM tape and counts every save and restore."""

    def __init__(self, tapes: dict, base: np.ndarray) -> None:
        self.tapes = {tuple(key): value for key, value in tapes.items()}
        self.ram = base
        self.base = base
        self.mask = IDLE
        self.cursor = 0
        self.saves = 0
        self.restores = 0
        self.steps = 0
        self.raise_on_step = False

    def get_state(self) -> bytes:
        self.saves += 1
        return b"pin"

    def set_state(self, state: bytes) -> None:
        assert state == b"pin"
        self.restores += 1
        self.ram = self.base
        self.cursor = 0

    def set_button_mask(self, mask, _player: int = 0) -> None:
        assert isinstance(mask, np.ndarray) and mask.dtype == np.uint8, mask
        self.mask = tuple(int(value) for value in mask)

    def step(self) -> None:
        if self.raise_on_step:
            raise RuntimeError("core died mid-rollout")
        self.steps += 1
        tape = self.tapes.get(self.mask, ())
        if self.cursor < len(tape):
            self.ram = tape[self.cursor]
        self.cursor += 1


class _FakeEnv:
    def __init__(self, em: _FakeEmulator) -> None:
        self.em = em

    def get_ram(self) -> np.ndarray:
        return self.em.ram


def _env(tapes: dict, base: np.ndarray | None = None):
    base = _ram() if base is None else base
    em = _FakeEmulator(tapes, base)
    return _FakeEnv(em), em


def test_press_matches_snes_action_and_rejects_a_typo() -> None:
    assert press("down") == tuple(int(value) for value in snes_action("DOWN"))
    assert press() == (0,) * len(press())
    assert len(press()) == 12
    assert press("LEFT", "Y") != press("LEFT")
    with pytest.raises(ValueError, match="unknown SNES button"):
        press("LEFTT")


def test_hold_stands_on_none_and_on_stand() -> None:
    assert hold(None, 3) == (IDLE, IDLE, IDLE)
    assert hold("STAND", 3) == (IDLE, IDLE, IDLE)
    assert hold("LEFT", 3) == (LEFT, LEFT, LEFT)
    assert hold("LEFT", 0) == ()


def test_fan_restores_once_per_plan_and_at_the_end() -> None:
    env, em = _env({})
    before = restores()
    Rollout(env).fan({"a": hold("LEFT", 3), "b": hold("RIGHT", 3)})
    assert em.saves == 1
    assert em.restores == 3
    assert restores() == before + 3
    assert env.get_ram() is em.base


def test_fan_restores_when_the_core_raises() -> None:
    env, em = _env({})
    em.raise_on_step = True
    before = restores()
    with pytest.raises(RuntimeError, match="core died"):
        Rollout(env).fan({"a": hold("LEFT", 3)})
    assert em.restores >= 1
    assert restores() > before
    assert env.get_ram() is em.base


def test_fan_of_nothing_never_touches_the_emulator() -> None:
    env, em = _env({})
    before = restores()
    assert Rollout(env).fan({}) == ()
    assert (em.saves, em.restores, em.steps) == (0, 0, 0)
    assert restores() == before


def test_fan_preserves_plan_order() -> None:
    env, _em = _env({})
    labels = [
        outcome.label
        for outcome in Rollout(env).fan(
            {"UP": hold("UP", 1), "DOWN": hold("DOWN", 1), "STAND": hold(None, 1)}
        )
    ]
    assert labels == ["UP", "DOWN", "STAND"]


def test_frames_pads_a_short_plan_by_holding_its_last_frame() -> None:
    env, em = _env({})
    Rollout(env).fan({"a": (LEFT,)}, frames=5)
    assert em.steps == 5
    assert em.mask == LEFT


def test_frames_truncates_a_long_plan() -> None:
    env, em = _env({})
    out = Rollout(env).fan({"a": hold("LEFT", 40)}, frames=6)[0]
    assert (em.steps, out.frames) == (6, 6)


def test_an_empty_plan_padded_is_idle() -> None:
    env, em = _env({})
    Rollout(env).fan({"a": ()}, frames=3)
    assert (em.steps, em.mask) == (3, IDLE)


def test_contact_frame_is_the_first_hp_drop() -> None:
    tape = [_ram(hp=value) for value in (0x18, 0x18, 0x17, 0x10)]
    env, _em = _env({LEFT: tape})
    out = Rollout(env).fan({"left": hold("LEFT", 4)})[0]
    assert out.contact_frame == 3
    assert out.hp_before == 0x18 and out.hp_after == 0x10
    assert out.report()["contact_frame"] == 3


def test_a_later_drop_does_not_move_the_first_contact_frame() -> None:
    tape = [_ram(hp=value) for value in (0x20, 0x1C)]
    env, _em = _env({LEFT: tape}, base=_ram(hp=0x18))
    out = Rollout(env).fan({"left": hold("LEFT", 2)})[0]
    assert out.contact_frame == 2
    assert out.hp_after == 0x1C


def test_healing_is_not_contact() -> None:
    env, _em = _env({LEFT: [_ram(hp=0x20)]}, base=_ram(hp=0x18))
    out = Rollout(env).fan({"left": hold("LEFT", 1)})[0]
    assert out.contact_frame is None
    assert out.hp_after == 0x20


def test_moved_is_manhattan_from_the_fan_start() -> None:
    env, _em = _env({RIGHT: [_ram(x=104, y=208)]})
    out = Rollout(env).fan({"r": hold("RIGHT", 1)})[0]
    assert out.moved == 4 + 8
    assert out.report()["moved"] == 12
    assert out.report()["xy"] == [104, 208]


def test_room_change_is_the_room_word() -> None:
    gone = _ram(room=0x01, x=50, y=90)
    env, _em = _env({UP: [gone]}, base=_ram(room=0x72, x=100, y=200))
    out = Rollout(env).fan({"up": hold("UP", 1)})[0]
    assert out.room_changed is True
    assert (out.room_before, out.room_after) == (0x72, 0x01)
    assert out.moved == abs(50 - 100) + abs(90 - 200)
    assert out.report()["room_changed"] is True


def test_a_high_byte_change_is_still_a_room_change() -> None:
    env, _em = _env({UP: [_ram(room=0x172)]}, base=_ram(room=0x72))
    out = Rollout(env).fan({"up": hold("UP", 1)})[0]
    assert out.room_changed is True
    assert out.room_after == 0x172


def test_plans_do_not_see_each_others_hp_drop() -> None:
    env, em = _env(
        {LEFT: [_ram(hp=0x10, x=110)], RIGHT: [_ram(hp=0x18, x=130, y=200)]}
    )
    into, away = Rollout(env).fan(
        {"into": hold("LEFT", 1), "away": hold("RIGHT", 1)}
    )
    assert into.contact_frame == 1 and into.hp_after == 0x10
    assert away.contact_frame is None and away.hp_after == 0x18
    assert away.moved == 30
    assert env.get_ram() is em.base
    assert int(env.get_ram()[wram_index(LINK_HP)]) == 0x18


def test_no_motion_reports_zero_distance_and_the_same_room() -> None:
    env, _em = _env({LEFT: [_ram()]})
    out = Rollout(env).fan({"left": hold("LEFT", 1)})[0]
    assert out.contact_frame is None
    assert out.moved == 0
    assert out.room_changed is False
    assert out.room_before == out.room_after == 0x72


def test_report_counts_rollouts_not_the_final_restore() -> None:
    env, _em = _env({})
    rollout = Rollout(env)
    rollout.fan({"a": hold("LEFT", 4), "b": hold("RIGHT", 4)})
    assert rollout.report() == {"rollouts": 2, "frames_rolled": 8}


class _Hold:
    """A hand controller that repeats one direction and counts its own steps."""

    max_frames = 50
    success = False
    failed = False

    def __init__(self, direction: str = "RIGHT") -> None:
        self.direction = direction
        self.steps = 0

    def step(self, _snap: object):
        from retro_harness.input_script import FrameAction

        self.steps += 1
        return FrameAction(list(press(self.direction)), f"walk_{self.direction.lower()}")


def test_policy_guard_horizon_is_the_snes_budget() -> None:
    assert POLICY_GUARD_HORIZON == 25


def test_policy_guard_follows_the_inner_plan_when_the_rom_says_it_is_clean() -> None:
    from alttp.ram import read_snapshot

    env, em = _env({RIGHT: [_ram(x=102), _ram(x=104)]})
    guard = PolicyGuard(_Hold("RIGHT"), horizon=6, detours=(4,), directions=("UP",), reach=None)
    guard.bind_env(env)
    act = guard.step(read_snapshot(env.get_ram()))
    assert tuple(act.action) == RIGHT and guard.inner.steps == 1
    assert guard.detours_taken == 0
    assert em.ram is em.base


def test_policy_guard_detours_without_stepping_the_inner_when_its_plan_is_hit() -> None:
    from alttp.ram import read_snapshot

    env, em = _env(
        {RIGHT: [_ram(x=102), _ram(hp=0x14)], UP: [_ram(y=98)] * 6}
    )
    guard = PolicyGuard(_Hold("RIGHT"), horizon=6, detours=(4,), directions=("UP",), reach=None)
    guard.bind_env(env)
    act = guard.step(read_snapshot(env.get_ram()))
    assert tuple(act.action) == UP and guard.inner.steps == 0
    assert len(guard._queue) == 3 and guard.detours_taken == 1
    assert ROM_CHECKED in act.reason
    assert em.ram is em.base


def test_policy_guard_queued_detour_frames_stay_rom_checked() -> None:
    from alttp.ram import read_snapshot

    env, _em = _env(
        {RIGHT: [_ram(x=102), _ram(hp=0x14)], UP: [_ram(y=98)] * 6}
    )
    guard = PolicyGuard(_Hold("RIGHT"), horizon=6, detours=(4,), directions=("UP",), reach=None)
    guard.bind_env(env)
    snap = read_snapshot(env.get_ram())
    first = guard.step(snap)
    queued = guard.step(snap)
    assert tuple(first.action) == UP and guard.inner.steps == 0
    assert tuple(queued.action) == UP and ROM_CHECKED in queued.reason
    assert guard.checks == 1


def test_policy_guard_skips_a_detour_that_leaves_the_room() -> None:
    from alttp.ram import read_snapshot

    env, em = _env(
        {
            DOWN: [_ram(hp=0x18), _ram(hp=0x14)],
            LEFT: [_ram(room=0x01)] * 6,
            RIGHT: [_ram(x=120)] * 6,
        }
    )
    guard = PolicyGuard(
        _Hold("DOWN"),
        horizon=4,
        detours=(2,),
        directions=("LEFT", "RIGHT"),
        reach=None,
    )
    guard.bind_env(env)
    act = guard.step(read_snapshot(env.get_ram()))
    assert tuple(act.action) == RIGHT
    assert guard.detours_taken == 1
    assert em.ram is em.base


def test_policy_guard_stays_on_the_plan_when_every_detour_leaves_the_room() -> None:
    from alttp.ram import read_snapshot

    env, _em = _env(
        {
            DOWN: [_ram(hp=0x14)],
            LEFT: [_ram(room=0x01)] * 4,
            RIGHT: [_ram(sub=14)] * 4,
        }
    )
    guard = PolicyGuard(
        _Hold("DOWN"),
        horizon=3,
        detours=(2,),
        directions=("LEFT", "RIGHT"),
        reach=None,
    )
    guard.bind_env(env)
    act = guard.step(read_snapshot(env.get_ram()))
    assert tuple(act.action) == DOWN
    assert guard.inner.steps == 1
    assert guard.detours_taken == 0


def test_policy_guard_does_not_detour_the_inners_own_exit() -> None:
    from alttp.ram import read_snapshot

    env, _em = _env({UP: [_ram(room=0x72), _ram(room=0x01)]})
    guard = PolicyGuard(
        _Hold("UP"), horizon=4, detours=(2,), directions=("LEFT",), reach=None
    )
    guard.bind_env(env)
    act = guard.step(read_snapshot(env.get_ram()))
    assert tuple(act.action) == UP
    assert guard.detours_taken == 0
    assert guard.inner.steps == 1


def test_policy_guard_does_not_commit_frames_past_the_horizon() -> None:
    from alttp.ram import read_snapshot

    env, _em = _env({DOWN: [_ram(hp=0x14)], RIGHT: [_ram(x=140)] * 8})
    guard = PolicyGuard(
        _Hold("DOWN"), horizon=4, detours=(100,), directions=("RIGHT",), reach=None
    )
    guard.bind_env(env)
    guard.step(read_snapshot(env.get_ram()))
    assert guard.detours_taken == 1
    assert len(guard._queue) == 3


def _stamp(
    ram: np.ndarray,
    *,
    kind: int,
    x: int,
    y: int,
    slot: int = 0,
    floor: int = 0,
    state: int = 0x09,
) -> np.ndarray:
    from alttp.primitives import (
        SPRITE_FLOOR,
        SPRITE_STATE,
        SPRITE_TYPE,
        SPRITE_X_HIGH,
        SPRITE_X_LOW,
        SPRITE_Y_HIGH,
        SPRITE_Y_LOW,
    )

    ram[SPRITE_STATE + slot] = state & 0xFF
    ram[SPRITE_TYPE + slot] = kind & 0xFF
    ram[SPRITE_FLOOR + slot] = floor & 0xFF
    ram[SPRITE_X_LOW + slot] = x & 0xFF
    ram[SPRITE_X_HIGH + slot] = (x >> 8) & 0xFF
    ram[SPRITE_Y_LOW + slot] = y & 0xFF
    ram[SPRITE_Y_HIGH + slot] = (y >> 8) & 0xFF
    return ram


def test_hazard_gate_uses_distance_and_eta_and_skips_pickups() -> None:
    from alttp.rollout import HAZARD_REACH, hazard_in_reach

    empty, _seen = hazard_in_reach(_ram())
    assert empty is False
    close = _stamp(_ram(), kind=0x41, x=110, y=210)
    assert hazard_in_reach(close)[0] is True
    far = _stamp(_ram(), kind=0x41, x=100, y=200 + HAZARD_REACH + 10)
    assert hazard_in_reach(far)[0] is False
    # 10 px/frame toward Link from 80 px out arrives inside a 25-frame horizon.
    far_prev = {0: (100, 200 + HAZARD_REACH + 10 + 10, 0x41)}
    assert hazard_in_reach(far, far_prev, horizon=25)[0] is True
    heart = _stamp(_ram(), kind=0xD8, x=100, y=200)
    uncle = _stamp(_ram(), kind=0x73, x=108, y=200)
    other_floor = _stamp(_ram(), kind=0x41, x=110, y=210, floor=1)
    arrow = _stamp(_ram(), kind=0x1B, x=120, y=200)
    assert hazard_in_reach(heart)[0] is False
    assert hazard_in_reach(uncle)[0] is False
    assert hazard_in_reach(other_floor)[0] is False
    assert hazard_in_reach(arrow)[0] is True


def test_policy_guard_stays_quiet_when_nothing_is_in_reach() -> None:
    from alttp.ram import read_snapshot

    env, em = _env({})
    guard = PolicyGuard(_Hold("DOWN"), horizon=4, detours=(2,), directions=("RIGHT",))
    guard.bind_env(env)
    act = guard.step(read_snapshot(env.get_ram()))
    assert tuple(act.action) == DOWN
    assert guard.quiet == 1 and guard.checks == 0 and em.steps == 0
    report = guard.report()["policy_guard"]
    assert report["fraction_rolled"] == 0.0
    assert report["play_frames"] == 1


def test_policy_guard_rolls_when_a_soldier_is_inside_reach() -> None:
    from alttp.ram import read_snapshot

    base = _stamp(_ram(), kind=0x41, x=110, y=210)
    env, em = _env({RIGHT: [_ram(x=102), _ram(x=104)]}, base=base)
    guard = PolicyGuard(_Hold("RIGHT"), horizon=4, detours=(2,), directions=("UP",))
    guard.bind_env(env)
    act = guard.step(read_snapshot(env.get_ram()))
    assert tuple(act.action) == RIGHT
    assert guard.checks == 1 and guard.quiet == 0
    assert em.steps == 4
    assert guard.report()["policy_guard"]["fraction_rolled"] == 1.0


def test_policy_guard_unbound_follows_the_inner_without_rolling() -> None:
    from alttp.ram import read_snapshot

    env, em = _env({})
    guard = PolicyGuard(_Hold("LEFT"))
    act = guard.step(read_snapshot(env.get_ram()))
    assert tuple(act.action) == LEFT
    assert guard.checks == 0 and em.steps == 0


def _soldier(ram: np.ndarray, *, x: int, y: int, hp: int, timer: int = 0) -> np.ndarray:
    stamped = ram.copy()
    stamped[SPRITE_STATE] = 0x09
    stamped[SPRITE_HP] = hp & 0xFF
    stamped[SPRITE_HIT_TIMER] = timer & 0xFF
    stamped[SPRITE_X_LOW] = x & 0xFF
    stamped[SPRITE_X_HIGH] = (x >> 8) & 0xFF
    stamped[SPRITE_Y_LOW] = y & 0xFF
    stamped[SPRITE_Y_HIGH] = (y >> 8) & 0xFF
    return stamped


def _swing_tapes(frames: list[np.ndarray]) -> dict:
    return {press("B"): frames, press(): frames}


def test_strike_commits_when_the_target_hp_drops_and_link_is_safe() -> None:
    base = _soldier(_ram(), x=100, y=224, hp=6)
    later = _soldier(_ram(), x=100, y=240, hp=4)
    env, em = _env(_swing_tapes([later] * 13), base)
    out = strike_if_safe(env, 0)
    assert out is not None and len(out) == 13
    assert all(ROM_CHECKED in act.reason for act in out)
    assert env.get_ram() is em.base


def test_strike_refuses_when_link_is_hit() -> None:
    base = _soldier(_ram(), x=100, y=224, hp=6)
    hurt = _soldier(_ram(hp=20), x=100, y=230, hp=6)
    env, em = _env(_swing_tapes([hurt] * 13), base)
    assert strike_if_safe(env, 0) is None
    assert env.get_ram() is em.base


def test_strike_commits_a_push_and_refuses_a_whiff_or_hitstun() -> None:
    base = _soldier(_ram(), x=100, y=224, hp=6)
    # A shove sets bit 7. Distance alone is the soldier walking away.
    shoved = _soldier(_ram(), x=100, y=250, hp=6, timer=0x94)
    env, _em = _env(_swing_tapes([shoved] * 13), base)
    assert strike_if_safe(env, 0) is not None
    walked = _soldier(_ram(), x=100, y=250, hp=6)
    env, em = _env(_swing_tapes([walked] * 13), base)
    assert strike_if_safe(env, 0) is None
    assert em.steps == 13
    still = _soldier(_ram(), x=100, y=224, hp=6)
    env, em = _env(_swing_tapes([still] * 13), still)
    assert strike_if_safe(env, 0) is None
    assert em.steps == 13
    stun = _soldier(_ram(), x=100, y=224, hp=6, timer=0x9D)
    env, em = _env({}, stun)
    assert strike_if_safe(env, 0) is None
    assert em.steps == 0


def test_policy_guard_deep_copy_does_not_share_the_inner() -> None:
    import copy

    guard = PolicyGuard(_Hold("LEFT"))
    twin = copy.deepcopy(guard)
    assert isinstance(twin.inner, _Hold) and twin.inner is not guard.inner
    assert twin.max_frames == 50 and twin.horizon == POLICY_GUARD_HORIZON


def test_strike_soldier_idles_through_hitstun_instead_of_stepping_in() -> None:
    stun = _soldier(_ram(), x=100, y=224, hp=6, timer=0x9D)
    env, em = _env({}, stun)
    out = strike_soldier(env, 0, frames=3)
    assert out["swings"] == 0 and out["dead"] is False
    assert em.steps == 3 and em.mask == IDLE


def test_walk_records_the_room_step_reverse_and_restores() -> None:
    from alttp.dungeon.nav import COMMIT_FRAMES

    env, em = _env({RIGHT: [_ram(x=102)]})
    plan, outcome, reached = Rollout(env).walk(
        [(102, 200), (100, 200)], frames=1 + COMMIT_FRAMES + 1, tol=0
    )
    assert plan == (RIGHT,) + (IDLE,) * COMMIT_FRAMES + (LEFT,)
    assert reached is False
    assert outcome.snap.link_x == 102
    assert em.ram is em.base


def test_walk_replay_ends_on_the_outcome_xy() -> None:
    tapes = {RIGHT: [_ram(x=100 + 2 * (i + 1)) for i in range(12)]}
    env, em = _env(tapes)
    plan, outcome, reached = Rollout(env).walk([(120, 200)], frames=40, tol=2)
    assert reached and plan == (RIGHT,) * 9
    assert outcome.snap.link_x == 118
    assert em.ram is em.base
    for frame in plan:
        em.set_button_mask(np.asarray(frame, dtype=np.uint8), 0)
        em.step()
    snap = read_snapshot(env.get_ram())
    assert (snap.link_x, snap.link_y) == (outcome.snap.link_x, outcome.snap.link_y)


def test_walk_already_there_does_not_step() -> None:
    env, em = _env({})
    plan, outcome, reached = Rollout(env).walk([(100, 200)], tol=2)
    assert plan == () and reached
    assert (outcome.snap.link_x, outcome.snap.link_y) == (100, 200)
    assert em.steps == 0


@pytest.mark.rom
def test_walk_on_castle_main_replays_to_the_same_xy() -> None:
    from alttp.paths import INTEGRATION_DIR
    from alttp.startup import build_boot_env

    if not (INTEGRATION_DIR / "rom.sfc").is_file():
        pytest.skip("ALTTP ROM missing")
    if not (INTEGRATION_DIR / "CastleMain.state").is_file():
        pytest.skip("CastleMain.state missing")
    env = build_boot_env("CastleMain")
    try:
        env.reset()
        before = read_snapshot(np.asarray(env.get_ram(), dtype=np.uint8))
        goal = (int(before.link_x), int(before.link_y) - 32)
        plan, outcome, reached = Rollout(env).walk([goal], frames=80, tol=2)
        assert reached, outcome.report()
        for frame in plan:
            env.em.set_button_mask(np.asarray(frame, dtype=np.uint8), 0)
            env.em.step()
        after = read_snapshot(np.asarray(env.get_ram(), dtype=np.uint8))
        assert (after.link_x, after.link_y) == (
            outcome.snap.link_x,
            outcome.snap.link_y,
        )
        assert (after.link_x, after.link_y) != (before.link_x, before.link_y)
    finally:
        env.close()


@pytest.mark.rom
def test_castle_soldier_dies_with_no_damage_on_ten_of_twelve_offsets() -> None:
    from alttp.paths import INTEGRATION_DIR
    from alttp.startup import build_boot_env

    if not (INTEGRATION_DIR / "rom.sfc").is_file():
        pytest.skip("ALTTP ROM missing")
    if not (INTEGRATION_DIR / "CastleB1Guard.state").is_file():
        pytest.skip("CastleB1Guard.state missing")
    env = build_boot_env("CastleB1Guard")
    try:
        env.reset()
        root = bytes(env.em.get_state())
        kills = 0
        rows = []
        for offset in range(12):
            env.em.set_state(root)
            for _ in range(offset):
                env.em.set_button_mask(np.asarray(press(), dtype=np.uint8), 0)
                env.em.step()
            result = strike_soldier(env, 0)
            rows.append((offset, result))
            if result["dead"] and result["hp"] == 24 and int(result["swings"]) >= 3:
                kills += 1
        assert kills >= 10, rows
    finally:
        env.close()
