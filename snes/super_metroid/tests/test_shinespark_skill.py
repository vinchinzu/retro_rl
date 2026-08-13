"""Unit tests for shinespark skill helpers (no emulator)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from super_metroid.ram import (
    ADDR_SHINESPARK_TIMER,
    ADDR_SPEED_COUNTER,
    ADDR_SPEED_FLAG,
    parse_state,
)
from super_metroid.routes.skills import shinespark as spark


class _FakeEnv:
    def __init__(self, ram: np.ndarray) -> None:
        self._ram = ram

    def get_ram(self) -> np.ndarray:
        return self._ram


class _FakeSession:
    def __init__(
        self,
        state: Any,
        *,
        env: _FakeEnv | None = None,
        transitions: list[Any] | None = None,
    ) -> None:
        self.state = state
        self.frame = int(getattr(state, "frame", 0))
        self.env = env
        self._transitions = list(transitions or [])
        self._step_i = 0
        self.reasons: list[str] = []

    def step(self, action: np.ndarray, reason: str = "") -> Any:
        del action
        self.frame += 1
        self.reasons.append(reason)
        if self._step_i < len(self._transitions):
            self.state = self._transitions[self._step_i]
            self._step_i += 1
        else:
            self.state = replace(self.state, frame=self.frame)
        return self.state


def _state(**kwargs: Any) -> Any:
    base = parse_state(np.zeros(0x10000, dtype=np.uint8))
    return replace(base, **kwargs)


def _ram_with_spark(*, echoes: int = 0, timer: int = 0, flag: int = 0) -> np.ndarray:
    ram = np.zeros(0x10000, dtype=np.uint8)
    # $0B3E lo=anim hi=echoes
    word = (echoes << 8) | 0x01
    ram[ADDR_SPEED_COUNTER] = word & 0xFF
    ram[ADDR_SPEED_COUNTER + 1] = (word >> 8) & 0xFF
    ram[ADDR_SHINESPARK_TIMER] = timer & 0xFF
    ram[ADDR_SHINESPARK_TIMER + 1] = (timer >> 8) & 0xFF
    ram[ADDR_SPEED_FLAG] = flag & 0xFF
    ram[ADDR_SPEED_FLAG + 1] = (flag >> 8) & 0xFF
    return ram


def test_constants_importable() -> None:
    assert spark.ECHOES_FULL == 4
    assert spark.TYPICAL_ARM_TIMER == 179
    assert spark.TYPICAL_CHARGE_FRAMES == 90
    assert 201 in spark.SPARK_POSES
    assert 9 in spark.STORE_OK_POSES
    assert 25 in spark.STORE_WIPE_POSES
    assert spark.is_spark_pose(201)
    assert not spark.is_spark_pose(9)
    assert spark.store_pose_ok(9)
    assert not spark.store_pose_ok(25)
    assert spark.NTSC_MAGIC_DASH_FRAMES == (25, 50, 70, 85)
    assert spark.PAL_MAGIC_DASH_FRAMES == (20, 40, 60, 70)
    assert spark.NTSC_SHORT_CHARGE_FRAMES == 86
    assert spark.PAL_SHORT_CHARGE_FRAMES == 71


def test_short_charge_plan_ntsc_simple() -> None:
    plan = spark.short_charge_plan("NTSC", stutter=False, store_on_last=False)
    assert len(plan) == 86
    # Non-magic: forward only
    assert plan[0] == ("RIGHT",)
    assert plan[24] == ("RIGHT",)
    assert plan[26] == ("RIGHT",)
    # Magic dash frames
    for f in (25, 50, 70, 85):
        assert plan[f] == ("RIGHT", "B"), f"magic frame {f}: {plan[f]}"
    # store_on_last adds DOWN only on final magic
    plan_s = spark.short_charge_plan("NTSC", store_on_last=True)
    assert plan_s[85] == ("RIGHT", "B", "DOWN")
    assert plan_s[70] == ("RIGHT", "B")


def test_short_charge_plan_ntsc_stutter() -> None:
    plan = spark.short_charge_plan("NTSC", stutter=True)
    assert len(plan) == 86
    # 3 F, rel, 4 F, … then F+B×3, B-only, then magic 25
    assert plan[0] == ("RIGHT",)
    assert plan[1] == ("RIGHT",)
    assert plan[2] == ("RIGHT",)
    assert plan[3] == ()  # release
    assert plan[4] == ("RIGHT",)
    # frames 21–23: F+B; 24: B only
    assert plan[21] == ("RIGHT", "B")
    assert plan[22] == ("RIGHT", "B")
    assert plan[23] == ("RIGHT", "B")
    assert plan[24] == ("B",)
    assert plan[25] == ("RIGHT", "B")
    # mid non-magic after first boost tick
    assert plan[30] == ("RIGHT",)
    assert plan[50] == ("RIGHT", "B")
    fwd = spark.stutter_forward_mask("NTSC")
    assert len(fwd) == 25
    assert fwd.count(False) == 5  # four mid releases + final F release


def test_short_charge_plan_pal() -> None:
    plan = spark.short_charge_plan("PAL", stutter=False)
    assert len(plan) == 71
    for f in (20, 40, 60, 70):
        assert "B" in plan[f]
    plan_st = spark.short_charge_plan("PAL", stutter=True, store_on_last=True)
    assert len(plan_st) == 71
    assert plan_st[70] == ("RIGHT", "B", "DOWN")
    assert len(spark.stutter_forward_mask("PAL")) == 20


def test_short_charge_until_boost_presses_magic_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake session: echoes hit 4 after the 4th magic dash frame executes."""
    session = _FakeSession(
        _state(pose=9, velocity_y=0, speed_counter=0, samus_x=100, frame=0),
    )
    pressed: list[tuple[int, tuple[str, ...]]] = []
    magic_hits = 0

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        nonlocal magic_hits
        for _ in range(frames):
            pressed.append((sess.frame, buttons))
            sess.step(np.zeros(12, dtype=np.int8), reason)
            if "B" in buttons:
                magic_hits += 1
                # After 4 dash presses, full echoes
                if magic_hits >= 4:
                    sess.state = replace(
                        sess.state,
                        speed_counter=4,
                        pose=9,
                        velocity_y=0,
                        samus_x=sess.state.samus_x + 1,
                        frame=sess.frame,
                    )
                else:
                    sess.state = replace(
                        sess.state,
                        speed_counter=magic_hits,
                        samus_x=sess.state.samus_x + 1,
                        frame=sess.frame,
                    )
            else:
                sess.state = replace(
                    sess.state,
                    samus_x=sess.state.samus_x + 1,
                    frame=sess.frame,
                )
        return sess.state

    monkeypatch.setattr(spark, "hold", _hold)
    out = spark.short_charge_until_boost(session, "RIGHT", stutter=False)
    assert out["ok"] is True
    assert out["mode"] == "short"
    assert magic_hits == 4
    # Only magic frames press B (no stutter prefix dash)
    b_frames = [f for f, btns in pressed if "B" in btns]
    assert b_frames == [25, 50, 70, 85]
    assert out["dash_frames"] == [25, 50, 70, 85]


def test_charge_until_boost_mode_short_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def _fake_short(session: Any, direction: str = "RIGHT", **kwargs: Any) -> dict:
        called["direction"] = direction
        called["kwargs"] = kwargs
        return {"ok": True, "frames": 86, "mode": "short"}

    monkeypatch.setattr(spark, "short_charge_until_boost", _fake_short)
    session = _FakeSession(_state(pose=9, velocity_y=0, speed_counter=0, frame=0))
    out = spark.charge_until_boost(session, "LEFT", mode="stutter")
    assert out["ok"] is True
    assert called["direction"] == "LEFT"
    assert called["kwargs"]["stutter"] is True


def test_read_spark_wram() -> None:
    env = _FakeEnv(_ram_with_spark(echoes=4, timer=179, flag=1))
    w = spark.read_spark_wram(env)
    assert w["speed_echoes"] == 4
    assert w["spark_timer"] == 179
    assert w["speed_flag"] == 1
    assert w["speed_anim"] == 1
    assert w["speed_counter_word"] == (4 << 8) | 1


def test_charge_until_boost_stops_on_echoes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sequence: building → full boost grounded
    states = [
        _state(pose=9, velocity_y=0, speed_counter=1, frame=1),
        _state(pose=9, velocity_y=0, speed_counter=3, frame=2),
        _state(
            pose=9,
            velocity_y=0,
            speed_counter=4,
            frame=3,
            samus_x=100,
            samus_y=50,
        ),
    ]
    session = _FakeSession(
        _state(pose=9, velocity_y=0, speed_counter=0, frame=0),
        transitions=states,
    )
    # Avoid env requirement in _snap by leaving env None — thin snap path
    calls: list[tuple[int, tuple[str, ...], str]] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        calls.append((frames, buttons, reason))
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), reason)
        return sess.state

    monkeypatch.setattr(spark, "hold", _hold)
    # First step inside loop sees speed_counter from transitions after holds;
    # seed so first observation already boosting after 0 holds:
    session.state = _state(pose=9, velocity_y=0, speed_counter=4, frame=0)
    out = spark.charge_until_boost(session, "RIGHT", budget=10)
    assert out["ok"] is True
    assert out["frames"] == 0
    assert out["boost"]["pose"] == 9
    assert calls == []  # already boosting


def test_charge_runs_until_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(
        _state(pose=9, velocity_y=0, speed_counter=0, frame=0),
    )

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), reason)
        return sess.state

    monkeypatch.setattr(spark, "hold", _hold)
    out = spark.charge_until_boost(session, "LEFT", budget=5)
    assert out["ok"] is False
    assert out["frames"] == 5
    assert out["direction"] == "LEFT"
    assert "never reached" in (out.get("error") or "")


def test_crouch_store_arms_on_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _FakeEnv(_ram_with_spark(echoes=4, timer=0))
    session = _FakeSession(
        _state(pose=9, velocity_y=0, speed_counter=4, shinespark_timer=0, frame=0),
        env=env,
    )
    armed_at = 2

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        assert "DOWN" in buttons
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), reason)
            # After armed_at steps, set timer in ram
            if sess.frame >= armed_at:
                sess.env._ram[ADDR_SHINESPARK_TIMER] = 179
                sess.env._ram[ADDR_SHINESPARK_TIMER + 1] = 0
                sess.state = replace(sess.state, shinespark_timer=179, pose=53)
        return sess.state

    monkeypatch.setattr(spark, "hold", _hold)
    out = spark.crouch_store(session, max_frames=10)
    assert out["ok"] is True
    assert out["armed"] is not None
    assert out["armed"]["spark_timer"] == 179
    assert out["armed"]["store_frame_index"] == armed_at - 1  # 0-based after step
    assert out["peak_timer_during_store"] == 179


def test_wait_store_window_tracks_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _FakeEnv(_ram_with_spark(echoes=0, timer=10))
    session = _FakeSession(
        _state(pose=39, shinespark_timer=10, frame=0),
        env=env,
    )

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), reason)
            # drain 1 per frame
            t = max(0, int(sess.env._ram[ADDR_SHINESPARK_TIMER]) - 1)
            sess.env._ram[ADDR_SHINESPARK_TIMER] = t
            sess.state = replace(sess.state, shinespark_timer=t, frame=sess.frame)
        return sess.state

    monkeypatch.setattr(spark, "hold", _hold)
    out = spark.wait_store_window(session, 20, hold_down=False)
    assert out["timer_start"] == 9 or out["timer_start"] == 10  # after first step
    assert out["timer_end"] == 0
    assert out["frames_timer_gt0"] >= 1
    assert out["drain_per_frame"] is not None


def test_activate_shinespark_aim_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _FakeEnv(_ram_with_spark(timer=100))
    session = _FakeSession(
        _state(pose=39, shinespark_timer=100, samus_x=50, samus_y=200, frame=0),
        env=env,
    )
    seen: list[tuple[str, ...]] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        seen.append(buttons)
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), reason)
            if "A" in buttons:
                sess.state = replace(
                    sess.state,
                    pose=201,
                    shinespark_timer=50,
                    samus_x=sess.state.samus_x + 5,
                    samus_y=sess.state.samus_y - 3,
                    frame=sess.frame,
                )
            else:
                sess.state = replace(sess.state, pose=39, frame=sess.frame)
        return sess.state

    monkeypatch.setattr(spark, "hold", _hold)
    out = spark.activate_shinespark(
        session,
        "RIGHT",
        "UP",
        hold_frames=3,
        travel_budget=2,
        pre_stand_frames=2,
        pre_stand_buttons=("UP",),
    )
    assert out["ok"] is True
    assert out["spark_pose_seen"] is True
    assert out["aim"] == ("RIGHT", "UP")
    assert ("UP",) in seen  # pre-stand
    assert ("RIGHT", "UP", "A") in seen
    assert out["min_y"] < 200
    assert out["max_x"] > 50
    assert 203 in spark.SPARK_POSES


def test_api_shapes_charge_store_activate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(
        _state(pose=9, velocity_y=0, speed_counter=4, shinespark_timer=0, frame=0),
    )

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), reason)
            if "DOWN" in buttons:
                sess.state = replace(
                    sess.state, shinespark_timer=179, pose=53, frame=sess.frame
                )
            if "A" in buttons:
                sess.state = replace(
                    sess.state, pose=201, shinespark_timer=50, frame=sess.frame
                )
        return sess.state

    monkeypatch.setattr(spark, "hold", _hold)
    out = spark.charge_store_activate(
        session, idle_after_store=0, travel_budget=5, aim_buttons=("RIGHT", "UP")
    )
    assert "charge" in out and "store" in out and "activate" in out
    assert out["charge"]["ok"] is True
    assert out["store"]["ok"] is True
    assert out["activate"]["ok"] is True
    assert out["ok"] is True


def test_store_then_spin_unspin_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(
        _state(pose=53, shinespark_timer=179, frame=0),
    )
    reasons: list[str] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        reasons.append(reason)
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), reason)
            if reason.endswith("_act_0") or "act_" in reason:
                sess.state = replace(sess.state, pose=201, frame=sess.frame)
        return sess.state

    monkeypatch.setattr(spark, "hold", _hold)
    out = spark.store_then_spin_unspin_activate(
        session,
        stand_frames=2,
        hop_frames=3,
        unspin_frames=2,
        micro_run_frames=1,
        travel_budget=5,
    )
    assert out["ok"] is True
    joined = " ".join(reasons)
    assert "hop_carry_stand" in joined
    assert "hop_carry_hop" in joined
    assert "hop_carry_unspin" in joined
    assert "hop_carry_micro_run" in joined
