"""FightResult, the shared WRAM write, and detour drop. No ROM."""

from __future__ import annotations

import pytest

from alttp.dungeon.bosses import FightResult
from alttp.ram import assign_u8
from alttp.rollout import PolicyGuard, press


class _Memory:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def assign(self, bus: int, kind: str, value: int) -> None:
        self.calls.append((bus, kind, value))


class _Env:
    def __init__(self) -> None:
        self.data = type("Data", (), {"memory": _Memory()})()


def test_fight_result_defaults() -> None:
    row = FightResult(ok=True, frames=3)
    assert row.ok is True
    assert row.frames == 3
    assert row.blocker == ""
    assert row.strikes == 0
    assert row.floor_hits == 0
    assert row.fire_hits == 0
    assert row.sword_hits == 0


def test_assign_u8_refuses_follower() -> None:
    env = _Env()
    with pytest.raises(SystemExit, match=r"refusing to write \$F3CC"):
        assign_u8(env, 0xF3CC, 1)
    assert env.data.memory.calls == []


def test_assign_u8_writes_non_follower() -> None:
    inner = _Env()
    wrapped = type("Wrap", (), {"unwrapped": inner, "data": None})()
    assign_u8(wrapped, 0xF36D, 0x108)
    assign_u8(wrapped, 0x22, 0x1FF)
    assert inner.data.memory.calls == [
        (0x7E0000 + 0xF36D, "|u1", 0x08),
        (0x22, "|u1", 0xFF),
    ]
    assert wrapped.data is None


def test_drop_detour_clears_queued_frame() -> None:
    guard = PolicyGuard(object())
    queued = press("LEFT")
    guard._queue.append(queued)
    assert guard._queue == [queued]
    guard.drop_detour()
    assert guard._queue == []
