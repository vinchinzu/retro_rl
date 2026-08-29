"""ROM-free tests for the MK2 Liu Kang fireball policy."""

from __future__ import annotations

import numpy as np

from mortal_kombat_ii.ram import make_test_ram
from mortal_kombat_ii.scripted import (
    DOWN,
    KIND_SCRIPT,
    RIGHT,
    Y,
    ScriptedPolicy,
    fireball_sequence,
)

RGB = None


def _bits(frame: np.ndarray) -> set[int]:
    return {i for i, v in enumerate(frame) if v}


def test_kind_and_name() -> None:
    assert KIND_SCRIPT == "script"
    assert ScriptedPolicy.kind == "script"
    assert ScriptedPolicy.name == "scripted"


def test_fireball_is_qcf_then_hp() -> None:
    seq = fireball_sequence()
    assert len(seq) > 12
    assert _bits(seq[0]) == {DOWN}
    assert RIGHT in _bits(seq[8])
    assert Y in _bits(seq[12])
    assert DOWN not in _bits(seq[12])


def test_intro_is_idle_then_queues_fireball() -> None:
    policy = ScriptedPolicy(intro_frames=2)
    ram = make_test_ram()
    first = policy.act(ram, RGB)
    second = policy.act(ram, RGB)
    third = policy.act(ram, RGB)
    assert not _bits(first)
    assert not _bits(second)
    assert _bits(third) == {DOWN}


def test_ko_health_idles() -> None:
    policy = ScriptedPolicy(intro_frames=0)
    assert not _bits(policy.act(make_test_ram(p1_health=0, p2_health=80), RGB))
    assert not _bits(policy.act(make_test_ram(p1_health=80, p2_health=0), RGB))


def test_reset_replays_identically() -> None:
    policy = ScriptedPolicy(intro_frames=0)
    ram = make_test_ram()
    first = [policy.act(ram, RGB) for _ in range(8)]
    policy.reset()
    replay = [policy.act(ram, RGB) for _ in range(8)]
    assert all(np.array_equal(a, b) for a, b in zip(first, replay, strict=True))
