"""Unit tests for in-video RTA split panel + freeze-on-axe."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from smb.rta_panel import (
    RtaSplitTracker,
    draw_rta_split_panel,
    stage_label,
)


def _snap(*, world: int, level: int, oper_mode: int = 1) -> SimpleNamespace:
    return SimpleNamespace(world=world, level=level, oper_mode=oper_mode)


def test_stage_label_warps() -> None:
    assert stage_label(0, 0) == "1-1"
    assert stage_label(0, 1) == "1-2"
    assert stage_label(3, 0) == "4-1"
    assert stage_label(7, 3) == "8-4"


def test_tracker_locks_on_level_change_and_freezes_on_axe() -> None:
    t = RtaSplitTracker(fps=60.0)

    # Enter 1-1
    assert t.observe(_snap(world=0, level=0), clock_frames=1) is False
    assert t.current_label == "1-1"

    # Still 1-1
    assert t.observe(_snap(world=0, level=0), clock_frames=100) is False

    # Exit 1-1 → 1-2
    assert t.observe(_snap(world=0, level=1), clock_frames=1734) is True
    assert t.completed[-1]["label"] == "1-1"
    assert t.completed[-1]["cum_frames"] == 1734
    assert t.current_label == "1-2"

    # Load flicker 1-2 → 1-3 must NOT lock a bogus split.
    assert t.observe(_snap(world=0, level=2), clock_frames=2200) is False
    assert t.current_label == "1-2"
    assert len(t.completed) == 1

    # Warp 1-2 → 4-1
    assert t.observe(_snap(world=3, level=0), clock_frames=3556) is True
    assert t.completed[-1]["label"] == "1-2"

    # Fast-forward remaining warps route stages (incl. 4-3 flicker ignore)
    path = [
        ((3, 1), 5832),  # 4-1 → 4-2
        ((3, 2), 6000),  # flicker 4-3 — ignore
        ((7, 0), 7513),  # 4-2 → 8-1
        ((7, 1), 10603),  # 8-1 → 8-2
        ((7, 2), 12977),  # 8-2 → 8-3
        ((7, 3), 15371),  # 8-3 → 8-4
    ]
    for (w, lv), clock in path:
        t.observe(_snap(world=w, level=lv), clock_frames=clock)

    assert [r["label"] for r in t.completed] == [
        "1-1",
        "1-2",
        "4-1",
        "4-2",
        "8-1",
        "8-2",
        "8-3",
    ]
    assert t.current_label == "8-4"

    # Axe freezes 8-4
    assert t.observe(_snap(world=7, level=3, oper_mode=2), clock_frames=18031) is True
    assert t.frozen is True
    assert t.freeze_frame == 18031
    assert t.completed[-1]["label"] == "8-4"
    assert t.completed[-1]["kind"] == "axe"
    assert t.completed[-1]["cum_frames"] == 18031

    # Peach hold must not move the clock/splits
    assert t.observe(_snap(world=7, level=3, oper_mode=2), clock_frames=19000) is False
    assert t.freeze_frame == 18031
    assert len(t.completed) == 8

    lines = t.lines(clock_frames=19000)
    assert lines[0] == "RTA"
    assert any(line.startswith("8-4") and "*" in line for line in lines)
    assert lines[-1].startswith("AXE")
    assert "5:00.51" in lines[-1] or "5:00.52" in lines[-1]  # 18031/60


def test_draw_panel_composites_without_resize() -> None:
    obs = np.zeros((224, 256, 3), dtype=np.uint8)
    obs[:] = (40, 80, 120)
    out = draw_rta_split_panel(
        obs,
        ["RTA", "1-1  0:28.84", "1-2  0:59.15", "TOT 0:59.15"],
    )
    assert out.shape == (224, 256, 3)
    # Panel region should differ from flat fill.
    assert not np.array_equal(out[:40, :70], obs[:40, :70])
