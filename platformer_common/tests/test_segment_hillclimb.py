"""Unit tests for segment hillclimb mutator helpers (no emulator)."""

from __future__ import annotations

from platformer_common.segment_hillclimb import _mutate_window


def test_mutate_window_preserves_prefix() -> None:
    frames = [[0] * 12 for _ in range(100)]
    for i in range(100):
        frames[i][7] = 1 if i % 2 == 0 else 0
    prefix = [list(f) for f in frames[:40]]
    for _ in range(50):
        cand, strategy = _mutate_window(
            frames,
            lo=40,
            hi=80,
            prefer_trim=True,
            hold_stalls=[(50, 20)],
        )
        assert cand[:40] == prefix, f"prefix corrupted by {strategy}"
        assert strategy != "noop" or len(cand) >= 40
