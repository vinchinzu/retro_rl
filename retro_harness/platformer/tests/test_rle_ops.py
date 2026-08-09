"""Unit tests for RLE representation and mutation operators (no emulator)."""

from __future__ import annotations

from retro_harness.platformer.rle_ops import (
    SMB_ACTION_ATOMIC_PATTERNS,
    compress_rle,
    crossover_rle_frame_aligned,
    crossover_rle_single,
    expand_rle,
    mutate_duration,
    mutate_merge,
    mutate_rle,
    mutate_split,
    rle_normalize,
    rle_replace_window,
    rle_slice_frames,
    rle_total_frames,
    RleMutateConfig,
)


def test_compress_expand_roundtrip_indices() -> None:
    frames = [1, 1, 1, 2, 2, 0, 0, 0, 0]
    runs = compress_rle(frames)
    assert runs == [(1, 3), (2, 2), (0, 4)]
    assert expand_rle(runs) == frames
    assert rle_total_frames(runs) == len(frames)


def test_compress_expand_buttons() -> None:
    a = (0, 0, 0, 0, 0, 0, 0, 1, 0)
    b = (1, 0, 0, 0, 0, 0, 0, 1, 0)
    frames = [a, a, b, b, b]
    runs = compress_rle(frames)
    assert len(runs) == 2
    assert runs[0] == (a, 2)
    assert runs[1] == (b, 3)
    expanded = expand_rle(runs, as_list=True)
    assert expanded[0] == list(a)
    assert len(expanded) == 5


def test_normalize_merges_adjacent() -> None:
    runs = [(1, 2), (1, 3), (2, 1), (2, 0), (3, 4)]
    assert rle_normalize(runs) == [(1, 5), (2, 1), (3, 4)]


def test_expand_buttons_frames_are_independent() -> None:
    a = (0, 0, 0, 0, 0, 0, 0, 1, 0)
    expanded = expand_rle([(a, 3)], as_list=True)
    assert len(expanded) == 3
    expanded[0][7] = 0
    assert expanded[1] == list(a)
    assert expanded[2] == list(a)


def test_slice_and_replace_window() -> None:
    frames = [0] * 10 + [1] * 10 + [2] * 10
    runs = compress_rle(frames)
    mid = rle_slice_frames(runs, 10, 20)
    assert expand_rle(mid) == [1] * 10
    new = rle_replace_window(runs, 10, 20, [(9, 5)])
    assert expand_rle(new) == [0] * 10 + [9] * 5 + [2] * 10


def test_mutate_duration_preserves_payload() -> None:
    runs = [(2, 20), (3, 5)]
    out = mutate_duration(runs, max_delta=3)
    assert len(out) >= 1
    assert all(p in (2, 3) for p, _ in out)
    assert rle_total_frames(out) != 0


def test_split_and_merge() -> None:
    runs = [(1, 10)]
    split = mutate_split(runs)
    assert rle_total_frames(split) == 10
    assert len(split) == 2
    merged = mutate_merge(split)
    assert rle_total_frames(merged) == 10


def test_mutate_rle_produces_valid_sequence() -> None:
    runs = compress_rle([1] * 50 + [2] * 30 + [0] * 10)
    cfg = RleMutateConfig(
        atomic_patterns=SMB_ACTION_ATOMIC_PATTERNS,  # type: ignore[arg-type]
        num_actions=11,
    )
    for _ in range(30):
        out = mutate_rle(runs, config=cfg, n_ops=2)
        assert rle_total_frames(out) >= 1
        # round-trip expand/compress should not explode
        flat = expand_rle(out)
        assert len(flat) == rle_total_frames(out)


def test_crossover_lengths() -> None:
    p1 = compress_rle([1] * 100)
    p2 = compress_rle([2] * 80 + [3] * 40)
    c1 = crossover_rle_single(p1, p2)
    c2 = crossover_rle_frame_aligned(p1, p2)
    assert rle_total_frames(c1) >= 1
    assert rle_total_frames(c2) >= 1
