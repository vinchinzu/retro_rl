"""Unit tests for frame-saving helpers (no emulator required)."""

from __future__ import annotations

from dataclasses import dataclass

from platformer_common.frame_tools import (
    analyze_seed_static,
    cleanup_auto_inputs,
    compress_hold_window,
    count_leading_idle,
    count_trailing_idle,
    find_button_hold_stalls,
    find_stalls,
    is_idle_frame,
    load_raw_frames,
    save_raw_seed,
    search_hold_compressions,
    trim_after_completion,
    trim_leading_idle,
)


def _idle(n: int = 12) -> list[int]:
    return [0] * n


def _right(n: int = 12) -> list[int]:
    f = _idle(n)
    f[7] = 1
    return f


def test_is_idle_and_counts() -> None:
    frames = [_idle(), _idle(), _right(), _right(), _idle()]
    assert is_idle_frame(_idle())
    assert not is_idle_frame(_right())
    assert count_leading_idle(frames) == 2
    assert count_trailing_idle(frames) == 1


def test_find_stalls_progress() -> None:
    progress = [0, 1, 1, 1, 1, 1, 2, 3]
    stalls = find_stalls(progress, min_length=3)
    assert len(stalls) == 1
    assert stalls[0].start == 2
    assert stalls[0].length == 4


def test_find_button_hold_stalls() -> None:
    frames = [_right()] * 25 + [_idle()] * 5 + [_right()] * 10
    holds = find_button_hold_stalls(frames, min_length=20)
    assert len(holds) == 1
    assert holds[0].start == 0
    assert holds[0].length == 25


def test_compress_hold_window() -> None:
    frames = [_idle()] * 3 + [_right()] * 10 + [_idle()] * 2
    out = compress_hold_window(frames, 3, 13, 4)
    assert len(out) == 3 + 4 + 2
    assert all(f == _right() for f in out[3:7])


def test_cleanup_auto_inputs() -> None:
    frames = [_right(), _right(), _right()]
    states = [8, 4, 5]  # 4/5 automated
    cleaned, zeroed = cleanup_auto_inputs(frames, states, auto_states=(4, 5))
    assert zeroed == 2
    assert cleaned[0] == _right()
    assert cleaned[1] == _idle()
    assert cleaned[2] == _idle()


@dataclass
class _FakeResult:
    completed: bool
    total_frames: int
    fitness: float = 0.0


def test_trim_leading_idle_even_parity() -> None:
    """Even trims complete; odd trims fail — mirrors SMB phase traps."""
    base = [_idle()] * 10 + [_right()] * 20

    def evaluate(frames: list[list[int]]) -> _FakeResult:
        lead_removed = 10 - count_leading_idle(frames) if count_leading_idle(frames) <= 10 else 0
        # Infer trim from length
        trim = 30 - len(frames)
        if trim % 2 == 1:
            return _FakeResult(completed=False, total_frames=len(frames), fitness=0)
        clear = 25 - trim  # even trims save frames
        return _FakeResult(completed=True, total_frames=clear, fitness=100_000 - clear)

    result = trim_leading_idle(base, evaluate, parity="even", require_completion=True)
    assert result.completed
    assert result.trim % 2 == 0
    assert result.trim == 10  # max even leading idle
    assert result.clear_frames == 15


def test_trim_after_completion_pads_idle() -> None:
    base = [_right()] * 50 + [_idle()] * 100

    def evaluate(frames: list[list[int]]) -> _FakeResult:
        # Completes at frame 40 if we still have at least 40 frames of content
        if len(frames) < 40:
            return _FakeResult(completed=False, total_frames=len(frames))
        return _FakeResult(completed=True, total_frames=40, fitness=99_960)

    result = trim_after_completion(base, evaluate, pad=5)
    assert result.completed
    assert len(result.frames) == 45
    assert all(is_idle_frame(f) for f in result.frames[40:])


def test_search_hold_compressions_shortens() -> None:
    # Long hold of RIGHT that can be cut to 5 without failing
    base = [_right()] * 40 + [_idle()] * 10

    def evaluate(frames: list[list[int]]) -> _FakeResult:
        # Need at least 5 RIGHT at start
        rights = 0
        for f in frames:
            if f == _right():
                rights += 1
            else:
                break
        if rights < 5:
            return _FakeResult(completed=False, total_frames=len(frames))
        clear = rights + 5  # fake clear after rights + a bit
        return _FakeResult(completed=True, total_frames=clear, fitness=100_000 - clear)

    result = search_hold_compressions(
        base, evaluate, min_hold=20, min_keep=5, max_trials_per_hold=12
    )
    assert result.completed
    assert len(result.frames) < len(base)
    assert count_leading_idle(result.frames) == 0


def test_save_and_load_raw_roundtrip(tmp_path) -> None:
    frames = [_idle(), _right(), _right()]
    path = tmp_path / "seed.json"
    save_raw_seed(path, frames, metadata={"completed": True})
    loaded = load_raw_frames(path)
    assert loaded == frames


def test_load_nes9_rle(tmp_path) -> None:
    import json

    path = tmp_path / "rle.json"
    path.write_text(
        json.dumps(
            {
                "format": "nes9_rle",
                "segments": [
                    {"b": [0, 0, 0, 0, 0, 0, 0, 0, 0], "n": 3},
                    {"b": [0, 0, 0, 0, 0, 0, 0, 1, 0], "n": 2},
                ],
            }
        )
    )
    frames = load_raw_frames(path)
    assert len(frames) == 5
    assert is_idle_frame(frames[0])
    assert frames[3][7] == 1


def test_analyze_seed_static() -> None:
    frames = [_idle()] * 5 + [_right()] * 30 + [_idle()] * 3
    report = analyze_seed_static(frames)
    assert report["leading_idle"] == 5
    assert report["trailing_idle"] == 3
    assert report["longest_hold"] >= 30
    assert report["num_frames"] == 38
