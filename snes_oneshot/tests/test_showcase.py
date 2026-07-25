"""Tests for generic oneshot showcase recording."""

from __future__ import annotations

from snes_oneshot.recording import FooterLabels, RecordingSession
from snes_oneshot.showcase import default_output_path, load_showcase_game


class _Sink:
    def __init__(self) -> None:
        self.frames: list[object] = []

    def write(self, frame: object) -> None:
        self.frames.append(frame)


def test_load_great_waldo_showcase() -> None:
    game = load_showcase_game("great_waldo_search")
    assert game.slug == "great_waldo_search"
    assert len(game.clips()) == 5
    output = default_output_path(game)
    assert output.name == "great_waldo_search_segmented_completion_showcase.mp4"


def test_recording_session_uses_footer_provider() -> None:
    class _Env:
        def step(self, action: list[int]) -> tuple[object, dict[str, object]]:
            del action
            import numpy as np

            return np.zeros((224, 256, 3), dtype=np.uint8), {}

    sink = _Sink()

    def footer(
        env: object,
        action: list[int],
        frame: int,
        fps: float,
    ) -> FooterLabels:
        del env, action, fps
        return FooterLabels(
            upper_left=f"FRAME {frame}",
            upper_right="00:00",
            lower_left="TEST",
        )

    session = RecordingSession(_Env(), sink=sink, footer=footer, fps=60.0)
    session.step([0] * 12)
    assert len(sink.frames) == 1
