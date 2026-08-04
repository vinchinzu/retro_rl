"""ROM-free tests for route practice seed loading."""

import json

from retro_harness.platformer.route import _load_practice_seeds


def _write_attempt(path, actions, raw_buttons=None):
    path.write_text(json.dumps({"actions": actions}))
    if raw_buttons is not None:
        path.with_name(path.stem + "_raw.json").write_text(
            json.dumps({"raw_buttons": raw_buttons}),
        )


def test_load_practice_seeds_prefers_paired_raw_buttons(tmp_path):
    actions = [3] * 3
    raw_buttons = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
    _write_attempt(tmp_path / "attempt_000.json", actions, raw_buttons)

    assert _load_practice_seeds(tmp_path, min_frames=3) == [
        (raw_buttons, True),
    ]


def test_load_practice_seeds_falls_back_to_legacy_actions(tmp_path):
    actions = [1, 2, 3]
    _write_attempt(tmp_path / "attempt_000.json", actions)

    assert _load_practice_seeds(tmp_path, min_frames=3) == [(actions, False)]


def test_load_practice_seeds_filters_raw_companions_and_short_attempts(tmp_path):
    _write_attempt(
        tmp_path / "attempt_000.json",
        [1] * 5,
        raw_buttons=[[0, 0, 0]] * 2,
    )
    _write_attempt(tmp_path / "attempt_001.json", [2] * 3)

    assert _load_practice_seeds(tmp_path, min_frames=3) == [
        ([2, 2, 2], False),
    ]
