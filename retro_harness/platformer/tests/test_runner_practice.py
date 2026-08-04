"""ROM-free tests for shared practice-mode bookkeeping."""

import json
from types import SimpleNamespace

import numpy as np

from retro_harness.platformer.level_config import LevelConfig, PlatformerRAM
from retro_harness.platformer import runner
from retro_harness.platformer.runner import (
    _best_practice_attempt,
    _load_practice_pb_frames,
    _parse_room_id_arg,
    _practice_completion_token,
)


def _config(**overrides) -> LevelConfig:
    values = {
        "level_id": "practice_test",
        "display_name": "Practice Test",
        "game_name": "Test-Snes",
        "game_dir_name": "test",
        "start_state": "Start",
        "ram": PlatformerRAM(level_id=(0, "u16")),
        "target_level_id": 10,
        "completion_level_ids": [20],
        "completion_min_progress": 2.0,
    }
    values.update(overrides)
    return LevelConfig(**values)


def test_practice_completion_requires_configured_room_and_progress():
    config = _config()

    assert _practice_completion_token(config, {"level_id": 10}, 3.0) is None
    assert _practice_completion_token(config, {"level_id": 30}, 3.0) is None
    assert _practice_completion_token(config, {"level_id": 20}, 1.9) is None
    assert _practice_completion_token(config, {"level_id": 20}, 2.0) == (
        "level_id",
        20,
    )


def test_practice_completion_rejects_aliases_and_exclusions():
    config = _config(level_id_aliases=[11], completion_exclude_ids=[30])
    config.completion_level_ids = []

    assert _practice_completion_token(config, {"level_id": 11}, 3.0) is None
    assert _practice_completion_token(config, {"level_id": 30}, 3.0) is None
    assert _practice_completion_token(config, {"level_id": 40}, 3.0) == (
        "level_id",
        40,
    )


def test_practice_completion_supports_ram_flags():
    config = _config(
        completion_signal="ram_flag",
        completion_ram_key="goal",
        completion_ram_value=7,
    )

    assert _practice_completion_token(config, {"goal": 6}, 3.0) is None
    assert _practice_completion_token(config, {"goal": 7}, 3.0) == (
        "ram_flag",
        7,
    )


def test_load_practice_pb_frames_uses_fastest_completed_attempt(tmp_path):
    attempts = [
        (0, 120, True),
        (1, 90, False),
        (2, 105, True),
    ]
    for attempt, frames, completed in attempts:
        data = {
            "actions": [0] * frames,
            "num_frames": frames,
            "metadata": {"completed": completed, "total_frames": frames},
        }
        (tmp_path / f"attempt_{attempt:03d}.json").write_text(json.dumps(data))
    (tmp_path / "attempt_999.json").write_text("not json")
    (tmp_path / "attempt_002_raw.json").write_text(
        json.dumps({"metadata": {"completed": True, "total_frames": 1}}),
    )

    assert _load_practice_pb_frames(tmp_path) == 105


def test_load_practice_pb_frames_returns_none_without_completions(tmp_path):
    assert _load_practice_pb_frames(tmp_path) is None


def test_best_practice_attempt_prefers_fastest_completion():
    attempts = [
        {"attempt": 0, "frames": 500, "max_progress": 8.0, "completed": True},
        {"attempt": 1, "frames": 300, "max_progress": 7.0, "completed": True},
        {"attempt": 2, "frames": 100, "max_progress": 9.0, "completed": False},
    ]

    assert _best_practice_attempt(attempts) == 1


def test_parse_room_id_accepts_int_decimal_and_hex():
    assert _parse_room_id_arg(0x9B5B) == 0x9B5B
    assert _parse_room_id_arg("39771") == 39771
    assert _parse_room_id_arg("0x9B5B") == 0x9B5B
    assert _parse_room_id_arg("9B5B") == 0x9B5B


def test_cmd_practice_saves_success_and_resets_cached_state(
    tmp_path,
    monkeypatch,
):
    config = _config(
        ram=PlatformerRAM(
            level_id=(0, "u16"),
            player_x=(2, "u16"),
            player_y=(4, "u16"),
            extras={"health": (6, "u16")},
        ),
        completion_min_progress=0.0,
        progress_axis="player_y",
    )

    class FakeEmulator:
        def __init__(self):
            self.loaded_states = []

        def get_state(self):
            return b"cached-practice-state"

        def set_state(self, state):
            self.loaded_states.append(state)

    class FakeEnv:
        def __init__(self):
            self.em = FakeEmulator()
            self.ram_reads = 0

        def reset(self):
            return np.zeros((1, 1, 3), dtype=np.uint8), {}

        def get_ram(self):
            ram = np.zeros(8, dtype=np.uint8)
            room_id = 10 if self.ram_reads == 0 else 20
            self.ram_reads += 1
            ram[0], ram[1] = room_id & 0xFF, room_id >> 8
            ram[2], ram[3] = 123, 0
            ram[4], ram[5] = 200, 0
            ram[6], ram[7] = 99, 0
            return ram

    class FakePlaySession:
        def __init__(self, env, **kwargs):
            self.last_action_post_sanitize = [0] * 12

        def run(self):
            for _ in range(61):
                self.on_step(None, 0.0, False, {})

    env = FakeEnv()
    monkeypatch.setattr(runner, "_resolve_config", lambda args: config)
    monkeypatch.setattr("retro_harness.env.make_env", lambda **kwargs: env)
    monkeypatch.setattr("retro_harness.play_session.PlaySession", FakePlaySession)

    runner.cmd_practice(
        SimpleNamespace(
            level=config.level_id,
            state=None,
            scale=1,
            save_name=None,
            output_dir=str(tmp_path),
            session_label="worker-a",
        ),
    )

    attempt = json.loads((tmp_path / "attempt_000.json").read_text())
    raw = json.loads((tmp_path / "attempt_000_raw.json").read_text())
    summary = json.loads((tmp_path / "practice_summary.json").read_text())

    assert attempt["metadata"]["completed"] is True
    assert attempt["metadata"]["session_label"] == "worker-a"
    assert set(raw) == {"raw_buttons", "metadata"}
    assert summary["best_completion_frames"] == 1
    assert env.em.loaded_states == [b"cached-practice-state"]


def test_cmd_practice_continues_across_configured_completion_to_target(
    tmp_path,
    monkeypatch,
):
    config = _config(
        ram=PlatformerRAM(
            level_id=(0, "u16"),
            player_x=(2, "u16"),
            player_y=(4, "u16"),
            extras={
                "health": (6, "u16"),
                "game_state": (8, "u16"),
                "door_transition": (10, "u16"),
                "max_health": (12, "u16"),
                "missiles": (14, "u16"),
                "max_missiles": (16, "u16"),
                "super_missiles": (18, "u16"),
                "max_super_missiles": (20, "u16"),
            },
        ),
        completion_min_progress=0.0,
        progress_axis="player_y",
    )

    class FakeEmulator:
        def __init__(self):
            self.loaded_states = []

        def get_state(self):
            return b"continuous-start"

        def set_state(self, state):
            self.loaded_states.append(state)

    class FakeEnv:
        def __init__(self):
            self.em = FakeEmulator()
            self.rooms = [10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30]
            self.ram_reads = 0

        def reset(self):
            return np.zeros((1, 1, 3), dtype=np.uint8), {}

        def get_ram(self):
            read_index = self.ram_reads
            room_id = self.rooms[min(read_index, len(self.rooms) - 1)]
            self.ram_reads += 1
            ram = np.zeros(22, dtype=np.uint8)
            ram[0], ram[1] = room_id & 0xFF, room_id >> 8
            ram[2], ram[3] = self.ram_reads, 0
            ram[4], ram[5] = 200, 0
            ram[6], ram[7] = 99, 0
            playable = room_id == 30 and read_index >= 10
            ram[8] = 8 if playable else 6
            ram[10] = 0 if playable else int(room_id == 30)
            ram[12] = 99
            ram[14] = 4
            ram[16] = 5
            ram[18] = 2
            ram[20] = 3
            return ram

    class FakePlaySession:
        def __init__(self, env, **kwargs):
            self.last_action_post_sanitize = [1] + [0] * 11

        def run(self):
            for _ in range(70):
                self.on_step(None, 0.0, False, {})

    env = FakeEnv()
    monkeypatch.setattr(runner, "_resolve_config", lambda args: config)
    monkeypatch.setattr("retro_harness.env.make_env", lambda **kwargs: env)
    monkeypatch.setattr("retro_harness.play_session.PlaySession", FakePlaySession)

    runner.cmd_practice(
        SimpleNamespace(
            level=config.level_id,
            state=None,
            scale=1,
            save_name=None,
            output_dir=str(tmp_path),
            session_label="through-rooms",
            keep_playing=True,
            until_room="0x001E",
            until_playable=True,
            until_label="Spore Spawn Super Room",
            room_debounce=3,
        ),
    )

    attempt = json.loads((tmp_path / "attempt_000.json").read_text())
    raw = json.loads((tmp_path / "attempt_000_raw.json").read_text())
    metadata = attempt["metadata"]

    assert attempt["num_frames"] == 10
    assert metadata["completed"] is True
    assert metadata["terminal_reason"] == "until_room"
    assert metadata["until_room_id"] == 30
    assert metadata["until_playable"] is True
    assert metadata["until_label"] == "Spore Spawn Super Room"
    assert [split["room_id"] for split in metadata["room_splits"]] == [20, 30]
    assert [split["frame"] for split in metadata["room_splits"]] == [3, 7]
    assert metadata["room_splits"][0]["configured_completion"] is True
    assert metadata["room_splits"][0]["health"] == 99
    assert metadata["room_splits"][0]["max_health"] == 99
    assert metadata["room_splits"][0]["missiles"] == 4
    assert metadata["room_splits"][0]["max_missiles"] == 5
    assert metadata["room_splits"][0]["super_missiles"] == 2
    assert metadata["room_splits"][0]["max_super_missiles"] == 3
    assert metadata["room_splits"][0]["game_state"] == 6
    assert metadata["room_splits"][0]["door_transition"] == 0
    assert len(raw["raw_buttons"]) == 10
    summary = json.loads((tmp_path / "practice_summary.json").read_text())
    assert summary["until_label"] == "Spore Spawn Super Room"
    assert env.em.loaded_states == [b"continuous-start"]


def test_cmd_practice_checkpoint_load_truncates_frames_and_splits(
    tmp_path,
    monkeypatch,
):
    config = _config(
        ram=PlatformerRAM(level_id=(0, "u16")),
        completion_min_progress=0.0,
        progress_axis="player_y",
    )

    class FakeEmulator:
        def get_state(self):
            return b"checkpoint-start"

        def set_state(self, state):
            pass

    class FakeEnv:
        def __init__(self):
            self.em = FakeEmulator()
            self.rooms = [10] * 6 + [20] * 3
            self.ram_reads = 0

        def reset(self):
            return np.zeros((1, 1, 3), dtype=np.uint8), {}

        def get_ram(self):
            room_id = self.rooms[min(self.ram_reads, len(self.rooms) - 1)]
            self.ram_reads += 1
            ram = np.zeros(2, dtype=np.uint8)
            ram[0], ram[1] = room_id & 0xFF, room_id >> 8
            return ram

    class FakePlaySession:
        def __init__(self, env, **kwargs):
            self.last_action_post_sanitize = [1] + [0] * 11

        def save_checkpoint(self, slot):
            return 5

        def load_checkpoint(self, slot):
            return 5

        def run(self):
            for _ in range(5):
                self.on_step(None, 0.0, False, {})
            self.on_trigger_save(1)
            for _ in range(3):
                self.on_step(None, 0.0, False, {})
            self.on_trigger_load(1)

    env = FakeEnv()
    monkeypatch.setattr(runner, "_resolve_config", lambda args: config)
    monkeypatch.setattr("retro_harness.env.make_env", lambda **kwargs: env)
    monkeypatch.setattr("retro_harness.play_session.PlaySession", FakePlaySession)

    runner.cmd_practice(
        SimpleNamespace(
            level=config.level_id,
            state=None,
            scale=1,
            save_name=None,
            output_dir=str(tmp_path),
            session_label="rewind",
            keep_playing=True,
            until_room=None,
            room_debounce=3,
        ),
    )

    attempt = json.loads((tmp_path / "attempt_000.json").read_text())
    assert attempt["num_frames"] == 5
    assert attempt["metadata"]["room_splits"] == []
    assert attempt["metadata"]["terminal_reason"] == "user_exit"


def test_cmd_practice_done_saves_one_terminal_episode(tmp_path, monkeypatch):
    config = _config(
        ram=PlatformerRAM(level_id=(0, "u16")),
        completion_min_progress=0.0,
        progress_axis="player_y",
    )

    class FakeEmulator:
        def __init__(self):
            self.loaded_states = []

        def get_state(self):
            return b"done-start"

        def set_state(self, state):
            self.loaded_states.append(state)

    class FakeEnv:
        def __init__(self):
            self.em = FakeEmulator()

        def reset(self):
            return np.zeros((1, 1, 3), dtype=np.uint8), {}

        def get_ram(self):
            ram = np.zeros(2, dtype=np.uint8)
            ram[0] = 10
            return ram

    class FakePlaySession:
        def __init__(self, env, **kwargs):
            self.last_action_post_sanitize = [1] + [0] * 11

        def run(self):
            self.on_step(None, 0.0, True, {})
            for _ in range(60):
                self.on_step(None, 0.0, False, {})

    env = FakeEnv()
    monkeypatch.setattr(runner, "_resolve_config", lambda args: config)
    monkeypatch.setattr("retro_harness.env.make_env", lambda **kwargs: env)
    monkeypatch.setattr("retro_harness.play_session.PlaySession", FakePlaySession)

    runner.cmd_practice(
        SimpleNamespace(
            level=config.level_id,
            state=None,
            scale=1,
            save_name=None,
            output_dir=str(tmp_path),
            session_label="done",
        ),
    )

    attempt = json.loads((tmp_path / "attempt_000.json").read_text())
    assert attempt["num_frames"] == 1
    assert attempt["metadata"]["terminal_reason"] == "env_done"
    assert env.em.loaded_states == [b"done-start"]


def test_cmd_practice_discards_no_input_tail_on_exit(tmp_path, monkeypatch):
    config = _config(
        ram=PlatformerRAM(level_id=(0, "u16")),
        completion_min_progress=0.0,
        progress_axis="player_y",
    )

    class FakeEmulator:
        def get_state(self):
            return b"tail-start"

        def set_state(self, state):
            pass

    class FakeEnv:
        def __init__(self):
            self.em = FakeEmulator()

        def reset(self):
            return np.zeros((1, 1, 3), dtype=np.uint8), {}

        def get_ram(self):
            ram = np.zeros(2, dtype=np.uint8)
            ram[0] = 10
            return ram

    class FakePlaySession:
        def __init__(self, env, **kwargs):
            self.last_action_post_sanitize = [0] * 12

        def run(self):
            for _ in range(30):
                self.on_step(None, 0.0, False, {})

    monkeypatch.setattr(runner, "_resolve_config", lambda args: config)
    monkeypatch.setattr("retro_harness.env.make_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr("retro_harness.play_session.PlaySession", FakePlaySession)

    runner.cmd_practice(
        SimpleNamespace(
            level=config.level_id,
            state=None,
            scale=1,
            save_name=None,
            output_dir=str(tmp_path),
            session_label="empty-tail",
        ),
    )

    summary = json.loads((tmp_path / "practice_summary.json").read_text())
    assert summary["total_attempts"] == 0
    assert not list(tmp_path.glob("attempt_*.json"))
