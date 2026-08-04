"""Tests for shared state save paths."""

from __future__ import annotations

import gzip
import importlib.util
import sys
import types
from pathlib import Path


def _retro_stub() -> types.SimpleNamespace:
    integrations = types.SimpleNamespace(add_custom_path=lambda *_args, **_kwargs: None)
    return types.SimpleNamespace(
        RetroEnv=object,
        data=types.SimpleNamespace(Integrations=integrations),
    )


def _load_module(name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader

    previous = sys.modules.get("stable_retro")
    sys.modules["stable_retro"] = _retro_stub()
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
        if previous is None:
            sys.modules.pop("stable_retro", None)
        else:
            sys.modules["stable_retro"] = previous
    return module


save_state = _load_module("retro_harness_env_test", "env.py").save_state
RecordingSession = _load_module("retro_harness_recorder_test", "recorder.py").RecordingSession


class _FakeEmulator:
    def __init__(self, payload: bytes):
        self._payload = payload

    def get_state(self) -> bytes:
        return self._payload


class _FakeEnv:
    def __init__(self, payload: bytes):
        self.em = _FakeEmulator(payload)


def test_save_state_writes_only_to_custom_integrations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = b"super-metroid-state"
    env = _FakeEnv(payload)
    game_dir = tmp_path / "demo_game"

    save_path = save_state(env, game_dir, "SuperMetroid-Snes", "QuickSave")

    expected = game_dir / "custom_integrations" / "SuperMetroid-Snes" / "QuickSave.state"
    assert save_path == expected
    assert expected.exists()
    assert not (tmp_path / "QuickSave.state").exists()
    with gzip.open(expected, "rb") as fh:
        assert fh.read() == payload


def test_recording_session_quick_save_writes_only_to_custom_integrations(tmp_path):
    payload = b"practice-state"
    env = _FakeEnv(payload)
    game_dir = tmp_path / "donkey_kong_country"
    session = RecordingSession(
        label="level_217",
        game="DonkeyKongCountry-Snes",
        game_dir=game_dir,
    )

    save_path = session.quick_save(env, "QuickSave")

    expected = game_dir / "custom_integrations" / "DonkeyKongCountry-Snes" / "QuickSave.state"
    assert save_path == expected
    assert expected.exists()
    assert not (game_dir / "QuickSave.state").exists()
    with gzip.open(expected, "rb") as fh:
        assert fh.read() == payload
