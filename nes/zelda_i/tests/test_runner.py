"""Unit tests for shared zelda_i.runner helpers (no emulator)."""

from __future__ import annotations

from enum import Enum, auto
from types import SimpleNamespace

from zelda_i.runner import (
    add_common_args,
    add_video_args,
    controller_stopped,
    resolve_video,
    write_report,
)
import argparse
from pathlib import Path


class _Phase(Enum):
    SETTLE = auto()
    FAILED = auto()
    DONE = auto()


def test_controller_stopped_accepts_enum_and_string_phases() -> None:
    assert controller_stopped(SimpleNamespace(success=True, failed=False, phase=None))
    assert controller_stopped(SimpleNamespace(success=False, failed=True, phase=None))
    assert controller_stopped(
        SimpleNamespace(success=False, failed=False, phase=_Phase.FAILED)
    )
    assert controller_stopped(
        SimpleNamespace(success=False, failed=False, phase=_Phase.DONE)
    )
    assert not controller_stopped(
        SimpleNamespace(success=False, failed=False, phase=_Phase.SETTLE)
    )
    assert controller_stopped(
        SimpleNamespace(success=False, failed=False, phase="failed")
    )
    assert controller_stopped(SimpleNamespace(success=False, failed=False, phase="DONE"))
    assert not controller_stopped(
        SimpleNamespace(success=False, failed=False, phase="walk")
    )


def test_add_common_args_and_write_report(tmp_path, monkeypatch) -> None:
    from zelda_i import runner as runner_mod

    monkeypatch.setattr(runner_mod, "RECORDINGS_DIR", tmp_path)
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="Level2Compass", default_tag="isolated")
    args = parser.parse_args([])
    assert args.from_state == "Level2Compass"
    assert args.tag == "isolated"
    assert args.infinite_life is False

    path = write_report("level2_bomb_north", {"ok": True}, tag="isolated")
    assert path == tmp_path / "level2_bomb_north_isolated.json"
    assert path.is_file()


def test_spine_video_args_default_on_and_no_video_wins() -> None:
    parser = argparse.ArgumentParser()
    add_video_args(parser, default_on=True)
    on = parser.parse_args([])
    assert on.video == "AUTO"
    assert on.no_video is False
    off = parser.parse_args(["--no-video"])
    path, config, _intro = resolve_video(
        off, default_path=Path("/tmp/survival_spine.mp4")
    )
    assert path is None
    assert config is None
