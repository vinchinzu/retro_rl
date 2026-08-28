"""No-ROM tests for the 4x4 autobot room-grid demo."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest

from super_metroid.demo.room_grid import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    DEFAULT_TILES,
    FrameBudget,
    GridTile,
    composite_grid,
    label_frame,
    probe_parallel,
    record_play_flags,
    tile_inventory,
    xstack_filter,
)
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS
from super_metroid.source_states import get_source


def test_default_tiles_are_a_4x4_of_distinct_rooms() -> None:
    n = DEFAULT_COLS * DEFAULT_ROWS
    assert len(DEFAULT_TILES) == n
    segments = [t.segment for t in DEFAULT_TILES]
    sources = [t.source_id for t in DEFAULT_TILES]
    rooms = [get_source(t.source_id).room_id for t in DEFAULT_TILES]
    assert len(set(segments)) == n
    assert len(set(sources)) == n
    assert len(set(rooms)) == n
    for tile in DEFAULT_TILES:
        assert tile.segment in KPDR_SEGMENTS
        assert tile.label.strip()
        get_source(tile.source_id)


def test_inventory_reports_registration_without_booting() -> None:
    rows = tile_inventory(DEFAULT_TILES)
    assert len(rows) == 16
    assert all(row["segmentRegistered"] for row in rows)
    assert {row["roomIdHex"] for row in rows} == {
        get_source(t.source_id).room_hex() for t in DEFAULT_TILES
    }


def test_xstack_filter_uses_ffmpeg_grid() -> None:
    text = xstack_filter(16, cols=4, rows=4)
    assert "grid=4x4" in text
    assert text.endswith("[out]")
    assert "[0:v]" in text and "[15:v]" in text
    with pytest.raises(ValueError):
        xstack_filter(15, cols=4, rows=4)


def test_label_frame_paints_a_top_bar() -> None:
    frame = np.zeros((224, 256, 3), dtype=np.uint8)
    frame[:] = (40, 80, 120)
    labeled = label_frame(frame, "Ice Beam")
    assert labeled.shape == frame.shape
    assert tuple(int(v) for v in labeled[2, 8]) != (40, 80, 120)


def test_probe_parallel_refuses_high_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("super_metroid.demo.room_grid.os.cpu_count", lambda: 8)
    monkeypatch.setattr(
        "super_metroid.demo.room_grid.os.getloadavg", lambda: (7.0, 6.0, 5.0)
    )
    monkeypatch.setattr("super_metroid.demo.room_grid.mem_available_mib", lambda: 32_000)
    verdict = probe_parallel(16)
    assert verdict.ncpus == 8
    assert verdict.workers == 16
    assert verdict.ok is False
    assert "--force" in verdict.reason


def test_probe_parallel_ok_on_idle_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("super_metroid.demo.room_grid.os.cpu_count", lambda: 32)
    monkeypatch.setattr(
        "super_metroid.demo.room_grid.os.getloadavg", lambda: (1.2, 1.0, 0.8)
    )
    monkeypatch.setattr("super_metroid.demo.room_grid.mem_available_mib", lambda: 50_000)
    verdict = probe_parallel(16)
    assert verdict.ok is True
    assert verdict.recommended_workers >= 1


def _tiny_clip(path: Path, color: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=64x48:d=0.2:r=60",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_composite_grid_2x2(tmp_path: Path) -> None:
    colors = ("red", "green", "blue", "yellow")
    clips = []
    for i, color in enumerate(colors):
        path = tmp_path / f"{i}.mp4"
        _tiny_clip(path, color)
        clips.append(path)
    out = tmp_path / "grid.mp4"
    composite_grid(
        clips,
        out,
        cols=2,
        rows=2,
        seconds=0.4,
        cell_w=64,
        cell_h=48,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = probe.stdout.strip().split(",")
    assert int(width) == 128
    assert int(height) == 96


def test_record_tiles_refuses_workers_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from super_metroid.demo.room_grid import record_tiles

    monkeypatch.setattr("super_metroid.demo.room_grid.os.cpu_count", lambda: 4)
    monkeypatch.setattr(
        "super_metroid.demo.room_grid.os.getloadavg", lambda: (4.0, 4.0, 4.0)
    )
    monkeypatch.setattr("super_metroid.demo.room_grid.mem_available_mib", lambda: 8_000)
    tiles = (
        GridTile("ice_snake_to_ice", "post_ice_acid_to_snake_pure", "Ice"),
        GridTile("moat_cross", "post_kihunter_to_moat_pure", "Moat"),
    )
    with pytest.raises(RuntimeError, match="load1"):
        record_tiles(tiles, tmp_path, workers=2, force=False)


def test_capped_clip_is_not_hop_success() -> None:
    hop_ok, capped = record_play_flags(None)
    assert hop_ok is True
    assert capped is False
    hop_ok, capped = record_play_flags(FrameBudget(1800))
    assert hop_ok is False
    assert capped is True
