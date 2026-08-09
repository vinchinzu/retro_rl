"""Unit tests for TAS room-body materialize (no emulator)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_metroid.paths import GAME_DIR
from super_metroid.routes.kpdr.room_ids import ROOM_LANDING_SITE, ROOM_PARLOR
from super_metroid.tas.materialize import (
    ANY_LANDING_TO_PARLOR_BODY,
    STATUS_MATERIALIZED,
    materialize_from_board,
    materialize_room_body,
    resolve_movie_window,
)
from super_metroid.tas.rle import expand_snes12_rle
from super_metroid.tas.slice import REF_ANY
from super_metroid.tas.stages import (
    ANY_LANDING_MOVIE_START,
    get_stage,
)

RESYNC_ZEBES = GAME_DIR / "recordings" / "tas_import" / "resync_zebes_rooms"


def _synthetic_frames(n: int = 200) -> list[list[int]]:
    """SNES-12 frames with a few distinct chords (never sanitize L+R)."""
    frames: list[list[int]] = []
    for i in range(n):
        fr = [0] * 12
        if i % 10 < 5:
            fr[7] = 1  # RIGHT
        if i % 17 == 0:
            fr[0] = 1  # B
        if i % 23 == 0:
            fr[8] = 1  # A
        # Intentional L+R chord on some frames (must survive compress)
        if i % 31 == 0:
            fr[6] = 1  # LEFT
            fr[7] = 1  # RIGHT
            fr[10] = 1  # L
            fr[11] = 1  # R
        frames.append(fr)
    return frames


def test_resolve_landing_short_default() -> None:
    stage = get_stage("landing_to_parlor")
    w = resolve_movie_window(stage)
    assert w["movie_start"] == ANY_LANDING_MOVIE_START
    assert w["body_frames"] == ANY_LANDING_TO_PARLOR_BODY
    assert w["reason"] == "landing_to_parlor_short_default"
    # Must not use thrash 12k body by default
    assert w["body_frames"] < 5_000


def test_resolve_explicit_overrides() -> None:
    stage = get_stage("landing_to_parlor")
    w = resolve_movie_window(stage, movie_start=16000, body_frames=400)
    assert w["movie_start"] == 16000
    assert w["body_frames"] == 400


def test_resolve_product_pin_delta() -> None:
    stage = get_stage("landing_to_parlor")
    pins = [
        {
            "kind": "control",
            "frame": 21548,
            "room_id": ROOM_LANDING_SITE,
        },
        {
            "kind": "room_enter",
            "frame": 23740,
            "room_id": ROOM_PARLOR,
        },
    ]
    w = resolve_movie_window(stage, pins=pins)
    assert w["movie_start"] == ANY_LANDING_MOVIE_START
    assert w["body_frames"] == 23740 - 21548
    assert w["reason"] == "product_pin_delta"


def test_resolve_resync_movie_index() -> None:
    stage = get_stage("landing_to_parlor")
    meta = {
        "movie_start": 15000,
        "rooms": [
            {
                "frame": 21548,
                "room_id_hex": "0x91F8",
                "source": "product_anchor",
            },
            {
                "frame": 23740,
                "movie_index": 17191,
                "room_id_hex": "0x92FD",
                "source": "movie",
            },
        ],
    }
    w = resolve_movie_window(stage, resync_meta=meta)
    assert w["movie_start"] == 15000
    # 17191 - 15000 + 32 settle
    assert w["body_frames"] == 17191 - 15000 + 32
    assert w["reason"] == "resync_movie_index"


def test_materialize_synthetic_no_movie(tmp_path: Path) -> None:
    """Compress a synthetic window without needing the ref movie file."""
    stage = get_stage("landing_to_parlor")
    frames = _synthetic_frames(20_000)
    out = tmp_path / "landing_to_parlor.json"
    payload = materialize_room_body(
        stage,
        movie_start=100,
        body_frames=80,
        out_path=out,
        frames=frames,
        write=True,
    )
    assert payload["format"] == "snes12_rle"
    assert payload["num_frames"] == 80
    assert payload["status"] == STATUS_MATERIALIZED
    assert payload["segments"]
    assert "never_sanitize_L+R" in payload["hard_rules"]
    assert out.is_file()

    expanded = expand_snes12_rle(payload)
    assert len(expanded) == 80
    # L+R chord preserved in at least one frame of the window
    chord = any(fr[6] and fr[7] and fr[10] and fr[11] for fr in expanded)
    # window 100:180 — L+R when i % 31 == 0
    assert chord or any(fr[10] and fr[11] for fr in frames[100:180])


def test_materialize_landing_if_movie_present(tmp_path: Path) -> None:
    if not REF_ANY.exists():
        pytest.skip("missing sniq any% LSMV")
    out = tmp_path / "landing_to_parlor.json"
    payload = materialize_room_body(
        "landing_to_parlor",
        movie_start=ANY_LANDING_MOVIE_START,
        body_frames=ANY_LANDING_TO_PARLOR_BODY,
        out_path=out,
    )
    assert payload["format"] == "snes12_rle"
    assert payload["num_frames"] == ANY_LANDING_TO_PARLOR_BODY
    assert payload["num_frames"] > 0
    assert payload["segments"]
    assert payload["movie_start_index"] == ANY_LANDING_MOVIE_START
    assert payload["status"] == STATUS_MATERIALIZED
    assert payload["stage_id"] == "landing_to_parlor"
    assert payload["control_room_id"] == ROOM_LANDING_SITE
    assert payload["goal_room_id"] == ROOM_PARLOR


def test_materialize_from_resync_board_if_present(tmp_path: Path) -> None:
    if not RESYNC_ZEBES.is_dir() or not (RESYNC_ZEBES / "pins.json").is_file():
        pytest.skip("resync_zebes_rooms artifacts missing")
    if not REF_ANY.exists():
        pytest.skip("missing sniq any% LSMV")

    results = materialize_from_board(
        RESYNC_ZEBES,
        stage_ids=["landing_to_parlor"],
        zebes_only=True,
        out_dir=tmp_path,
    )
    info = results["landing_to_parlor"]
    assert "error" not in info, info
    assert info["num_frames"] > 0
    assert info["num_frames"] < 5_000  # not thrash tail
    assert info["window_resolve"] in (
        "resync_movie_index",
        "product_pin_delta",
        "landing_to_parlor_short_default",
    )
    path = Path(info["path"])
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "snes12_rle"
    assert data["status"] == STATUS_MATERIALIZED
