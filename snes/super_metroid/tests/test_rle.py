"""Unit locks for shared RLE loader / script player (rr-7sn.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_metroid.routes.kpdr.spazer import helpers as spazer_helpers
from super_metroid.routes.rle import load_rle_json, play_script

_KPDR_DATA = Path(__file__).resolve().parents[1] / "routes" / "kpdr" / "data"
_GATE_OPEN_JSON = _KPDR_DATA / "double_chamber_gate_open_rle.json"

# Product gate-open cadence (pre-rr-7sn.2 inlined tuple) — must match JSON.
_KNOWN_GATE_OPEN: tuple[tuple[int, tuple[str, ...]], ...] = (
    (2, ()),
    (27, ("R",)),
    (10, ("X", "R")),
    (11, ("R",)),
    (10, ("X", "R")),
    (2, ("R",)),
    (10, ("B", "A", "R")),
    (5, ("B", "A", "X", "R")),
    (3, ("A", "X", "R")),
    (2, ("X", "R")),
    (11, ("R",)),
    (32, ()),
    (19, ("R",)),
    (26, ("A", "R")),
    (1, ("A", "X", "R")),
    (8, ("X", "R")),
    (5, ()),
    (7, ("X",)),
    (2, ()),
    (6, ("X",)),
    (55, ()),
    (4, ("RIGHT",)),
    (56, ()),
    (8, ("SELECT",)),
    (33, ()),
    (9, ("SELECT",)),
    (8, ()),
    (13, ("A",)),
    (2, ("A", "X", "R")),
    (7, ("X", "R")),
    (4, ("R",)),
    (7, ("X", "R")),
    (2, ("R",)),
    (26, ()),
    (7, ("RIGHT",)),
    (2, ("B", "RIGHT")),
    (11, ("B", "RIGHT", "A")),
    (6, ("RIGHT",)),
    (33, ("B", "RIGHT")),
    (1, ("RIGHT",)),
    (43, ()),
    (15, ("RIGHT",)),
)


def test_load_gate_open_rle_matches_known_product_tuple() -> None:
    loaded = load_rle_json(_GATE_OPEN_JSON)
    assert loaded == _KNOWN_GATE_OPEN
    assert len(loaded) == 42
    assert sum(n for n, _ in loaded) == 551


def test_load_rle_json_accepts_n_b_and_list_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            [
                {"n": 3, "b": ["RIGHT"]},
                {"n": 1, "buttons": ["B", "A"]},
                [2, ["LEFT"]],
                {"n": 4, "b": []},
            ]
        ),
        encoding="utf-8",
    )
    assert load_rle_json(path) == (
        (3, ("RIGHT",)),
        (1, ("B", "A")),
        (2, ("LEFT",)),
        (4, ()),
    )


def test_load_rle_json_rejects_bad_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"n": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        load_rle_json(path)


def test_spazer_helpers_play_script_is_shared() -> None:
    """Spazer must not keep a private play_script implementation."""
    assert spazer_helpers.play_script is play_script


def test_k4_wave_loads_gate_open_from_data() -> None:
    """Wave product path loads gate-open RLE from data/*.json (rr-7sn.1 + .2)."""
    from super_metroid.routes.kpdr.wave import scripts as wave_scripts

    assert wave_scripts.HUMAN_GATE_OPEN_RLE == _KNOWN_GATE_OPEN
    # Must resolve under data/ (not an inlined paste-only constant).
    assert wave_scripts._GATE_OPEN_RLE_PATH.name == "double_chamber_gate_open_rle.json"
    assert wave_scripts._GATE_OPEN_RLE_PATH.is_file()
    # No inlined private paste on the thin k4_wave facade.
    from super_metroid.routes.kpdr import k4_wave

    assert not hasattr(k4_wave, "_HUMAN_GATE_OPEN_RLE")
