"""Unit tests for super_metroid.start_presets (canonical + alias map)."""

from __future__ import annotations

from super_metroid.start_presets import (
    POWER_ON_STARTS,
    START_PRESETS,
    _ALIASES,
    _CANONICAL,
    resolve_start_preset,
)


def test_known_keys_present() -> None:
    for name in (
        "morph",
        "bomb",
        "post-bombs",
        "varia",
        "grapple",
        "main-street",
        "plasma-beam",
        "golden-torizo",
        "metal-pirates",
        "post-ridley",
        "post-draygon",
    ):
        assert name in START_PRESETS


def test_main_street_is_full_start_v1_seam() -> None:
    path, blurb = resolve_start_preset("main-street")
    assert path == "scratch/full_start_v1_main_street.state"
    assert "0xCFC9" in blurb
    assert resolve_start_preset("maridia")[0] == path
    assert resolve_start_preset("full-start-main-street")[0] == path


def test_golden_torizo_is_full_start_v1_recover() -> None:
    path, blurb = resolve_start_preset("golden-torizo")
    assert path == "scratch/full_start_v1_golden_torizo.state"
    assert "0xB283" in blurb
    assert "left door" in blurb
    assert resolve_start_preset("gt")[0] == path
    assert resolve_start_preset("gt-entry")[0] == path
    assert resolve_start_preset("full-start-gt")[0] == path


def test_metal_pirates_is_full_start_v1_seam() -> None:
    path, blurb = resolve_start_preset("metal-pirates")
    assert path == "scratch/full_start_v1_metal_pirates.state"
    assert "0xB62B" in blurb
    assert "0x732F" in blurb
    assert resolve_start_preset("mp")[0] == path
    assert resolve_start_preset("pirates")[0] == path
    assert resolve_start_preset("full-start-metal-pirates")[0] == path


def test_post_ridley_is_full_start_v1_seam() -> None:
    path, blurb = resolve_start_preset("post-ridley")
    assert path == "scratch/full_start_v1_ridley.state"
    assert "0xB698" in blurb
    assert "0x732F" in blurb
    assert resolve_start_preset("ridley-tank")[0] == path
    assert resolve_start_preset("full-start-ridley")[0] == path


def test_plasma_beam_is_full_start_v1_seam() -> None:
    path, blurb = resolve_start_preset("plasma-beam")
    assert path == "scratch/full_start_v1_plasma.state"
    assert "0xD2AA" in blurb
    assert "0x100F" in blurb
    assert resolve_start_preset("plasma")[0] == path
    assert resolve_start_preset("post-plasma")[0] == path
    assert resolve_start_preset("full-start-plasma")[0] == path


def test_no_empty_paths() -> None:
    for key, (path, blurb) in START_PRESETS.items():
        assert path, f"empty path for {key!r}"
        assert blurb, f"empty blurb for {key!r}"


def test_aliases_share_canonical_path() -> None:
    for alias, canon in _ALIASES.items():
        assert canon in _CANONICAL, f"alias {alias!r} → missing {canon!r}"
        assert START_PRESETS[alias][0] == START_PRESETS[canon][0]
        assert START_PRESETS[alias][1] == f"Alias of {canon}"


def test_more_keys_than_unique_paths() -> None:
    paths = {v[0] for v in START_PRESETS.values()}
    assert len(START_PRESETS) >= len(paths)
    assert len(START_PRESETS) > len(paths)  # aliases exist
    assert len(paths) == len(_CANONICAL)
    assert len(START_PRESETS) == len(_CANONICAL) + len(_ALIASES)


def test_resolve_start_preset() -> None:
    assert resolve_start_preset("morph") == START_PRESETS["morph"]
    assert resolve_start_preset("post-bombs") == START_PRESETS["post-bombs"]
    assert resolve_start_preset("no-such-preset") is None


def test_power_on_starts_unchanged() -> None:
    assert POWER_ON_STARTS == frozenset(
        {"start", "power-on", "beginning", "full", "poweron"}
    )


def test_canonical_paths_unique() -> None:
    paths = [p for p, _ in _CANONICAL.values()]
    assert len(paths) == len(set(paths))
