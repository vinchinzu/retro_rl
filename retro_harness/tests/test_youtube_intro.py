"""Tests for generic YouTube project intro slides."""

from __future__ import annotations

from retro_harness.video import FOOTER_HEIGHT
from retro_harness.youtube_intro import (
    PROJECT_NAME,
    PROJECT_REPO,
    project_intro_lines,
    render_intro_card,
)


def test_project_intro_lines_are_generic_and_game_aware() -> None:
    lines = project_intro_lines(
        game_title="Super Mario Bros. (NES)",
        run_summary="Clean power-on any% warp → 8-4 ending",
    )
    assert lines[0] == PROJECT_NAME
    assert "Super Mario Bros." in lines[2]
    assert any("power-on" in line.lower() for line in lines)
    assert PROJECT_REPO in lines
    # Shared method disclosure so every game's YouTube card matches.
    assert any("continuous" in line.lower() for line in lines)
    assert any("stitch" in line.lower() for line in lines)
    # Default bitmap fonts are ASCII-only — keep intro glyphs renderable.
    joined = "\n".join(lines)
    assert "→" not in joined
    assert "·" not in joined


def test_project_intro_lines_do_not_emit_blank_lines() -> None:
    lines = project_intro_lines(
        game_title="Blank Intervention",
        run_summary="Continuous reset-to-ending clear",
        intervention="   ",
    )
    assert all(line for line in lines)


def test_render_intro_card_matches_gameplay_geometry() -> None:
    lines = project_intro_lines(
        game_title="Test Game",
        run_summary="Continuous reset-to-ending clear",
    )
    card = render_intro_card(lines, width=256, height=224, with_footer=True)
    assert card.shape == (224 + FOOTER_HEIGHT, 256, 3)
    bare = render_intro_card(lines, width=256, height=224, with_footer=False)
    assert bare.shape == (224, 256, 3)
