"""Tests for SMB full-run stitch / playthrough planning."""

from __future__ import annotations

from pathlib import Path

import pytest

from smb.full_run import (
    build_session_playthrough_plan,
    build_stitch_plan,
    resolve_state_path,
    verify_clip_deathless,
)
from smb.paths import (
    FULLGAME_RECORDINGS_DIR,
    INTEGRATION_V0_DIR,
    SNES_EDITOR_SMB_ROOT,
)
from smb.routes import ROUTE_ALL_EXITS, ROUTE_WARP_ANY_PERCENT, get_route, list_routes


def test_warp_route_has_eight_exits() -> None:
    route = get_route("warp")
    assert route.route_id == ROUTE_WARP_ANY_PERCENT.route_id
    assert len(route.exits) == 8
    assert [e.exit_id for e in route.exits] == [
        "1-1",
        "1-2",
        "4-1",
        "4-2",
        "8-1",
        "8-2",
        "8-3",
        "8-4",
    ]


def test_all_exits_route_has_thirty_two() -> None:
    route = get_route("all_exits")
    assert route.route_id == ROUTE_ALL_EXITS.route_id
    assert len(route.exits) == 32
    assert route.exits[0].exit_id == "1-1"
    assert route.exits[-1].exit_id == "8-4"
    assert route.exits[0].segment_id == "smb_1_1"
    assert route.exits[-1].segment_id == "smb_8_4"


def test_list_routes_dedupes() -> None:
    routes = list_routes()
    ids = [r.route_id for r in routes]
    assert ids.count(ROUTE_WARP_ANY_PERCENT.route_id) == 1
    assert ids.count(ROUTE_ALL_EXITS.route_id) == 1


def test_resolve_state_path_integration_basename() -> None:
    if not INTEGRATION_V0_DIR.exists():
        pytest.skip("SuperMarioBros-Nes-v0 integration not linked")
    path = resolve_state_path(
        "Level1_1.state",
        state_name="Level1_1",
        integration_dir=INTEGRATION_V0_DIR,
    )
    assert path.exists()
    assert path.name == "Level1_1.state"


def test_resolve_state_path_rewrites_legacy_absolute() -> None:
    if not SNES_EDITOR_SMB_ROOT.exists():
        pytest.skip("snes_editor SMB tree not present")
    legacy = (
        Path("/home/v/01_projects/11_games/speedrun/retro_rl/super_mario_bros")
        / "custom_integrations"
        / "SuperMarioBros-Nes-v0"
        / "Practice_smb_8_2_start.state"
    )
    if not (INTEGRATION_V0_DIR / "Practice_smb_8_2_start.state").exists():
        pytest.skip("practice state missing")
    path = resolve_state_path(
        str(legacy),
        state_name="Practice_smb_8_2_start",
        integration_dir=INTEGRATION_V0_DIR,
    )
    assert path.exists()


@pytest.mark.skipif(
    not (FULLGAME_RECORDINGS_DIR / "leaderboard.json").exists(),
    reason="fullgame leaderboard not linked",
)
def test_build_legal_stitch_plan_warp() -> None:
    plan = build_stitch_plan("warp", source="legal_stitch", skip_missing=True)
    assert plan.source_kind == "legal_stitch"
    assert len(plan.clips) == 8
    assert plan.missing == []
    assert plan.total_play_frames == sum(c.frames for c in plan.clips)
    # Leaderboard legal stitch is ~19k frames.
    assert 15000 < plan.total_play_frames < 25000
    for clip in plan.clips:
        assert clip.state_path.exists(), clip.state_path
        assert len(clip.play_buttons) == clip.frames
        assert clip.frames > 0
    # 4-1 best starts mid-branch — should carry a non-zero prefix.
    four_one = next(c for c in plan.clips if c.exit.exit_id == "4-1")
    assert four_one.meta.get("branch_offset", 0) > 0
    assert len(four_one.prefix_buttons) == four_one.meta["branch_offset"]

    manifest = plan.to_manifest()
    assert manifest["route_id"] == ROUTE_WARP_ANY_PERCENT.route_id
    assert len(manifest["clips"]) == 8


@pytest.mark.skipif(
    not (FULLGAME_RECORDINGS_DIR / "leaderboard.json").exists(),
    reason="fullgame leaderboard not linked",
)
def test_build_legal_stitch_plan_all_exits_skips_missing() -> None:
    plan = build_stitch_plan("all_exits", source="legal_stitch", skip_missing=True)
    # Only warp-route stages are in the any% leaderboard today.
    assert len(plan.clips) == 8
    assert len(plan.missing) == 24
    assert "2-1" in plan.missing


def _emulator_available() -> bool:
    try:
        import retro  # type: ignore

        integrations = getattr(getattr(retro, "data", None), "Integrations", None)
        return integrations is not None and hasattr(integrations, "CUSTOM")
    except Exception:
        return False


@pytest.mark.skipif(
    not (FULLGAME_RECORDINGS_DIR / "20260429_172649" / "summary.json").exists(),
    reason="playthrough session not linked",
)
@pytest.mark.skipif(not _emulator_available(), reason="real stable-retro not available")
def test_session_playthrough_172649_all_clean() -> None:
    """Known completed any% session verifies death-free for every warp exit."""
    plan = build_session_playthrough_plan(
        get_route("warp"),
        "20260429_172649",
        require_verified=True,
        skip_missing=False,
    )
    assert plan.source_kind == "playthrough"
    assert len(plan.clips) == 8
    assert plan.missing == []
    assert all(c.session_id == "20260429_172649" for c in plan.clips)
    # 8-1 / 8-2 must be present and non-trivial.
    eight_one = next(c for c in plan.clips if c.exit.exit_id == "8-1")
    eight_two = next(c for c in plan.clips if c.exit.exit_id == "8-2")
    assert eight_one.frames > 500
    assert eight_two.frames > 500
    for clip in (eight_one, eight_two):
        report = verify_clip_deathless(clip)
        assert report["ok"], (clip.exit.exit_id, report)


@pytest.mark.skipif(
    not (FULLGAME_RECORDINGS_DIR / "leaderboard.json").exists(),
    reason="fullgame leaderboard not linked",
)
@pytest.mark.skipif(not _emulator_available(), reason="real stable-retro not available")
def test_default_playthrough_source_selects_verified_session() -> None:
    plan = build_stitch_plan("warp", source="playthrough", skip_missing=False)
    assert plan.source_kind == "playthrough"
    assert len(plan.clips) == 8
    assert plan.missing == []
    sessions = {c.session_id for c in plan.clips}
    assert len(sessions) == 1
    assert plan.total_play_frames > 10000


@pytest.mark.skipif(
    not (FULLGAME_RECORDINGS_DIR / "20260429_214207" / "summary.json").exists(),
    reason="playthrough session not linked",
)
def test_session_playthrough_windows_resolve_without_emulator() -> None:
    """Window extraction works even when verify is skipped (CI stub)."""
    plan = build_session_playthrough_plan(
        get_route("warp"),
        "20260429_214207",
        require_verified=False,
        skip_missing=False,
    )
    assert len(plan.clips) == 8
    assert all(c.session_id == "20260429_214207" for c in plan.clips)
    eight_one = next(c for c in plan.clips if c.exit.exit_id == "8-1")
    eight_two = next(c for c in plan.clips if c.exit.exit_id == "8-2")
    assert eight_one.frames >= 2000  # long mid-level clear, not a desync death run
    assert eight_two.frames >= 500
    for clip in plan.clips:
        assert clip.state_path.exists()
        assert clip.frames == len(clip.play_buttons) > 0
