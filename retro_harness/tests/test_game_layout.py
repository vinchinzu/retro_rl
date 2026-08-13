"""Standard game path layout helper."""

from __future__ import annotations

from pathlib import Path

from retro_harness.game_layout import game_paths


def test_flat_game_paths(tmp_path: Path) -> None:
    package = tmp_path / "snes" / "demo" / "paths.py"
    package.parent.mkdir(parents=True)
    package.write_text("# stub\n")
    gp = game_paths(package, "Demo-Snes")
    assert gp.game_dir == package.parent
    assert gp.repo_root == tmp_path
    assert gp.integration == "Demo-Snes"
    assert gp.game == "Demo-Snes"
    assert gp.integration_dir == package.parent / "custom_integrations" / "Demo-Snes"
    assert gp.recordings_dir == package.parent / "recordings"


def test_nested_workspace_parent(tmp_path: Path) -> None:
    package = tmp_path / "snes" / "harvest" / "harvest" / "paths.py"
    package.parent.mkdir(parents=True)
    package.write_text("# stub\n")
    gp = game_paths(package, "HarvestMoon-Snes", workspace_parent=True)
    assert gp.game_dir == tmp_path / "snes" / "harvest"
    assert gp.repo_root == tmp_path


def test_converted_nes_paths_game_dir() -> None:
    import castlevania.paths as paths

    assert paths.GAME_DIR == Path(paths.__file__).resolve().parent


def test_nested_hals_golf_paths() -> None:
    import hals_golf.paths as paths

    workspace = Path(paths.__file__).resolve().parents[1]
    assert paths.PROJECT_DIR == workspace
    assert paths.GAME == "HalsHoleInOne-Snes"
    assert paths.GAME_DIR == workspace / "custom_integrations" / paths.GAME
    assert paths.RECORDINGS_DIR == workspace / "recordings"
