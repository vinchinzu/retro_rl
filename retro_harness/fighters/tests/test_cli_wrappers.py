"""Offline tests for extracted fighter CLIs (no pygame / torch import)."""

from __future__ import annotations

import ast
from pathlib import Path

from retro_harness.repo import resolve_game_dir

FIGHTERS_DIR = Path(__file__).resolve().parents[1]
CLI_MODULES = ("watch", "validate_states", "validate_single_state")
RUN_BOT_GAMES = (
    "mortal_kombat_ii",
    "street_fighter_ii",
    "super_street_fighter_ii",
)
GAMES = (
    "mortal_kombat",
    "mortal_kombat_ii",
    "street_fighter_ii",
    "super_street_fighter_ii",
)
PUBLIC_FNS = {
    "watch": ("main", "watch"),
    "validate_states": ("main", "validate_state"),
    "validate_single_state": ("main", "validate_state"),
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_names(tree: ast.AST, module: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def _calls_name(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == name:
                return True
    return False


def _top_level_functions(tree: ast.AST) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_canonical_modules_expose_public_fns() -> None:
    for name, required in PUBLIC_FNS.items():
        defs = _top_level_functions(_parse(FIGHTERS_DIR / f"{name}.py"))
        assert set(required) <= defs, f"{name} missing {set(required) - defs}"


def test_canonical_clis_use_resolve_game_dir() -> None:
    for name in CLI_MODULES:
        path = FIGHTERS_DIR / f"{name}.py"
        source = path.read_text(encoding="utf-8")
        tree = _parse(path)
        assert "resolve_game_dir" in _imported_names(tree, "retro_harness.repo")
        assert _calls_name(tree, "resolve_game_dir")
        assert "SCRIPT_DIR.parent" not in source
        assert "ROOT_DIR /" not in source


def test_run_bot_wrappers_call_harness() -> None:
    for game in RUN_BOT_GAMES:
        path = resolve_game_dir(game) / "run_bot.py"
        names = _imported_names(_parse(path), "retro_harness.fighters.run_bot")
        assert "main" in names, f"{game}/run_bot.py must wrap fighters.run_bot"


def test_per_game_wrappers_reexport_main() -> None:
    for game in GAMES:
        game_dir = resolve_game_dir(game)
        for name, required in PUBLIC_FNS.items():
            path = game_dir / f"{name}.py"
            names = _imported_names(_parse(path), f"retro_harness.fighters.{name}")
            missing = set(required) - names
            assert not missing, f"{game}/{name}.py must re-export {sorted(missing)}"
