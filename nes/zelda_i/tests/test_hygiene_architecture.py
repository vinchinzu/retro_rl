"""Layer contracts: anchors, bomb-wall engine, door stands, dungeon IDs."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from zelda_i.anchors import (
    ENTRANCES,
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL5_DOOR,
    SCREEN_LEVEL5_ENTRANCE,
    TF_BIT_L3,
    TRIFORCE_BITS_BY_LEVEL,
)
from zelda_i.door_graph.level2_exits import _BOMB_STAND_6F_N
from zelda_i.level2.puzzles import BOMB_WALL_6F_NORTH

_PKG_ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_ROOT_PY = frozenset(
    {
        "__init__.py",
        "ram.py",
        "paths.py",
        "assist.py",
        "menus.py",
        "runner.py",
        "screen_glance.py",
        "combat.py",
        "beam.py",
        "anchors.py",
        "room_timer.py",
        # Engine primitive beside combat.py: forward prediction against the
        # ROM itself. Root because every engine dir (walk, overworld,
        # dungeon) needs it and none of them may own it.
        "rollout.py",
        # The multi-segment search built on rollout.py. Root for the same
        # reason: a room solver is not a level's property and not an engine
        # dir's -- L7 binds it, L6 and L8 are meant to, and none may own it.
        "solver.py",
    }
)
_ENGINE_DIRS = ("walk", "overworld", "dungeon", "door_graph")
_ENGINE_ROOT_FILES = ("ram.py", "combat.py", "beam.py", "anchors.py", "rollout.py")
_LEVEL_EXITS_ALLOWLIST = frozenset(f"level{n}_exits.py" for n in range(1, 10))
_GONE_FLAT_MODULES = (
    "zelda_i.level6_hops",
    "zelda_i.walk_physics",
    "zelda_i.dungeon_ids",
    "zelda_i.survival_spine",
    "zelda_i.ow_path",
    "zelda_i.overworld_nav",
    "zelda_i.spine_hops",
    "zelda_i.hop_controller",
)
_NESTED_DEF = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _py_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    )


def _engine_source_files() -> tuple[Path, ...]:
    files = [_PKG_ROOT / name for name in _ENGINE_ROOT_FILES]
    for dirname in _ENGINE_DIRS:
        files.extend(_py_files(_PKG_ROOT / dirname))
    return tuple(files)


def _is_allowlisted_level_exits(path: Path) -> bool:
    return path.parent.name == "door_graph" and path.name in _LEVEL_EXITS_ALLOWLIST


def _module_name_for(path: Path) -> str:
    rel = path.relative_to(_PKG_ROOT)
    parts = ["zelda_i", *rel.with_suffix("").parts]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_level_import_nodes(tree: ast.AST) -> tuple[ast.AST, ...]:
    found: list[ast.AST] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _NESTED_DEF):
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                found.append(child)
            else:
                walk(child)

    walk(tree)
    return tuple(found)


def _resolve_from_module(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    parts = _module_name_for(path).split(".")
    pkg = parts[:-1]
    up = node.level - 1
    if up:
        pkg = pkg[:-up] if up < len(pkg) else []
    if node.module:
        return ".".join((*pkg, *node.module.split("."))) if pkg else node.module
    return ".".join(pkg)


def _imported_module_names(path: Path, node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()
    base = _resolve_from_module(path, node)
    names = [base] if base else []
    if base == "zelda_i" or node.module == "zelda_i":
        for alias in node.names:
            if alias.name != "*":
                names.append(f"zelda_i.{alias.name}")
    return tuple(names)


def _is_level_module(name: str) -> bool:
    for n in range(1, 10):
        prefix = f"zelda_i.level{n}"
        if name == prefix or name.startswith(prefix + "."):
            return True
    return False


def test_anchors_are_single_source() -> None:
    assert SCREEN_LEVEL3_ENTRANCE == 0x74
    assert SCREEN_LEVEL5_DOOR is SCREEN_LEVEL5_ENTRANCE
    assert TRIFORCE_BITS_BY_LEVEL[3] == TF_BIT_L3 == 0x04
    assert ENTRANCES[3].verified
    assert ENTRANCES[4].verified  # rr-0fx live entry
    assert ENTRANCES[4].entry_room == 0x71
    assert ENTRANCES[7].verified  # pond drain + L7 play 0x79 (recon whistle poke)
    assert ENTRANCES[7].entry_room == 0x79


def test_door_graph_stands_match_puzzle_catalog() -> None:
    assert _BOMB_STAND_6F_N == BOMB_WALL_6F_NORTH.stand


def test_dungeon_ids_has_l4_l5_enemy_types() -> None:
    from zelda_i.dungeon import ids as ids

    assert ids.LIKE_LIKE_OBJECT_TYPE == 0x17
    assert ids.POLS_VOICE_OBJECT_TYPE == 0x16
    assert ids.GIBDO_OBJECT_TYPE == 0x30
    assert ids.BUBBLE_OBJECT_TYPE == 0x40
    assert ids.GORIYA_BOOMERANG_OBJECT_TYPE == 0x5C
    assert ids.OBJECT_NAMES[0x5C] == "boomerang_projectile"


def test_root_is_thin() -> None:
    found = {path.name for path in _PKG_ROOT.glob("*.py")}
    assert found == _ALLOWED_ROOT_PY


def test_engines_do_not_import_level_packages_at_module_level() -> None:
    bad: list[str] = []
    for path in _engine_source_files():
        if _is_allowlisted_level_exits(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_PKG_ROOT)
        for node in _module_level_import_nodes(tree):
            for name in _imported_module_names(path, node):
                if _is_level_module(name):
                    bad.append(f"{rel}:{node.lineno} {name}")
    assert not bad, "engine imported level package at module level:\n  " + "\n  ".join(
        bad
    )


def test_dungeon_engine_level_specs_are_lazy() -> None:
    path = _PKG_ROOT / "dungeon" / "engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad: list[str] = []
    lazy: list[str] = []
    for node in tree.body:
        if not isinstance(node, _NESTED_DEF):
            continue
        for child in ast.walk(node):
            if not isinstance(child, (ast.Import, ast.ImportFrom)):
                continue
            for name in _imported_module_names(path, child):
                if not _is_level_module(name):
                    continue
                loc = f"{node.name}:{child.lineno} {name}"
                if node.name == "ensure_default_specs":
                    lazy.append(name)
                else:
                    bad.append(loc)
    assert not bad, "level imports outside ensure_default_specs:\n  " + "\n  ".join(
        bad
    )
    assert lazy, "ensure_default_specs should import zelda_i.levelN.dungeon"


def test_no_leftover_flat_modules() -> None:
    lingering = [
        name
        for name in _GONE_FLAT_MODULES
        if importlib.util.find_spec(name) is not None
    ]
    assert not lingering, "flat modules still importable: " + ", ".join(lingering)
