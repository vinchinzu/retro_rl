"""Non-ROM layer rules for KPDR hops vs combat vs skills.

See ``docs/ARCHITECTURE.md`` “Where a new file goes”.
"""

from __future__ import annotations

import ast
from pathlib import Path

import super_metroid.combat as combat_pkg

_COMBAT_ROOT = Path(combat_pkg.__file__).resolve().parent
_ALLOWED_KPDR_FROM_COMBAT = frozenset(
    {
        "super_metroid.routes.kpdr.room_ids",
    }
)


def _py_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts))


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return tuple(found)


def test_combat_does_not_import_kpdr_hop_modules() -> None:
    """Combat may read cycle-free ROOM_* from room_ids, not hop controllers."""
    bad: list[str] = []
    for path in _py_files(_COMBAT_ROOT):
        for module in _imported_modules(path):
            if not module.startswith("super_metroid.routes.kpdr"):
                continue
            if module in _ALLOWED_KPDR_FROM_COMBAT:
                continue
            rel = path.relative_to(_COMBAT_ROOT)
            bad.append(f"{rel}: {module}")
    assert not bad, "combat imported hop modules:\n  " + "\n  ".join(bad)


def test_pickup_scan_lives_in_primitives() -> None:
    from super_metroid.combat.primitives import Pickup, list_pickups
    from super_metroid.combat.spore_spawn import (
        Pickup as SporePickup,
        list_pickups as spore_list_pickups,
    )

    assert Pickup is SporePickup
    assert list_pickups is spore_list_pickups
    src = (_COMBAT_ROOT / "spore_spawn.py").read_text(encoding="utf-8")
    assert "def list_pickups" not in src
