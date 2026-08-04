"""Registry of game editors orchestrated by ``retro_harness.editor_launcher``.

Projects are discovered from ``snes/*/editor_registration.py`` and
``nes/*/editor_registration.py``. Each module may expose:

- ``EDITOR_PROJECT``: an :class:`EditorProject`, a mapping of its fields, or a
  zero-arg callable returning either
- ``editor_project()``: zero-arg callable returning the same shapes

Only projects whose ``project_root`` exists are registered.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class EditorProject:
    """One installable editor entry in the monorepo."""

    project_id: str
    display_name: str
    project_root: Path
    editor_module: str
    bridge_module: str
    description: str = ""


def monorepo_root() -> Path:
    from retro_harness.repo import monorepo_root as _root

    return _root()


def _load_registration_module(path: Path) -> ModuleType | None:
    """Import a registration file without requiring package layout."""
    module_name = f"_retro_rl_editor_reg_{path.parent.name}_{path.stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        return None
    return module


def _coerce_editor_project(raw: Any) -> EditorProject | None:
    """Normalize registration payloads into :class:`EditorProject`."""
    if raw is None:
        return None
    # Zero-arg factory (function / lambda). Skip classes (e.g. EditorProject).
    if callable(raw) and not isinstance(raw, type) and not isinstance(
        raw, EditorProject
    ):
        raw = raw()
    if isinstance(raw, EditorProject):
        return raw
    if not isinstance(raw, dict):
        return None
    try:
        project_root = Path(raw["project_root"])
        return EditorProject(
            project_id=str(raw["project_id"]),
            display_name=str(raw["display_name"]),
            project_root=project_root,
            editor_module=str(raw["editor_module"]),
            bridge_module=str(raw["bridge_module"]),
            description=str(raw.get("description", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _discover_editor_projects(root: Path) -> list[EditorProject]:
    found: list[EditorProject] = []
    seen_ids: set[str] = set()
    for console in ("snes", "nes"):
        console_dir = root / console
        if not console_dir.is_dir():
            continue
        for game_dir in sorted(console_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            reg_path = game_dir / "editor_registration.py"
            if not reg_path.is_file():
                continue
            module = _load_registration_module(reg_path)
            if module is None:
                continue
            raw = getattr(module, "EDITOR_PROJECT", None)
            if raw is None and callable(getattr(module, "editor_project", None)):
                raw = module.editor_project
            project = _coerce_editor_project(raw)
            if project is None:
                continue
            if not project.project_root.exists():
                continue
            key = project.project_id.casefold()
            if key in seen_ids:
                continue
            seen_ids.add(key)
            found.append(project)
    return found


def registered_editor_projects() -> tuple[EditorProject, ...]:
    """Return all editors known to the shared launcher."""
    return tuple(_discover_editor_projects(monorepo_root()))


def get_editor_project(project_id: str) -> EditorProject:
    normalized = project_id.strip().casefold()
    for project in registered_editor_projects():
        if project.project_id.casefold() == normalized:
            return project
    known = ", ".join(item.project_id for item in registered_editor_projects())
    raise KeyError(f"Unknown editor project {project_id!r}. Known: {known}")
