"""Registry of game editors orchestrated by ``retro_harness.editor_launcher``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    return Path(__file__).resolve().parent.parent


def registered_editor_projects() -> tuple[EditorProject, ...]:
    """Return all editors known to the shared launcher."""

    root = monorepo_root()
    harvest_root = root / "harvest"
    earthbound_root = root / "earthbound"
    return (
        EditorProject(
            project_id="harvest",
            display_name="Harvest Moon",
            project_root=harvest_root,
            editor_module="harvest.tools.editor_app",
            bridge_module="harvest.runtime.editor_bridge",
            description="ROM-first map editor with live emulator overlay and state patching.",
        ),
        EditorProject(
            project_id="earthbound",
            display_name="EarthBound",
            project_root=earthbound_root,
            editor_module="earthbound_editor.__main__",
            bridge_module="earthbound_editor.editor_bridge",
            description="ROM-first map editor with live emulator overlay, NPC tracking, and dialogue scripting.",
        ),
    )


def get_editor_project(project_id: str) -> EditorProject:
    normalized = project_id.strip().casefold()
    for project in registered_editor_projects():
        if project.project_id.casefold() == normalized:
            return project
    known = ", ".join(item.project_id for item in registered_editor_projects())
    raise KeyError(f"Unknown editor project {project_id!r}. Known: {known}")
