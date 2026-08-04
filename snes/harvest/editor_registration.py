"""Editor launcher registration for Harvest Moon."""

from __future__ import annotations

from pathlib import Path

# Loaded by ``retro_harness.editor_registry`` discovery. Keep this free of
# imports from editor_registry to avoid circular import at discovery time.
EDITOR_PROJECT = {
    "project_id": "harvest",
    "display_name": "Harvest Moon",
    "project_root": Path(__file__).resolve().parent,
    "editor_module": "harvest.tools.editor_app",
    "bridge_module": "harvest.runtime.editor_bridge",
    "description": (
        "ROM-first map editor with live emulator overlay and state patching."
    ),
}
