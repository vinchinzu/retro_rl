"""Harvest Moon editor Cursor SDK agent dock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harvest.paths import MONOREPO_DIR, PROJECT_DIR
from retro_harness.editor.cursor_agent import EditorAgentContext, compact_snapshot
from retro_harness.editor.cursor_agent_panel import (
    CursorAgentPanelConfig,
    attach_cursor_agent_dock,
)

if TYPE_CHECKING:
    from harvest.tools.editor_app import EditorWindow

HARVEST_AGENT_INSTRUCTIONS = (
    "You are assisting with Harvest Moon SNES ROM extraction and modding inside retro_rl.",
    "Use live emulator snapshots, tilemap overlays, and save-state edits as ground truth.",
    "Prefer changes under harvest/ unless a helper clearly belongs in retro_harness/.",
    "When proposing WRAM or map work, cite tile IDs, map_config entries, or task recordings.",
)


def build_harvest_agent_context(
    *,
    state_name: str | None,
    selected_tile: tuple[int, int] | None,
    emulator_snapshot: dict[str, object] | None,
    tilemap_id: int | None,
    player_tile: tuple[int, int] | None,
    live_overlay_enabled: bool,
    state_document_name: str | None,
) -> EditorAgentContext | None:
    """Build structured editor context for a Cursor agent prompt."""

    summary_parts: list[str] = []
    details: dict[str, object] = {
        "game": "harvest",
        "project_root": str(PROJECT_DIR),
    }

    if state_name:
        details["state_name"] = state_name
        summary_parts.append(f"state `{state_name}`")

    if state_document_name:
        details["state_document"] = state_document_name
        summary_parts.append(f"edited `{state_document_name}`")

    if tilemap_id is not None:
        details["tilemap_id"] = tilemap_id
        summary_parts.append(f"map `0x{tilemap_id:02X}`")

    if player_tile is not None:
        details["player_tile"] = {"x": player_tile[0], "y": player_tile[1]}
        summary_parts.append(f"player ({player_tile[0]}, {player_tile[1]})")

    if selected_tile is not None:
        details["selected_tile"] = {"x": selected_tile[0], "y": selected_tile[1]}
        summary_parts.append(f"tile ({selected_tile[0]}, {selected_tile[1]})")

    details["live_overlay_enabled"] = live_overlay_enabled

    if emulator_snapshot:
        details["emulator_snapshot"] = compact_snapshot(emulator_snapshot)
        map_name = emulator_snapshot.get("mapName")
        if map_name:
            summary_parts.append(f"live `{map_name}`")

    if not summary_parts and not emulator_snapshot:
        return None

    summary = ", ".join(summary_parts) if summary_parts else "Live editor session"
    return EditorAgentContext(
        title="Harvest Moon Editor",
        summary=summary,
        details=details,
    )


def harvest_agent_context_from_window(window: EditorWindow) -> EditorAgentContext | None:
    """Collect live Harvest editor context from an ``EditorWindow`` instance."""

    tilemap_id = None
    player_tile = None
    if window._last_ram is not None:
        from harvest.tools.editor_app import TILE_PX, _get_pos, _get_tilemap_id

        tilemap_id = _get_tilemap_id(window._last_ram)
        px, py = _get_pos(window._last_ram)
        player_tile = (px // TILE_PX, py // TILE_PX)

    emulator_snapshot = window._emu_panel.last_snapshot()
    state_document_name = (
        window._state_doc.state_name if window._state_doc is not None else None
    )
    return build_harvest_agent_context(
        state_name=window._current_state_name or window._emu_panel.selected_state(),
        selected_tile=window._state_editor.selected_tile,
        emulator_snapshot=emulator_snapshot,
        tilemap_id=tilemap_id,
        player_tile=player_tile,
        live_overlay_enabled=window._canvas.live_overlay_enabled(),
        state_document_name=state_document_name,
    )


def attach_harvest_agent_dock(window: EditorWindow) -> None:
    """Add the shared Cursor SDK agent dock to a Harvest editor window."""

    config = CursorAgentPanelConfig(
        workspace_cwd=MONOREPO_DIR,
        settings_org="retro-rl",
        settings_app="harvest-editor",
        session_name="harvest-editor",
        instructions=HARVEST_AGENT_INSTRUCTIONS,
    )
    window.agent_dock, window.cursor_agent_panel = attach_cursor_agent_dock(
        window,
        config=config,
        context_provider=lambda: harvest_agent_context_from_window(window),
    )
    window.agent_dock.hide()
