"""Dock panels for the Harvest Moon map editor."""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from harvest.core.harvest_state import HarvestStateDocument, WEATHER_CODES
from harvest.maps.map_config import ROUTES
from harvest.planner.day_plan_decision import auto_day_plan_decision
from harvest.tools.editor_canvas import (
    ADDR_MAP,
    BUILDING_TILES,
    CROP_TILES,
    DEBRIS_TILES,
    DOOR_CANDIDATE_TILES,
    GRASS_TILES,
    MAP_WIDTH,
    RENDER_MODE_ATLAS,
    RENDER_MODE_EXACT,
    STRUCTURE_TILES,
    TILE_NAMES,
    TileMapCanvas,
    WALKABLE_TILES,
    WATER_TILES,
    _is_walkable,
    _tile_color_rgb,
)


# ---------------------------------------------------------------------------
# Tile Info Panel
# ---------------------------------------------------------------------------

class TileInfoPanel(QWidget):
    """Shows selected tile info and color legend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._info_label = QLabel("Click a tile for details")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #ccc; font-size: 12px; padding: 4px;")
        layout.addWidget(self._info_label)

        legend_label = QLabel("Legend:")
        legend_label.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold; margin-top: 8px;")
        layout.addWidget(legend_label)

        legend_items = [
            (0x00, "Empty ground"), (0x01, "Untilled soil"), (0x02, "Tilled"),
            (0x08, "Watered"), (0x03, "Weed"), (0x04, "Stone"), (0x05, "Fence"),
            (0x06, "Rock"), (0x70, "Planted grass"), (0x80, "Mature grass"),
            (0x1E, "Crop"), (0xA0, "Path"), (0xA1, "Structure"), (0xA6, "Pond"),
            (0xC1, "Building"), (0xF0, "Water"), (0xFF, "Wall"),
        ]
        for tile_id, name in legend_items:
            r, g, b = _tile_color_rgb(tile_id)
            item = QLabel(f"  {name}")
            item.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            item.setStyleSheet(
                f"color: #ccc; font-size: 10px; padding: 1px 4px; "
                f"border-left: 8px solid rgb({r},{g},{b});"
            )
            layout.addWidget(item)
        layout.addStretch()

    def show_tile_info(self, tx: int, ty: int, tile_id: int, tilemap_id: int | None = None):
        name = TILE_NAMES.get(tile_id, "unknown")
        if tilemap_id is None:
            walkable = "Yes" if tile_id in WALKABLE_TILES else "No"
        else:
            walkable = "Yes" if _is_walkable(tilemap_id, tile_id) else "No"
        debris = "Yes" if tile_id in DEBRIS_TILES else "No"
        doorish = "Yes" if tile_id in DOOR_CANDIDATE_TILES else "No"
        self._info_label.setText(
            f"Tile ({tx}, {ty})\n"
            f"ID: 0x{tile_id:02X} ({name})\n"
            f"Walkable: {walkable}\n"
            f"Debris: {debris}\n"
            f"Door/Exit Candidate: {doorish}"
        )


class LayerControlsPanel(QWidget):
    def __init__(self, canvas: TileMapCanvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        hint = QLabel("Useful map layers")
        hint.setStyleSheet("color: #ddd; font-size: 12px; font-weight: bold;")
        layout.addWidget(hint)

        render_label = QLabel("Base render")
        render_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(render_label)

        self._render_mode = QComboBox()
        self._render_mode.addItem("Exact observed pixels", RENDER_MODE_EXACT)
        self._render_mode.addItem("ROM atlas (debug only)", RENDER_MODE_ATLAS)
        current = canvas.render_mode()
        self._render_mode.setCurrentIndex(0 if current == RENDER_MODE_EXACT else 1)
        self._render_mode.currentIndexChanged.connect(self._on_render_mode_changed)
        layout.addWidget(self._render_mode)

        self._doors = QCheckBox("Doors / transitions")
        self._doors.setChecked(canvas.doors_overlay_enabled())
        self._doors.toggled.connect(canvas.set_doors_overlay_enabled)
        layout.addWidget(self._doors)

        self._collision = QCheckBox("Collision / blocked tiles")
        self._collision.setChecked(canvas.collision_overlay_enabled())
        self._collision.toggled.connect(canvas.set_collision_overlay_enabled)
        layout.addWidget(self._collision)

        self._clamp = QCheckBox("Sprite clamp bounds")
        self._clamp.setChecked(canvas.clamp_overlay_enabled())
        self._clamp.toggled.connect(canvas.set_clamp_overlay_enabled)
        layout.addWidget(self._clamp)

        self._sprites = QCheckBox("Sprite delta (live only)")
        self._sprites.setChecked(canvas.sprite_delta_enabled())
        self._sprites.toggled.connect(canvas.set_sprite_delta_enabled)
        layout.addWidget(self._sprites)

        self._player = QCheckBox("Player marker")
        self._player.setChecked(canvas.player_marker_enabled())
        self._player.toggled.connect(canvas.set_player_marker_enabled)
        layout.addWidget(self._player)

        self._entities = QCheckBox("Game objects / NPCs")
        self._entities.setChecked(canvas.entities_overlay_enabled())
        self._entities.toggled.connect(canvas.set_entities_overlay_enabled)
        layout.addWidget(self._entities)

        self._live = QCheckBox("Live viewport overlay")
        self._live.setChecked(canvas.live_overlay_enabled())
        self._live.toggled.connect(canvas.set_live_overlay_enabled)
        layout.addWidget(self._live)

        self._route = QCheckBox("Route waypoints")
        self._route.setChecked(canvas.route_overlay_enabled())
        self._route.toggled.connect(self._on_route_overlay_toggled)
        layout.addWidget(self._route)

        self._route_combo = QComboBox()
        self._route_combo.addItem("None", "")
        for route_name in sorted(ROUTES):
            self._route_combo.addItem(route_name, route_name)
        self._route_combo.setEnabled(self._route.isChecked())
        self._route_combo.currentIndexChanged.connect(self._on_route_changed)
        layout.addWidget(self._route_combo)

        note = QLabel(
            "Exact mode draws observed pixels (from ROM render or emulator). "
            "ROM atlas mode renders each tile individually from the atlas. "
            "Doors show known cross-map exits plus door-like tiles. "
            "Routes use map_config waypoints; game objects come from WRAM."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(note)
        layout.addStretch()

    def _on_render_mode_changed(self, _index: int) -> None:
        mode = self._render_mode.currentData()
        self._canvas.set_render_mode(mode)

    def _on_route_overlay_toggled(self, enabled: bool) -> None:
        self._route_combo.setEnabled(enabled)
        self._canvas.set_route_overlay_enabled(enabled)
        if enabled:
            self._canvas.set_route_overlay(str(self._route_combo.currentData() or ""))

    def _on_route_changed(self, _index: int) -> None:
        if self._route.isChecked():
            self._canvas.set_route_overlay(str(self._route_combo.currentData() or ""))

    def set_live_overlay_checked(self, enabled: bool) -> None:
        if self._live.isChecked() == enabled:
            self._canvas.set_live_overlay_enabled(enabled)
            return
        self._live.blockSignals(True)
        self._live.setChecked(enabled)
        self._live.blockSignals(False)
        self._canvas.set_live_overlay_enabled(enabled)


# ---------------------------------------------------------------------------
# Tile Stats Panel
# ---------------------------------------------------------------------------

class TileStatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._tree = QTreeWidget()
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tree.setHeaderLabels(["Category", "Count"])
        self._tree.setColumnWidth(0, 120)
        layout.addWidget(self._tree)

    def update_from_ram(self, ram: np.ndarray):
        self._tree.clear()
        tile_data = ram[ADDR_MAP:ADDR_MAP + MAP_WIDTH * MAP_WIDTH]
        categories: dict[str, int] = {}
        for tid in tile_data:
            cat = self._categorize(int(tid))
            categories[cat] = categories.get(cat, 0) + 1
        for cat in sorted(categories, key=lambda c: -categories[c]):
            QTreeWidgetItem(self._tree, [cat, str(categories[cat])])

    @staticmethod
    def _categorize(tile_id: int) -> str:
        if tile_id in DEBRIS_TILES:    return "Debris"
        if tile_id in CROP_TILES:      return "Crop"
        if tile_id in GRASS_TILES:     return "Grass"
        if tile_id in WATER_TILES:     return "Water"
        if tile_id in BUILDING_TILES:  return "Building"
        if tile_id in STRUCTURE_TILES: return "Structure"
        if tile_id in (0xA0, 0xA2, 0xA3): return "Path"
        if tile_id == 0xFF:            return "Wall"
        if tile_id in (0x01, 0x02, 0x07, 0x08): return "Farmland"
        if tile_id == 0x00:            return "Empty"
        return "Other"


# ---------------------------------------------------------------------------
# Day Plan Preview Panel
# ---------------------------------------------------------------------------

class PlanPreviewPanel(QWidget):
    """Read-only RAM-backed day-plan preview for editor validation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._summary = QLabel("Load a snapshot or start the emulator to preview the day plan.")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #aaa; font-size: 11px; padding: 4px;")
        layout.addWidget(self._summary)

        self._tree = QTreeWidget()
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tree.setHeaderLabels(["Item", "Kind", "Detail"])
        self._tree.setColumnWidth(0, 170)
        self._tree.setColumnWidth(1, 90)
        layout.addWidget(self._tree, 1)

    def update_from_ram(self, ram: np.ndarray | None, *, state_name: str | None = None) -> None:
        self._tree.clear()
        if ram is None:
            self._summary.setText("No RAM snapshot available.")
            return
        try:
            decision = auto_day_plan_decision(state_name=state_name, ram=ram)
        except Exception as exc:
            self._summary.setText(f"Plan preview failed: {exc}")
            return

        facts = decision.facts.to_jsonable()
        hour = int(facts.get("hour") or 0)
        minute = int(facts.get("minute") or 0)
        time_text = f"{hour:02}:{minute:02}"
        map_text = facts.get("tilemap")
        map_label = f"0x{int(map_text):02X}" if map_text is not None else "--"
        self._summary.setText(
            f"{len(decision.phases)} phases from {facts.get('source')} facts | "
            f"time {time_text} | map {map_label}"
        )

        phases_root = QTreeWidgetItem(self._tree, ["Phases", "", ""])
        for index, phase in enumerate(decision.phases, start=1):
            detail = self._phase_detail(phase.params)
            item = QTreeWidgetItem(phases_root, [f"{index:02d}. {phase.phase}", phase.kind, detail])
            if phase.failure_policy != "required":
                item.setText(2, f"{detail} | {phase.failure_policy}" if detail else phase.failure_policy)

        if decision.deferred:
            deferred_root = QTreeWidgetItem(self._tree, ["Deferred", "", ""])
            for item in decision.deferred:
                QTreeWidgetItem(
                    deferred_root,
                    [item.phase, item.kind, f"{item.reason} -> {item.retry}"],
                )

        if decision.notes:
            notes_root = QTreeWidgetItem(self._tree, ["Notes", "", ""])
            for note in decision.notes:
                QTreeWidgetItem(notes_root, [str(note), "", ""])

        facts_root = QTreeWidgetItem(self._tree, ["Facts", "", ""])
        for key in sorted(facts):
            QTreeWidgetItem(facts_root, [key, "", str(facts[key])])
        self._tree.expandToDepth(1)

    @staticmethod
    def _phase_detail(params: dict) -> str:
        if not params:
            return ""
        if route := params.get("route"):
            return f"route={route}"
        if target := params.get("target_px"):
            return f"target={tuple(target)}"
        if task := params.get("task_name"):
            return f"task={task}"
        if recording := params.get("recording_name"):
            return f"recording={recording}"
        if direction := params.get("direction"):
            return f"direction={direction}"
        return ", ".join(f"{key}={value}" for key, value in sorted(params.items())[:3])


# ---------------------------------------------------------------------------
# State Editor Panel
# ---------------------------------------------------------------------------

class StateEditorPanel(QWidget):
    """Editable snapshot fields, source-tagged by confidence."""

    state_changed = Signal()
    load_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._document: HarvestStateDocument | None = None
        self._selected_tile: tuple[int, int] | None = None
        self._loading_tree = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._hint = QLabel("Load a snapshot to edit persistent state.")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #aaa; font-size: 11px; padding: 4px;")
        layout.addWidget(self._hint)

        actions = QHBoxLayout()
        self._load_button = QPushButton("Load Snapshot")
        self._load_button.clicked.connect(self.load_requested.emit)
        actions.addWidget(self._load_button)
        self._save_button = QPushButton("Save Patched State")
        self._save_button.clicked.connect(self.save_requested.emit)
        self._save_button.setEnabled(False)
        actions.addWidget(self._save_button)
        layout.addLayout(actions)

        self._tree = QTreeWidget()
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tree.setHeaderLabels(["Field", "Value", "Source"])
        self._tree.setColumnWidth(0, 180)
        self._tree.setColumnWidth(1, 90)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree, 1)

    def set_document(self, document: HarvestStateDocument | None) -> None:
        self._document = document
        if document is None:
            self._hint.setText(
                "Live emulator sessions are view-only here. Select a snapshot in the "
                "emulator panel, click Load Snapshot, edit values, then save a patched state."
            )
            self._selected_tile = None
            self._save_button.setEnabled(False)
        else:
            self._hint.setText(
                "Sources: state=validated from local state diffs, retro=stable-retro metadata, "
                "decomp=provisional. Edit values directly, then save a patched state "
                "to a new *_edited.state file."
            )
            self._save_button.setEnabled(True)
        self._rebuild_tree()

    def select_tile(self, tx: int, ty: int) -> None:
        self._selected_tile = (tx, ty)
        if self._document is not None:
            self._rebuild_tree()

    @property
    def selected_tile(self) -> tuple[int, int] | None:
        return self._selected_tile

    def _format_scalar_value(self, key: str, value: int) -> str:
        if key == "weather_tomorrow":
            return str(value)
        return str(value)

    def _add_item(
        self,
        parent: QTreeWidgetItem,
        label: str,
        value: str,
        source: str,
        payload: tuple[object, ...] | None = None,
        *,
        editable: bool = False,
        tooltip: str | None = None,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [label, value, source])
        if editable:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        if payload is not None:
            item.setData(0, Qt.ItemDataRole.UserRole, payload)
        if tooltip:
            item.setToolTip(0, tooltip)
            item.setToolTip(1, tooltip)
            item.setToolTip(2, tooltip)
        return item

    def _rebuild_tree(self) -> None:
        self._loading_tree = True
        self._tree.clear()
        if self._document is None:
            self._loading_tree = False
            return

        sections: dict[str, QTreeWidgetItem] = {}
        for spec in self._document.scalar_fields():
            section = sections.get(spec.section)
            if section is None:
                section = QTreeWidgetItem(self._tree, [spec.section, "", ""])
                sections[spec.section] = section
            value = self._document.scalar_value(spec.key)
            tooltip = spec.note
            if spec.key == "weather_tomorrow":
                tooltip = f"{spec.note} Known codes: {', '.join(WEATHER_CODES.values())}"
            self._add_item(
                section,
                spec.label,
                self._format_scalar_value(spec.key, value),
                spec.source,
                ("scalar", spec.key),
                editable=True,
                tooltip=tooltip or None,
            )

        tile_root = QTreeWidgetItem(self._tree, ["Selected Tile", "", ""])
        if self._selected_tile is None:
            self._add_item(tile_root, "Status", "Click a map tile", "state")
        else:
            tx, ty = self._selected_tile
            tile = self._document.farm_tile(tx, ty)
            self._add_item(tile_root, "Coords", f"({tx}, {ty})", tile.source)
            self._add_item(
                tile_root,
                "Persistent Tile",
                f"0x{tile.persistent_value:02X}",
                tile.source,
                ("farm_tile", tx, ty),
                editable=True,
                tooltip="Persistent farm-state byte. This is the value saved back into the snapshot.",
            )
            self._add_item(
                tile_root,
                "Visible Tile",
                f"0x{tile.visible_value:02X}",
                "state",
                tooltip="Current rendered tile in the active map buffer.",
            )

        cows_root = QTreeWidgetItem(self._tree, ["Cow Slots", "", ""])
        for cow in self._document.cows():
            cow_item = QTreeWidgetItem(cows_root, [f"Cow {cow.slot + 1:02d}", "", cow.source])
            self._add_item(cow_item, "Status Raw", f"0x{cow.status_raw:02X}", cow.source, ("cow", cow.slot, "status_raw"), editable=True)
            self._add_item(cow_item, "Raw 1", f"0x{cow.raw_1:02X}", cow.source, ("cow", cow.slot, "raw_1"), editable=True)
            self._add_item(cow_item, "Home Map Raw", f"0x{cow.home_map_raw:02X}", cow.source, ("cow", cow.slot, "home_map_raw"), editable=True)
            self._add_item(cow_item, "Pregnancy Raw", f"0x{cow.pregnancy_raw:02X}", cow.source, ("cow", cow.slot, "pregnancy_raw"), editable=True)
            self._add_item(cow_item, "Happiness", str(cow.happiness), cow.source, ("cow", cow.slot, "happiness"), editable=True)
            self._add_item(cow_item, "Raw 5", f"0x{cow.raw_5:02X}", cow.source, ("cow", cow.slot, "raw_5"), editable=True)
            self._add_item(cow_item, "Pos X", str(cow.position_x), cow.source, ("cow", cow.slot, "position_x"), editable=True)
            self._add_item(cow_item, "Pos Y", str(cow.position_y), cow.source, ("cow", cow.slot, "position_y"), editable=True)

        chickens_root = QTreeWidgetItem(self._tree, ["Chicken Slots", "", ""])
        for chicken in self._document.chickens():
            chicken_item = QTreeWidgetItem(chickens_root, [f"Chicken {chicken.slot + 1:02d}", "", chicken.source])
            self._add_item(chicken_item, "Status Raw", f"0x{chicken.status_raw:02X}", chicken.source, ("chicken", chicken.slot, "status_raw"), editable=True)
            self._add_item(chicken_item, "Raw 1", f"0x{chicken.raw_1:02X}", chicken.source, ("chicken", chicken.slot, "raw_1"), editable=True)
            self._add_item(chicken_item, "Raw 2", f"0x{chicken.raw_2:02X}", chicken.source, ("chicken", chicken.slot, "raw_2"), editable=True)
            self._add_item(chicken_item, "Raw 3", f"0x{chicken.raw_3:02X}", chicken.source, ("chicken", chicken.slot, "raw_3"), editable=True)
            self._add_item(chicken_item, "Pos X", str(chicken.position_x), chicken.source, ("chicken", chicken.slot, "position_x"), editable=True)
            self._add_item(chicken_item, "Pos Y", str(chicken.position_y), chicken.source, ("chicken", chicken.slot, "position_y"), editable=True)

        self._tree.expandToDepth(0)
        self._loading_tree = False

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._loading_tree or column != 1 or self._document is None:
            return

        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if payload is None:
            return

        text = item.text(1).strip()
        try:
            value = int(text, 0)
        except ValueError:
            self._rebuild_tree()
            return

        kind = payload[0]
        try:
            if kind == "scalar":
                self._document.set_scalar_value(payload[1], value)
            elif kind == "farm_tile":
                self._document.set_farm_tile_value(payload[1], payload[2], value)
            elif kind == "cow":
                self._document.set_cow_field(payload[1], payload[2], value)
            elif kind == "chicken":
                self._document.set_chicken_field(payload[1], payload[2], value)
            else:
                return
        except (KeyError, IndexError, ValueError):
            self._rebuild_tree()
            return

        self.state_changed.emit()

