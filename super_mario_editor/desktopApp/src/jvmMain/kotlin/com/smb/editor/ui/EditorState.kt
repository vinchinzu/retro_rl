package com.smb.editor.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.smb.editor.data.LevelEdit
import com.smb.editor.rom.TileOverlay

enum class EditorTool {
    SELECT,
    PLACE_OBJECT,
    PLACE_ENEMY,
    ERASE
}

class EditorState {
    var selectedObjectIndex: Int? by mutableStateOf(null)
    var selectedEnemyIndex: Int? by mutableStateOf(null)
    var tool: EditorTool by mutableStateOf(EditorTool.SELECT)
    var zoom: Float by mutableStateOf(2.0f)
    var scrollX: Float by mutableStateOf(0f)
    var scrollY: Float by mutableStateOf(0f)
    var overlays: Set<TileOverlay> by mutableStateOf(setOf(TileOverlay.GRID))

    // Object placement
    var placementObjectType: Int? by mutableStateOf(null)
    var placementEnemyType: Int? by mutableStateOf(null)

    // Undo/redo
    private val undoStack = mutableListOf<LevelEdit>()
    private val redoStack = mutableListOf<LevelEdit>()

    val canUndo: Boolean get() = undoStack.isNotEmpty()
    val canRedo: Boolean get() = redoStack.isNotEmpty()

    fun applyEdit(edit: LevelEdit) {
        undoStack.add(edit)
        redoStack.clear()
    }

    fun undo(): LevelEdit? {
        if (undoStack.isEmpty()) return null
        val edit = undoStack.removeAt(undoStack.lastIndex)
        redoStack.add(edit)
        return edit
    }

    fun redo(): LevelEdit? {
        if (redoStack.isEmpty()) return null
        val edit = redoStack.removeAt(redoStack.lastIndex)
        undoStack.add(edit)
        return edit
    }

    fun toggleOverlay(overlay: TileOverlay) {
        overlays = if (overlay in overlays) overlays - overlay else overlays + overlay
    }

    fun clearSelection() {
        selectedObjectIndex = null
        selectedEnemyIndex = null
    }
}
