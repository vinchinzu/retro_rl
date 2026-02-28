package com.smb.editor.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import java.awt.event.MouseEvent
import com.smb.editor.data.*
import com.smb.editor.rom.*

@Composable
fun MapCanvas(
    rom: NesRom?,
    levelData: LevelData?,
    editorState: EditorState
) {
    if (rom == null || levelData == null) {
        Box(
            modifier = Modifier.fillMaxSize().background(Color(0xFF1A1A2E)),
            contentAlignment = Alignment.Center
        ) {
            Text(
                "Select a level to view",
                color = Color.White.copy(alpha = 0.5f)
            )
        }
        return
    }

    // Pre-render to pixel data for display
    val renderResult = remember(levelData) {
        LevelRenderer.renderLevel(rom, levelData)
    }

    val zoom = editorState.zoom
    val overlays = editorState.overlays
    val semanticOverlays = remember(overlays) {
        overlays.intersect(
            setOf(
                TileOverlay.COINS,
                TileOverlay.POWERUPS,
                TileOverlay.PIPES,
                TileOverlay.WARP_ZONES
            )
        )
    }
    val semanticOverlayItems = remember(levelData, semanticOverlays) {
        OverlayGenerator.generate(levelData, semanticOverlays)
    }

    Canvas(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF1A1A2E))
            .pointerInput(Unit) {
                detectTapGestures { offset ->
                    // Convert screen coords to level coords
                    val levelX = ((offset.x - editorState.scrollX) / zoom).toInt()
                    val levelY = ((offset.y - editorState.scrollY) / zoom).toInt()

                    // Find nearest object or enemy
                    val metatileCol = levelX / 16
                    val metatileRow = levelY / 16

                    // Check enemies first
                    val enemyIdx = levelData.enemies.indexOfFirst { enemy ->
                        val ex = enemy.page * 16 + enemy.col
                        val ey = enemy.row
                        ex == metatileCol && ey == metatileRow
                    }

                    if (enemyIdx >= 0) {
                        editorState.selectedObjectIndex = null
                        editorState.selectedEnemyIndex = enemyIdx
                        return@detectTapGestures
                    }

                    // Check objects
                    val objIdx = levelData.objects.indexOfFirst { obj ->
                        val ox = obj.page * 16 + obj.col
                        val oy = obj.row
                        ox == metatileCol && oy == metatileRow
                    }

                    if (objIdx >= 0) {
                        editorState.selectedEnemyIndex = null
                        editorState.selectedObjectIndex = objIdx
                    } else {
                        editorState.clearSelection()
                    }
                }
            }
            .pointerInput(Unit) {
                awaitPointerEventScope {
                    while (true) {
                        val event = awaitPointerEvent()
                        if (event.type == PointerEventType.Scroll) {
                            val scrollDelta = event.changes.firstOrNull()?.scrollDelta ?: continue
                            val ne = event.nativeEvent as? MouseEvent
                            val ctrlHeld = ne?.let { it.isControlDown || it.isMetaDown } ?: false
                            if (ctrlHeld) {
                                val zoomDelta = if (scrollDelta.y < 0) 1.1f else 0.9f
                                editorState.zoom = (editorState.zoom * zoomDelta).coerceIn(0.5f, 8f)
                            } else {
                                editorState.scrollX -= scrollDelta.x * 20
                                editorState.scrollY -= scrollDelta.y * 20
                            }
                            event.changes.forEach { it.consume() }
                        }
                    }
                }
            }
    ) {
        val w = renderResult.width
        val h = renderResult.height
        if (w <= 0 || h <= 0) return@Canvas

        // Draw level pixels
        drawIntoCanvas { canvas ->
            val paint = Paint()
            paint.isAntiAlias = false

            // Scale and translate
            canvas.save()
            canvas.translate(editorState.scrollX, editorState.scrollY)
            canvas.scale(zoom, zoom)

            // Draw each pixel as a rect (simple approach for now)
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val color = renderResult.pixels[y * w + x]
                    if (color != 0) {
                        paint.color = Color(color)
                        canvas.drawRect(
                            x.toFloat(), y.toFloat(),
                            (x + 1).toFloat(), (y + 1).toFloat(),
                            paint
                        )
                    }
                }
            }

            canvas.restore()
        }

        // Draw overlays
        if (TileOverlay.GRID in overlays) {
            drawGrid(renderResult.width, renderResult.height, zoom, editorState.scrollX, editorState.scrollY)
        }

        if (semanticOverlayItems.isNotEmpty()) {
            drawSemanticOverlays(semanticOverlayItems, zoom, editorState)
        }

        if (TileOverlay.OBJECTS in overlays) {
            drawObjectOverlays(levelData, zoom, editorState)
        }

        if (TileOverlay.ENEMIES in overlays) {
            drawEnemyOverlays(levelData, zoom, editorState)
        }
    }
}

private fun DrawScope.drawSemanticOverlays(items: List<OverlayItem>, zoom: Float, state: EditorState) {
    for (item in items) {
        val x = item.x * zoom + state.scrollX
        val y = item.y * zoom + state.scrollY
        val w = item.width * zoom
        val h = item.height * zoom
        val color = Color(item.color)

        when (item.type) {
            OverlayType.COIN_HIGHLIGHT -> {
                drawCircle(
                    color = color,
                    radius = (minOf(w, h) / 2f).coerceAtLeast(2f),
                    center = Offset(x + w / 2f, y + h / 2f)
                )
            }

            OverlayType.POWERUP_HIGHLIGHT -> {
                drawRect(
                    color = color,
                    topLeft = Offset(x, y),
                    size = Size(w, h)
                )
            }

            OverlayType.PIPE_MARKER, OverlayType.WARP_ZONE -> {
                drawRect(
                    color = color,
                    topLeft = Offset(x, y),
                    size = Size(w, h)
                )
            }

            else -> Unit
        }
    }
}

private fun DrawScope.drawGrid(levelWidth: Int, levelHeight: Int, zoom: Float, scrollX: Float, scrollY: Float) {
    val gridColor = Color.White.copy(alpha = 0.1f)
    val pageColor = Color.Yellow.copy(alpha = 0.15f)

    // Block grid (every 16px)
    for (x in 0..levelWidth step 16) {
        val screenX = x * zoom + scrollX
        val color = if (x % 256 == 0) pageColor else gridColor
        drawLine(color, Offset(screenX, scrollY), Offset(screenX, levelHeight * zoom + scrollY), strokeWidth = 1f)
    }
    for (y in 0..levelHeight step 16) {
        val screenY = y * zoom + scrollY
        drawLine(gridColor, Offset(scrollX, screenY), Offset(levelWidth * zoom + scrollX, screenY), strokeWidth = 1f)
    }
}

private fun DrawScope.drawObjectOverlays(levelData: LevelData, zoom: Float, state: EditorState) {
    for ((i, obj) in levelData.objects.withIndex()) {
        val x = (obj.page * 16 + obj.col) * 16f * zoom + state.scrollX
        val y = obj.row * 16f * zoom + state.scrollY
        val size = 16f * zoom

        val isSelected = state.selectedObjectIndex == i
        val color = if (isSelected) Color.Yellow.copy(alpha = 0.6f) else Color.Green.copy(alpha = 0.3f)

        drawRect(
            color = color,
            topLeft = Offset(x, y),
            size = Size(size, size)
        )
    }
}

private fun DrawScope.drawEnemyOverlays(levelData: LevelData, zoom: Float, state: EditorState) {
    for ((i, enemy) in levelData.enemies.withIndex()) {
        val x = (enemy.page * 16 + enemy.col) * 16f * zoom + state.scrollX
        val y = enemy.row * 16f * zoom + state.scrollY
        val size = 16f * zoom

        val isSelected = state.selectedEnemyIndex == i
        val color = if (isSelected) Color.Yellow.copy(alpha = 0.6f) else Color.Red.copy(alpha = 0.4f)

        drawCircle(
            color = color,
            radius = size / 2,
            center = Offset(x + size / 2, y + size / 2)
        )
    }
}
