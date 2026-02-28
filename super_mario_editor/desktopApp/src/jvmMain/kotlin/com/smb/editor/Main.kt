@file:OptIn(ExperimentalMaterial3Api::class)

package com.smb.editor

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.*
import com.smb.editor.data.*
import com.smb.editor.rom.*
import com.smb.editor.ui.*
import java.awt.FileDialog
import java.awt.Frame
import java.io.File

fun main() = application {
    val windowState = rememberWindowState(
        size = DpSize(1280.dp, 800.dp)
    )

    Window(
        onCloseRequest = ::exitApplication,
        title = "Super Mario Bros Editor",
        state = windowState
    ) {
        App()
    }
}

@Composable
fun App() {
    var rom by remember { mutableStateOf<NesRom?>(null) }
    var romPath by remember { mutableStateOf<String?>(null) }
    var selectedLevel by remember { mutableStateOf<String?>(null) }
    var levelData by remember { mutableStateOf<LevelData?>(null) }
    var editorState by remember { mutableStateOf(EditorState()) }

    // Decode level when selection changes
    LaunchedEffect(selectedLevel, rom) {
        if (rom != null && selectedLevel != null) {
            val entry = LevelRegistry.findByWorldLevel(selectedLevel!!)
            if (entry != null) {
                levelData = LevelDecoder.decodeLevel(rom!!, entry)
            }
        } else {
            levelData = null
        }
    }

    MaterialTheme(
        colorScheme = darkColorScheme()
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                // Top bar
                TopBar(
                    romPath = romPath,
                    onLoadRom = { path ->
                        try {
                            val data = File(path).readBytes()
                            rom = NesRom.fromBytes(data)
                            romPath = path
                            selectedLevel = "1-1"
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    },
                    editorState = editorState
                )

                Divider()

                // Main content
                if (rom == null) {
                    // No ROM loaded - show welcome
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            "Load a Super Mario Bros NES ROM to begin editing",
                            style = MaterialTheme.typography.titleLarge,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                        )
                    }
                } else {
                    // Three-column layout
                    var leftWidth by remember { mutableStateOf(220.dp) }
                    var rightWidth by remember { mutableStateOf(260.dp) }

                    Row(modifier = Modifier.fillMaxSize()) {
                        // Left panel: Level list
                        Box(modifier = Modifier.width(leftWidth).fillMaxHeight()) {
                            LevelListView(
                                selectedLevel = selectedLevel,
                                onSelect = { selectedLevel = it }
                            )
                        }

                        DraggableDividerVertical { delta ->
                            leftWidth = (leftWidth + delta.dp).coerceIn(150.dp, 400.dp)
                        }

                        // Center: Map canvas
                        Box(modifier = Modifier.weight(1f).fillMaxHeight()) {
                            MapCanvas(
                                rom = rom,
                                levelData = levelData,
                                editorState = editorState
                            )
                        }

                        DraggableDividerVertical { delta ->
                            rightWidth = (rightWidth - delta.dp).coerceIn(200.dp, 400.dp)
                        }

                        // Right panel: Object palette + Properties
                        Column(modifier = Modifier.width(rightWidth).fillMaxHeight()) {
                            ObjectPalette(
                                editorState = editorState,
                                modifier = Modifier.weight(1f)
                            )

                            Divider()

                            PropertyInspector(
                                editorState = editorState,
                                levelData = levelData,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun TopBar(
    romPath: String?,
    onLoadRom: (String) -> Unit,
    editorState: EditorState
) {
    Surface(
        tonalElevation = 2.dp
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Button(onClick = {
                val dialog = FileDialog(null as Frame?, "Open NES ROM", FileDialog.LOAD)
                dialog.setFilenameFilter { _, name -> name.endsWith(".nes") }
                dialog.isVisible = true
                if (dialog.file != null) {
                    onLoadRom(dialog.directory + dialog.file)
                }
            }) {
                Text("Load ROM")
            }

            if (romPath != null) {
                Text(
                    text = File(romPath).name,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            // Overlay toggles
            Text("Overlays:", style = MaterialTheme.typography.bodySmall)

            val overlayOrder = listOf(
                TileOverlay.GRID,
                TileOverlay.OBJECTS,
                TileOverlay.ENEMIES,
                TileOverlay.COINS,
                TileOverlay.POWERUPS,
                TileOverlay.PIPES,
                TileOverlay.WARP_ZONES
            )
            for (overlay in overlayOrder) {
                FilterChip(
                    selected = overlay in editorState.overlays,
                    onClick = { editorState.toggleOverlay(overlay) },
                    label = {
                        val label = when (overlay) {
                            TileOverlay.GRID -> "Grid"
                            TileOverlay.OBJECTS -> "Objects"
                            TileOverlay.ENEMIES -> "Enemies"
                            TileOverlay.COINS -> "Coins"
                            TileOverlay.POWERUPS -> "Powerups"
                            TileOverlay.PIPES -> "Pipes"
                            TileOverlay.WARP_ZONES -> "Warp Zones"
                        }
                        Text(label)
                    }
                )
            }

            Spacer(modifier = Modifier.width(16.dp))

            // Undo/Redo
            IconButton(
                onClick = { editorState.undo() },
                enabled = editorState.canUndo
            ) {
                Text("\u21B6")
            }
            IconButton(
                onClick = { editorState.redo() },
                enabled = editorState.canRedo
            ) {
                Text("\u21B7")
            }
        }
    }
}
