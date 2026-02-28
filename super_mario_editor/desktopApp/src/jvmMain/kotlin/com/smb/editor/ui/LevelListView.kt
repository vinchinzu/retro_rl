package com.smb.editor.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.smb.editor.data.LevelRegistry

// Area type colors
private val areaColors = mapOf(
    0 to Color(0xFF2196F3),  // Water - blue
    1 to Color(0xFF795548),  // Underground - brown
    2 to Color(0xFF757575),  // Castle - gray
    3 to Color(0xFF4CAF50)   // Overworld - green
)

@Composable
fun LevelListView(
    selectedLevel: String?,
    onSelect: (String) -> Unit
) {
    val levelsByWorld = remember {
        (1..8).map { world ->
            world to LevelRegistry.levelsByWorld(world)
        }
    }

    Surface(
        tonalElevation = 1.dp,
        modifier = Modifier.fillMaxSize()
    ) {
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(4.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            for ((world, levels) in levelsByWorld) {
                // World header
                item(key = "world-$world") {
                    Text(
                        text = "World $world",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp),
                        color = MaterialTheme.colorScheme.primary
                    )
                }

                // Levels in this world
                items(levels, key = { it.worldLevel }) { entry ->
                    val isSelected = entry.worldLevel == selectedLevel
                    val bgColor = if (isSelected) {
                        MaterialTheme.colorScheme.primaryContainer
                    } else {
                        Color.Transparent
                    }
                    val areaColor = areaColors[entry.areaType] ?: Color.Gray

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onSelect(entry.worldLevel) }
                            .background(bgColor)
                            .padding(horizontal = 8.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        // Area type color indicator
                        Box(
                            modifier = Modifier
                                .width(4.dp)
                                .height(24.dp)
                                .background(areaColor)
                        )

                        Column {
                            Text(
                                text = entry.worldLevel,
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                color = if (isSelected) {
                                    MaterialTheme.colorScheme.onPrimaryContainer
                                } else {
                                    MaterialTheme.colorScheme.onSurface
                                }
                            )
                            Text(
                                text = "${LevelRegistry.AREA_TYPE_NAMES[entry.areaType] ?: ""}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                            )
                        }
                    }
                }
            }
        }
    }
}
