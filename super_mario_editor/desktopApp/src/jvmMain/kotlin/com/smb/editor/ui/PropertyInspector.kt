package com.smb.editor.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.smb.editor.data.*

@Composable
fun PropertyInspector(
    editorState: EditorState,
    levelData: LevelData?,
    modifier: Modifier = Modifier
) {
    Surface(
        tonalElevation = 1.dp,
        modifier = modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(8.dp)) {
            Text(
                "Properties",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            when {
                levelData == null -> {
                    Text(
                        "No level loaded",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                    )
                }

                editorState.selectedObjectIndex != null -> {
                    val idx = editorState.selectedObjectIndex!!
                    if (idx in levelData.objects.indices) {
                        val obj = levelData.objects[idx]
                        ObjectProperties(obj, idx)
                    }
                }

                editorState.selectedEnemyIndex != null -> {
                    val idx = editorState.selectedEnemyIndex!!
                    if (idx in levelData.enemies.indices) {
                        val enemy = levelData.enemies[idx]
                        EnemyProperties(enemy, idx)
                    }
                }

                else -> {
                    // Show level header info
                    LevelHeaderProperties(levelData)
                }
            }
        }
    }
}

@Composable
private fun ObjectProperties(obj: LevelObject, index: Int) {
    val name = ObjectCatalog.getObjectName(obj)

    Text(
        "Object #$index",
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.primary
    )
    Spacer(modifier = Modifier.height(4.dp))

    PropertyRow("Type", "$name (0x${obj.type.toString(16).uppercase().padStart(2, '0')})")
    PropertyRow("Page", "${obj.page}")
    PropertyRow("Row", "${obj.row}")
    PropertyRow("Column", "${obj.col}")
    PropertyRow("Param", "${obj.param}")
    PropertyRow("Abs Column", "${obj.page * 16 + obj.col}")
}

@Composable
private fun EnemyProperties(enemy: Enemy, index: Int) {
    val name = EnemyCatalog.getEnemyName(enemy.type)

    Text(
        "Enemy #$index",
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.error
    )
    Spacer(modifier = Modifier.height(4.dp))

    PropertyRow("Type", "$name (0x${enemy.type.toString(16).uppercase().padStart(2, '0')})")
    PropertyRow("Page", "${enemy.page}")
    PropertyRow("Row", "${enemy.row}")
    PropertyRow("Column", "${enemy.col}")
    PropertyRow("Abs Column", "${enemy.page * 16 + enemy.col}")
}

@Composable
private fun LevelHeaderProperties(level: LevelData) {
    Text(
        "${level.worldLevel} - ${level.name}",
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.primary
    )
    Spacer(modifier = Modifier.height(4.dp))

    val areaTypeName = LevelRegistry.AREA_TYPE_NAMES[level.header.areaType] ?: "Unknown"
    PropertyRow("Area Type", "$areaTypeName (${level.header.areaType})")
    PropertyRow("Area Style", "${level.header.platformType}")
    PropertyRow("Terrain Control", "${level.header.terrainControl}")
    PropertyRow("Cloud Override", "${level.header.cloudTypeOverride}")
    PropertyRow("BG Scenery", "${level.header.bgScenery}")
    PropertyRow("BG Color", "${level.header.bgColor}")
    PropertyRow("Time Settings", "${level.header.timeSettings}")
    PropertyRow("FG Scenery", "${level.header.fgScenery}")

    Spacer(modifier = Modifier.height(8.dp))
    Divider()
    Spacer(modifier = Modifier.height(8.dp))

    PropertyRow("Objects", "${level.objects.size}")
    PropertyRow("Enemies", "${level.enemies.size}")
}

@Composable
private fun PropertyRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall
        )
    }
}
