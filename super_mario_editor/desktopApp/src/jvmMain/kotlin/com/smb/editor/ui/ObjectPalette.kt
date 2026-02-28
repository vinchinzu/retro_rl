@file:OptIn(ExperimentalMaterial3Api::class)

package com.smb.editor.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.smb.editor.data.ObjectCatalog

@Composable
fun ObjectPalette(
    editorState: EditorState,
    modifier: Modifier = Modifier
) {
    Surface(
        tonalElevation = 1.dp,
        modifier = modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(8.dp)) {
            Text(
                "Object Palette",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            // Tool selection
            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                modifier = Modifier.padding(bottom = 8.dp)
            ) {
                for (tool in EditorTool.values()) {
                    FilterChip(
                        selected = editorState.tool == tool,
                        onClick = { editorState.tool = tool },
                        label = { Text(tool.name.lowercase().replace("_", " ").replaceFirstChar { it.uppercase() }) }
                    )
                }
            }

            Divider()

            // Object list by category
            LazyColumn(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                for (category in ObjectCatalog.CATEGORIES) {
                    val objects = ObjectCatalog.byCategory(category)
                    if (objects.isEmpty()) continue

                    item(key = "cat-$category") {
                        Text(
                            text = category,
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(top = 8.dp, bottom = 4.dp)
                        )
                    }

                    items(objects, key = { it.id }) { obj ->
                        val isSelected = editorState.placementObjectType == obj.id
                        Surface(
                            tonalElevation = if (isSelected) 4.dp else 0.dp,
                            color = if (isSelected) MaterialTheme.colorScheme.primaryContainer else Color.Transparent,
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    editorState.placementObjectType = obj.id
                                    editorState.tool = EditorTool.PLACE_OBJECT
                                }
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Column {
                                Text(
                                    text = obj.name,
                                    style = MaterialTheme.typography.bodyMedium
                                )
                                Text(
                                    text = "ID: 0x${obj.id.toString(16).uppercase().padStart(2, '0')}${if (obj.expandable) " (expandable)" else ""}",
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
}
