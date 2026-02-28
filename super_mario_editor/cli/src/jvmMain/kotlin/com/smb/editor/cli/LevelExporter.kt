package com.smb.editor.cli

import com.smb.editor.rom.*
import com.smb.editor.data.*

class LevelExporter {
    fun exportLevel(rom: NesRom, entry: AreaEntry): LevelExportModel {
        val level = LevelDecoder.decodeLevel(rom, entry)
        val grid = LevelDecoder.expandToGrid(level)

        return LevelExportModel(
            worldLevel = entry.worldLevel,
            name = entry.name,
            areaType = entry.areaType,
            areaTypeName = LevelRegistry.AREA_TYPE_NAMES[entry.areaType] ?: "Unknown",
            header = HeaderExport(
                areaType = level.header.areaType,
                fgScenery = level.header.fgScenery,
                bgScenery = level.header.bgScenery,
                platformType = level.header.platformType,
                terrainControl = level.header.terrainControl,
                cloudTypeOverride = level.header.cloudTypeOverride,
                bgColor = level.header.bgColor,
                timeSettings = level.header.timeSettings
            ),
            objects = level.objects.mapIndexed { i, obj ->
                ObjectExport(
                    index = i,
                    row = obj.row,
                    col = obj.col,
                    page = obj.page,
                    type = obj.type,
                    typeHex = "0x${obj.type.toString(16).uppercase().padStart(2, '0')}",
                    typeName = ObjectCatalog.getObjectName(obj),
                    param = obj.param
                )
            },
            enemies = level.enemies.mapIndexed { i, enemy ->
                EnemyExport(
                    index = i,
                    row = enemy.row,
                    col = enemy.col,
                    page = enemy.page,
                    type = enemy.type,
                    typeHex = "0x${enemy.type.toString(16).uppercase().padStart(2, '0')}",
                    typeName = EnemyCatalog.getEnemyName(enemy.type)
                )
            },
            gridWidth = if (grid.isNotEmpty()) grid[0].size else 0,
            gridHeight = grid.size
        )
    }
}
