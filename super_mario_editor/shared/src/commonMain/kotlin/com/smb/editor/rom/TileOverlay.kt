package com.smb.editor.rom

import com.smb.editor.data.*

enum class TileOverlay {
    ENEMIES,
    OBJECTS,
    GRID,
    COINS,
    POWERUPS,
    PIPES,
    WARP_ZONES
}

data class OverlayItem(
    val x: Int,           // pixel x
    val y: Int,           // pixel y
    val width: Int,       // pixel width
    val height: Int,      // pixel height
    val label: String,
    val color: Int,       // ARGB
    val type: OverlayType
)

enum class OverlayType {
    OBJECT_BOUNDS,
    ENEMY_MARKER,
    GRID_LINE,
    COIN_HIGHLIGHT,
    POWERUP_HIGHLIGHT,
    PIPE_MARKER,
    WARP_ZONE
}

object OverlayGenerator {
    fun generate(level: LevelData, overlays: Set<TileOverlay>): List<OverlayItem> {
        val items = mutableListOf<OverlayItem>()
        val mapHeightPx = 13 * 16

        if (TileOverlay.OBJECTS in overlays) {
            for (obj in level.objects) {
                val x = (obj.page * 16 + obj.col) * 16
                val y = obj.row * 16
                val name = ObjectCatalog.getObjectName(obj)
                items.add(OverlayItem(x, y, 16, 16, name, 0x8000FF00.toInt(), OverlayType.OBJECT_BOUNDS))
            }
        }

        if (TileOverlay.ENEMIES in overlays) {
            for (enemy in level.enemies) {
                val x = (enemy.page * 16 + enemy.col) * 16
                val y = enemy.row * 16
                val name = EnemyCatalog.getEnemyName(enemy.type)
                items.add(OverlayItem(x, y, 16, 16, name, 0x80FF0000.toInt(), OverlayType.ENEMY_MARKER))
            }
        }

        if (TileOverlay.COINS in overlays) {
            for (obj in level.objects) {
                if (!isCoinObject(obj)) continue
                val x = (obj.page * 16 + obj.col) * 16
                val y = obj.row * 16
                items.add(
                    OverlayItem(
                        x = x,
                        y = y,
                        width = 16,
                        height = 16,
                        label = "Coin",
                        color = 0xB0FFD700.toInt(),
                        type = OverlayType.COIN_HIGHLIGHT
                    )
                )
            }
        }

        if (TileOverlay.POWERUPS in overlays) {
            for (obj in level.objects) {
                if (!isPowerupObject(obj)) continue
                val x = (obj.page * 16 + obj.col) * 16
                val y = obj.row * 16
                items.add(
                    OverlayItem(
                        x = x,
                        y = y,
                        width = 16,
                        height = 16,
                        label = "Powerup",
                        color = 0xB0FF8C00.toInt(),
                        type = OverlayType.POWERUP_HIGHLIGHT
                    )
                )
            }
        }

        if (TileOverlay.PIPES in overlays) {
            for (obj in level.objects) {
                if (!isPipeObject(obj)) continue
                val x = (obj.page * 16 + obj.col) * 16
                val y = obj.row * 16
                val (w, h) = pipeBounds(obj)
                items.add(
                    OverlayItem(
                        x = x,
                        y = y,
                        width = w,
                        height = h,
                        label = "Pipe",
                        color = 0xB000FFFF.toInt(),
                        type = OverlayType.PIPE_MARKER
                    )
                )
            }
        }

        if (TileOverlay.WARP_ZONES in overlays) {
            for (obj in level.objects) {
                if (!isWarpZoneCommand(obj)) continue
                val x = (obj.page * 16 + obj.col) * 16
                items.add(
                    OverlayItem(
                        x = x,
                        y = 0,
                        width = 16,
                        height = mapHeightPx,
                        label = "Warp Cmd",
                        color = 0x90FF00FF.toInt(),
                        type = OverlayType.WARP_ZONE
                    )
                )
            }
        }

        return items
    }

    private fun isCoinObject(obj: LevelObject): Boolean {
        val type = obj.type and 0x7F
        return when {
            obj.row < 0x0C -> {
                val classId = (type shr 4) and 0x07
                val param = type and 0x0F
                (classId == 0 && (param == 0x01 || param == 0x02 || param == 0x07)) ||
                    classId == 0x04
            }
            obj.row == 0x0C -> {
                val specialId = (type shr 4) and 0x07
                specialId == 0x06 || specialId == 0x07
            }
            else -> false
        }
    }

    private fun isPowerupObject(obj: LevelObject): Boolean {
        if (obj.row >= 0x0C) return false
        val classId = (obj.type shr 4) and 0x07
        val param = obj.type and 0x0F
        if (classId != 0) return false
        return param in setOf(
            0x00, // question block (power-up)
            0x03, // hidden 1-up
            0x04, // brick with mushroom
            0x05, // brick with vine
            0x06, // brick with star
            0x08  // brick with 1-up
        )
    }

    private fun isPipeObject(obj: LevelObject): Boolean {
        val type = obj.type and 0x7F
        return when {
            obj.row < 0x0C -> {
                val classId = (type shr 4) and 0x07
                val param = type and 0x0F
                classId == 0x07 || (classId == 0 && param == 0x09)
            }
            obj.row == 0x0D -> (type and 0x3F) == 0x00 // IntroPipe
            obj.row == 0x0F -> ((type shr 4) and 0x07) == 0x04 // ExitPipe
            else -> false
        }
    }

    private fun pipeBounds(obj: LevelObject): Pair<Int, Int> {
        return when {
            obj.row < 0x0C && ((obj.type shr 4) and 0x07) == 0x07 -> {
                val heightTiles = (obj.type and 0x07) + 1
                Pair(32, heightTiles.coerceAtLeast(2) * 16)
            }
            obj.row == 0x0D && (obj.type and 0x3F) == 0x00 -> Pair(32, 64)
            obj.row == 0x0F && ((obj.type shr 4) and 0x07) == 0x04 -> Pair(32, 64)
            else -> Pair(16, 16)
        }
    }

    private fun isWarpZoneCommand(obj: LevelObject): Boolean {
        // Row-13 command 0x05 toggles warp-zone scroll lock text/behavior.
        return obj.row == 0x0D && (obj.type and 0x40) != 0 && (obj.type and 0x3F) == 0x05
    }
}
