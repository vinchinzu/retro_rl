package com.smb.editor.data

data class AreaEntry(
    val worldLevel: String,
    val name: String,
    val areaType: Int,        // 0=water, 1=overground, 2=underground, 3=castle
    val areaNumber: Int,      // legacy field (not used by ROM lookup)
    val worldIndex: Int,      // 0-based world number (0-7)
    val areaSlot: Int         // index into that world's area-pointer table
)

object LevelRegistry {
    val AREA_TYPE_NAMES = mapOf(
        0 to "Water",
        1 to "Overground",
        2 to "Underground",
        3 to "Castle"
    )

    // All 32 world-levels.
    // worldIndex/areaSlot are used by LevelDecoder for ROM lookup:
    //   WorldAddrOffsets[worldIndex] + areaSlot -> AreaPointerTable entry
    // Some worlds include extra sub-area entries, so areaSlot is not always
    // the same as the human level number minus 1.
    val LEVELS: List<AreaEntry> = listOf(
        // World 1
        AreaEntry("1-1", "World 1-1", 1, 0, worldIndex = 0, areaSlot = 0),
        AreaEntry("1-2", "World 1-2", 2, 0, worldIndex = 0, areaSlot = 2),
        AreaEntry("1-3", "World 1-3", 1, 0, worldIndex = 0, areaSlot = 3),
        AreaEntry("1-4", "World 1-4", 3, 0, worldIndex = 0, areaSlot = 4),
        // World 2
        AreaEntry("2-1", "World 2-1", 1, 0, worldIndex = 1, areaSlot = 0),
        AreaEntry("2-2", "World 2-2", 0, 0, worldIndex = 1, areaSlot = 2),
        AreaEntry("2-3", "World 2-3", 1, 0, worldIndex = 1, areaSlot = 3),
        AreaEntry("2-4", "World 2-4", 3, 0, worldIndex = 1, areaSlot = 4),
        // World 3
        AreaEntry("3-1", "World 3-1", 1, 0, worldIndex = 2, areaSlot = 0),
        AreaEntry("3-2", "World 3-2", 1, 0, worldIndex = 2, areaSlot = 1),
        AreaEntry("3-3", "World 3-3", 1, 0, worldIndex = 2, areaSlot = 2),
        AreaEntry("3-4", "World 3-4", 3, 0, worldIndex = 2, areaSlot = 3),
        // World 4
        AreaEntry("4-1", "World 4-1", 1, 0, worldIndex = 3, areaSlot = 0),
        AreaEntry("4-2", "World 4-2", 2, 0, worldIndex = 3, areaSlot = 2),
        AreaEntry("4-3", "World 4-3", 1, 0, worldIndex = 3, areaSlot = 3),
        AreaEntry("4-4", "World 4-4", 3, 0, worldIndex = 3, areaSlot = 4),
        // World 5
        AreaEntry("5-1", "World 5-1", 1, 0, worldIndex = 4, areaSlot = 0),
        AreaEntry("5-2", "World 5-2", 1, 0, worldIndex = 4, areaSlot = 1),
        AreaEntry("5-3", "World 5-3", 1, 0, worldIndex = 4, areaSlot = 2),
        AreaEntry("5-4", "World 5-4", 3, 0, worldIndex = 4, areaSlot = 3),
        // World 6
        AreaEntry("6-1", "World 6-1", 1, 0, worldIndex = 5, areaSlot = 0),
        AreaEntry("6-2", "World 6-2", 1, 0, worldIndex = 5, areaSlot = 1),
        AreaEntry("6-3", "World 6-3", 1, 0, worldIndex = 5, areaSlot = 2),
        AreaEntry("6-4", "World 6-4", 3, 0, worldIndex = 5, areaSlot = 3),
        // World 7
        AreaEntry("7-1", "World 7-1", 1, 0, worldIndex = 6, areaSlot = 0),
        AreaEntry("7-2", "World 7-2", 0, 0, worldIndex = 6, areaSlot = 2),
        AreaEntry("7-3", "World 7-3", 1, 0, worldIndex = 6, areaSlot = 3),
        AreaEntry("7-4", "World 7-4", 3, 0, worldIndex = 6, areaSlot = 4),
        // World 8
        AreaEntry("8-1", "World 8-1", 1, 0, worldIndex = 7, areaSlot = 0),
        AreaEntry("8-2", "World 8-2", 1, 0, worldIndex = 7, areaSlot = 1),
        AreaEntry("8-3", "World 8-3", 1, 0, worldIndex = 7, areaSlot = 2),
        AreaEntry("8-4", "World 8-4", 3, 0, worldIndex = 7, areaSlot = 3),
    )

    fun findByWorldLevel(worldLevel: String): AreaEntry? =
        LEVELS.find { it.worldLevel == worldLevel }

    fun levelsByWorld(world: Int): List<AreaEntry> =
        LEVELS.filter { it.worldLevel.startsWith("$world-") }
}
