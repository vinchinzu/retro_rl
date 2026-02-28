package com.smb.editor.data

data class ObjectDef(
    val id: Int,
    val name: String,
    val category: String,     // "Block", "Pipe", "Scenery", "Structure", "Special"
    val fixedWidth: Int = 1,
    val fixedHeight: Int = 1,
    val expandable: Boolean = false,
    val expandDirection: String = "horizontal" // "horizontal", "vertical", "both"
)

object ObjectCatalog {
    val OBJECTS: Map<Int, ObjectDef> = mapOf(
        // Simple objects (row <= 0x0B, type = upper nibble of byte 1 bits 6-4)
        0x00 to ObjectDef(0x00, "Question Block (Mushroom)", "Block"),
        0x01 to ObjectDef(0x01, "Question Block (Coin)", "Block"),
        0x02 to ObjectDef(0x02, "Hidden Block (Coin)", "Block"),
        0x03 to ObjectDef(0x03, "Hidden Block (1-Up)", "Block"),
        0x04 to ObjectDef(0x04, "Brick (Mushroom)", "Block"),
        0x05 to ObjectDef(0x05, "Brick (Vine)", "Block"),
        0x06 to ObjectDef(0x06, "Brick (Star)", "Block"),
        0x07 to ObjectDef(0x07, "Brick (Multi-Coin)", "Block"),
        0x08 to ObjectDef(0x08, "Brick (1-Up)", "Block"),
        0x09 to ObjectDef(0x09, "Sideways Pipe", "Pipe"),
        0x0A to ObjectDef(0x0A, "Used Block", "Block"),
        0x0B to ObjectDef(0x0B, "Trampoline", "Special"),

        // Expandable / complex objects (row = 0x0C or type-encoded-with-length)
        0x0C to ObjectDef(0x0C, "Reverse L-Pipe", "Pipe", fixedWidth = 2, fixedHeight = 2),
        0x0D to ObjectDef(0x0D, "Horizontal Bricks", "Block", expandable = true),
        0x0E to ObjectDef(0x0E, "Horizontal Blocks", "Block", expandable = true),
        0x0F to ObjectDef(0x0F, "Horizontal Coins", "Block", expandable = true),
        0x10 to ObjectDef(0x10, "Vertical Bricks", "Block", expandable = true, expandDirection = "vertical"),
        0x11 to ObjectDef(0x11, "Vertical Blocks", "Block", expandable = true, expandDirection = "vertical"),
        0x12 to ObjectDef(0x12, "Untraversable Pipe", "Pipe", fixedWidth = 2, expandable = true, expandDirection = "vertical"),
        0x13 to ObjectDef(0x13, "Traversable Pipe", "Pipe", fixedWidth = 2, expandable = true, expandDirection = "vertical"),

        // Page-wide / scenery objects (row = 0x0D or 0x0E)
        0x14 to ObjectDef(0x14, "Hole", "Structure", expandable = true),
        0x15 to ObjectDef(0x15, "Balance Rope Horizontal", "Structure", expandable = true),
        0x16 to ObjectDef(0x16, "Bridge (High)", "Structure", expandable = true),
        0x17 to ObjectDef(0x17, "Bridge (Mid)", "Structure", expandable = true),
        0x18 to ObjectDef(0x18, "Bridge (Low)", "Structure", expandable = true),
        0x19 to ObjectDef(0x19, "Hole with Water/Lava", "Structure", expandable = true),
        0x1A to ObjectDef(0x1A, "Row of Coins (high)", "Block", expandable = true),
        0x1B to ObjectDef(0x1B, "Row of Coins (low)", "Block", expandable = true),

        // Full-column objects (row = 0x0E or 0x0F)
        0x1C to ObjectDef(0x1C, "Reverse L-pipe (tall)", "Pipe", fixedWidth = 2, fixedHeight = 4),
        0x1D to ObjectDef(0x1D, "Flagpole", "Special", fixedWidth = 1, fixedHeight = 10),

        // Staircases
        0x20 to ObjectDef(0x20, "Staircase (1 step)", "Structure", fixedWidth = 1, fixedHeight = 1),
        0x21 to ObjectDef(0x21, "Staircase (2 steps)", "Structure", fixedWidth = 2, fixedHeight = 2),
        0x22 to ObjectDef(0x22, "Staircase (3 steps)", "Structure", fixedWidth = 3, fixedHeight = 3),
        0x23 to ObjectDef(0x23, "Staircase (4 steps)", "Structure", fixedWidth = 4, fixedHeight = 4),
        0x24 to ObjectDef(0x24, "Staircase (5 steps)", "Structure", fixedWidth = 5, fixedHeight = 5),
        0x25 to ObjectDef(0x25, "Staircase (6 steps)", "Structure", fixedWidth = 6, fixedHeight = 6),
        0x26 to ObjectDef(0x26, "Staircase (7 steps)", "Structure", fixedWidth = 7, fixedHeight = 7),
        0x27 to ObjectDef(0x27, "Staircase (8 steps)", "Structure", fixedWidth = 8, fixedHeight = 8),

        // Castle
        0x2F to ObjectDef(0x2F, "Castle", "Structure", fixedWidth = 5, fixedHeight = 5),
    )

    val CATEGORIES = listOf("Block", "Pipe", "Scenery", "Structure", "Special")

    fun getObject(type: Int): ObjectDef? = OBJECTS[type]

    fun getObjectName(obj: LevelObject): String = getObjectName(obj.row, obj.type)

    fun getObjectName(row: Int, type: Int): String {
        return when {
            row < 0x0C -> {
                val classId = (type shr 4) and 0x07
                val param = type and 0x0F
                if (classId == 0) {
                    OBJECTS[param]?.name ?: "Small Object (0x${param.toString(16).uppercase()})"
                } else {
                    when (classId) {
                        0x01 -> "Area Style Object"
                        0x02 -> "Row of Bricks (len=${param + 1})"
                        0x03 -> "Row of Solid Blocks (len=${param + 1})"
                        0x04 -> "Row of Coins (len=${param + 1})"
                        0x05 -> "Column of Bricks (height=${param + 1})"
                        0x06 -> "Column of Solid Blocks (height=${param + 1})"
                        0x07 -> {
                            val warpPipe = (type and 0x08) != 0
                            val suffix = if (warpPipe) "Warp" else "Decor"
                            "Vertical Pipe ($suffix, h=${(type and 0x07) + 1})"
                        }
                        else -> "Object Class ${classId}"
                    }
                }
            }

            row == 0x0C -> {
                val specialId = (type shr 4) and 0x07
                val param = (type and 0x0F) + 1
                when (specialId) {
                    0x00 -> "Hole (len=$param)"
                    0x01 -> "Pulley Rope (len=$param)"
                    0x02 -> "Bridge (High, len=$param)"
                    0x03 -> "Bridge (Middle, len=$param)"
                    0x04 -> "Bridge (Low, len=$param)"
                    0x05 -> "Hole with Water/Lava (len=$param)"
                    0x06 -> "Question Row (High, len=$param)"
                    0x07 -> "Question Row (Low, len=$param)"
                    else -> "Row-12 Object (0x${type.toString(16).uppercase()})"
                }
            }

            row == 0x0D -> {
                if ((type and 0x40) == 0) {
                    "Page Control -> ${type and 0x1F}"
                } else {
                    when (type and 0x3F) {
                        0x00 -> "Intro Pipe"
                        0x01 -> "Flagpole"
                        0x02 -> "Axe"
                        0x03 -> "Chain"
                        0x04 -> "Castle Bridge"
                        0x05 -> "Scroll Lock (Warp Zone)"
                        0x06 -> "Scroll Lock Toggle"
                        0x07 -> "Scroll Lock Toggle"
                        0x08 -> "Area Frenzy (Flying Cheep)"
                        0x09 -> "Area Frenzy (Bullet/Cheep)"
                        0x0A -> "Area Frenzy Stop"
                        0x0B -> "Loop Command"
                        else -> "Row-13 Command (0x${type.toString(16).uppercase()})"
                    }
                }
            }

            row == 0x0E -> {
                if ((type and 0x40) == 0) {
                    val terrain = type and 0x0F
                    val bg = (type shr 4) and 0x03
                    "Alter Area Attr (terrain=$terrain, bg=$bg)"
                } else {
                    val value = type and 0x07
                    if (value >= 4) {
                        "Alter Area Attr (bgColor=$value)"
                    } else {
                        "Alter Area Attr (fgScenery=$value)"
                    }
                }
            }

            row == 0x0F -> {
                val specialId = (type shr 4) and 0x07
                val param = type and 0x0F
                when (specialId) {
                    0x00 -> "Endless Rope"
                    0x01 -> "Balance Platform Rope"
                    0x02 -> "Castle"
                    0x03 -> "Staircase (steps=${param + 1})"
                    0x04 -> "Exit Pipe"
                    0x05 -> "Flag Balls (residual)"
                    else -> "Row-15 Object (0x${type.toString(16).uppercase()})"
                }
            }

            else -> "Unknown Object (row=$row type=0x${type.toString(16).uppercase()})"
        }
    }

    fun getObjectName(type: Int): String =
        OBJECTS[type]?.name ?: "Object Type (0x${type.toString(16).uppercase()})"

    fun byCategory(category: String): List<ObjectDef> =
        OBJECTS.values.filter { it.category == category }
}
