package com.smb.editor.rom

import com.smb.editor.data.*

/**
 * Decodes SMB levels from ROM binary data.
 * Parses object/enemy streams and expands objects into a metatile grid.
 *
 * SMB area object byte format (per SMBDIS.ASM by doppelganger):
 *   byte0: bits 7-4 = X column within page, bits 3-0 = Y row
 *   byte1: bit 7 = page advance flag, bits 6-0 = object type/control
 *
 * Y row values:
 *   0x00-0x0B: regular gameplay rows (block/platform objects)
 *   0x0C: terrain/floor modification (gap/restore)
 *   0x0D: scenery change or area pointer
 *   0x0E: (unused or same as 0x0D)
 *   0x0F: page skip command
 */
object LevelDecoder {

    // SMB ROM address tables for area data lookup.
    private const val WORLD_ADDR_OFFSETS = 0x9CB4
    private const val AREA_POINTER_TABLE = 0x9CBC
    private const val AREA_ADDR_TYPE_OFFSETS = 0x9D28
    private const val AREA_DATA_LO = 0x9D2C
    private const val AREA_DATA_HI = 0x9D4E
    private const val ENEMY_ADDR_TYPE_OFFSETS = 0x9CE0
    private const val ENEMY_DATA_LO = 0x9CE4
    private const val ENEMY_DATA_HI = 0x9D06

    private data class AreaLookup(
        val areaType: Int,
        val areaTableIndex: Int,
        val objectPtr: Int,
        val enemyPtr: Int
    )

    private data class ColumnRenderState(
        val terrainControl: Int,
        val bgScenery: Int,
        val fgScenery: Int
    )

    private fun lookupArea(rom: NesRom, worldIndex: Int, areaSlot: Int): AreaLookup {
        val worldOffset = rom.readPrg(WORLD_ADDR_OFFSETS + worldIndex)
        val areaPointer = rom.readPrg(AREA_POINTER_TABLE + worldOffset + areaSlot)
        val areaType = (areaPointer shr 5) and 0x03
        val areaIdx = areaPointer and 0x1F

        val areaTypeOffset = rom.readPrg(AREA_ADDR_TYPE_OFFSETS + areaType)
        val areaTableIndex = areaTypeOffset + areaIdx
        val areaLo = rom.readPrg(AREA_DATA_LO + areaTableIndex)
        val areaHi = rom.readPrg(AREA_DATA_HI + areaTableIndex)
        val objectPtr = (areaHi shl 8) or areaLo

        val enemyTypeOffset = rom.readPrg(ENEMY_ADDR_TYPE_OFFSETS + areaType)
        val enemyTableIndex = enemyTypeOffset + areaIdx
        val enemyLo = rom.readPrg(ENEMY_DATA_LO + enemyTableIndex)
        val enemyHi = rom.readPrg(ENEMY_DATA_HI + enemyTableIndex)
        val enemyPtr = (enemyHi shl 8) or enemyLo

        return AreaLookup(
            areaType = areaType,
            areaTableIndex = areaTableIndex,
            objectPtr = objectPtr,
            enemyPtr = enemyPtr
        )
    }

    /**
     * Decode a level from ROM.
     */
    fun decodeLevel(rom: NesRom, entry: AreaEntry): LevelData {
        val lookup = lookupArea(rom, entry.worldIndex, entry.areaSlot)
        val objectPtr = lookup.objectPtr
        val enemyPtr = lookup.enemyPtr

        // Parse level header (first 2 bytes of object data) using SMB area-header layout:
        // Byte 0: bits 7-6=time, bits 5-3=entrance, bits 2-0=fg scenery OR bg-color control
        // Byte 1: bits 7-6=area style (3 => cloud override), bits 5-4=bg scenery, bits 3-0=terrain control
        // areaType comes from the area pointer entry.
        val headerByte0 = rom.readPrg(objectPtr)
        val headerByte1 = rom.readPrg(objectPtr + 1)

        val fgOrColor = headerByte0 and 0x07
        val fgScenery = if (fgOrColor < 4) fgOrColor else 0
        val bgColor = if (fgOrColor >= 4) fgOrColor else 0

        val areaStyleBits = (headerByte1 shr 6) and 0x03
        val cloudTypeOverride = areaStyleBits == 0x03
        val areaStyle = if (cloudTypeOverride) 0 else areaStyleBits

        val header = LevelHeader(
            fgScenery = fgScenery,
            bgScenery = (headerByte1 shr 4) and 0x03,
            platformType = areaStyle,
            terrainControl = headerByte1 and 0x0F,
            cloudTypeOverride = cloudTypeOverride,
            timeSettings = (headerByte0 shr 6) and 0x03,
            bgColor = bgColor,
            areaType = lookup.areaType
        )

        val objects = parseObjects(rom, objectPtr + 2)
        val enemies = parseEnemies(rom, enemyPtr)

        return LevelData(
            header = header,
            objects = objects,
            enemies = enemies,
            worldLevel = entry.worldLevel,
            name = entry.name,
            areaIndex = lookup.areaTableIndex
        )
    }

    private fun parseObjects(rom: NesRom, startAddr: Int): List<LevelObject> {
        val objects = mutableListOf<LevelObject>()
        var addr = startAddr
        var currentPage = 0

        while (true) {
            val byte0 = rom.readPrg(addr)
            if (byte0 == 0xFD) break  // terminator

            val byte1 = rom.readPrg(addr + 1)

            // Bit 7 of byte1 = page advance flag
            if (byte1 and 0x80 != 0) {
                currentPage++
            }

            // CRITICAL: byte0 upper nibble = X column, lower nibble = Y row
            val col = (byte0 shr 4) and 0x0F
            val row = byte0 and 0x0F
            val type = byte1 and 0x7F

            // Special row 13 page-control command (d6 clear): sets absolute page.
            if (row == 0x0D && (byte1 and 0x40) == 0) {
                currentPage = type and 0x1F
                addr += 2
                continue
            }

            objects.add(
                LevelObject(
                    row = row,
                    col = col,
                    page = currentPage,
                    type = type,
                    param = 0
                )
            )

            addr += 2
            if (objects.size > 500) break
        }

        return objects
    }

    private fun parseEnemies(rom: NesRom, startAddr: Int): List<Enemy> {
        val enemies = mutableListOf<Enemy>()
        var addr = startAddr
        var currentPage = 0

        while (true) {
            val byte0 = rom.readPrg(addr)
            if (byte0 and 0xFF == 0xFF) break  // terminator

            // Enemy byte0 uses: upper nibble = X, lower nibble = Y
            val col = (byte0 shr 4) and 0x0F
            val row = byte0 and 0x0F
            val byte1 = rom.readPrg(addr + 1)

            // Special row 15 enemy command sets absolute page and is not a placed enemy.
            if (row == 0x0F) {
                currentPage = byte1 and 0x3F
                addr += 2
                continue
            }

            // Page advance flag for regular enemy objects.
            if (byte1 and 0x80 != 0) {
                currentPage++
            }

            // Special row 14 enemy command uses three bytes (area transfer command).
            if (row == 0x0E) {
                addr += 3
                continue
            }

            val type = byte1 and 0x3F

            enemies.add(
                Enemy(
                    row = row,
                    col = col,
                    page = currentPage,
                    type = type
                )
            )

            addr += 2
            if (enemies.size > 800) break
        }

        return enemies
    }

    /**
     * Expand a level's objects into a 2D tile grid.
     * Grid is 13 rows x (pages x 16) columns of metatile indices.
     */
    fun expandToGrid(level: LevelData, numPages: Int? = null): Array<IntArray> {
        val maxObjPage = level.objects.maxOfOrNull { it.page } ?: 0
        // Geometry width is driven by area objects; enemy records can legally trail farther.
        val inferredPages = (maxObjPage + 1).coerceIn(1, 32)
        val totalPages = (numPages ?: inferredPages).coerceIn(1, 32)

        val cols = totalPages * 16
        val rows = 13

        // Initialize with area-type-dependent background
        val bgTile = when (level.header.areaType) {
            0 -> 0x19    // water body
            1 -> 0x00    // overground: sky (empty)
            2 -> 0x00    // underground: black
            3 -> 0x00    // castle: black
            else -> 0x00
        }

        val grid = Array(rows) { IntArray(cols) { bgTile } }

        val columnState = buildColumnRenderState(level, cols)

        // SMB renders background then foreground scenery and terrain before area objects.
        addScenery(grid, level.header, columnState, rows, cols)
        addForegroundScenery(grid, columnState, rows, cols)
        addTerrain(grid, level.header, columnState, rows, cols)

        // Place objects into grid
        for (obj in level.objects) {
            val absCol = obj.page * 16 + obj.col
            if (absCol >= cols) continue
            placeObject(
                grid = grid,
                obj = obj,
                absCol = absCol,
                rows = rows,
                cols = cols,
                areaType = level.header.areaType,
                areaStyle = level.header.platformType
            )
        }

        return grid
    }

    /**
     * Add background scenery from the original SMB tables:
     * `BackSceneryData` + `BackSceneryMetatiles` in smbdis.asm.
     */
    private fun buildColumnRenderState(level: LevelData, cols: Int): List<ColumnRenderState> {
        if (cols <= 0) return emptyList()

        val row14Commands = level.objects
            .withIndex()
            .filter { it.value.row == 0x0E }
            .sortedWith(
                compareBy<IndexedValue<LevelObject>>(
                    { it.value.page * 16 + it.value.col },
                    { it.index }
                )
            )

        var terrainControl = level.header.terrainControl and 0x0F
        var bgScenery = level.header.bgScenery and 0x03
        var fgScenery = level.header.fgScenery and 0x03
        var cmdIdx = 0
        val state = ArrayList<ColumnRenderState>(cols)

        for (col in 0 until cols) {
            while (cmdIdx < row14Commands.size) {
                val obj = row14Commands[cmdIdx].value
                val objCol = obj.page * 16 + obj.col
                if (objCol > col) break

                if ((obj.type and 0x40) == 0) {
                    // d6 clear: terrain control (low nibble) + bg scenery (bits 5-4)
                    terrainControl = obj.type and 0x0F
                    bgScenery = (obj.type shr 4) and 0x03
                } else {
                    // d6 set: foreground scenery (0-3) or bg-color control (4-7).
                    val v = obj.type and 0x07
                    fgScenery = if (v >= 4) 0 else v
                }
                cmdIdx++
            }

            state.add(
                ColumnRenderState(
                    terrainControl = terrainControl and 0x0F,
                    bgScenery = bgScenery and 0x03,
                    fgScenery = fgScenery and 0x03
                )
            )
        }

        return state
    }

    private fun addScenery(
        grid: Array<IntArray>,
        header: LevelHeader,
        columnState: List<ColumnRenderState>,
        rows: Int,
        cols: Int
    ) {
        // Only draw scenery for overground levels
        if (header.areaType != 1) return
        for (col in 0 until cols) {
            val sceneryType = columnState[col].bgScenery
            if (sceneryType !in 1..3) continue
            val segmentOffset = BACK_SCENE_DATA_OFFSETS[sceneryType - 1]
            val pageMod3 = (col / 16) % 3
            val colInPage = col and 0x0F
            val dataIdx = segmentOffset + pageMod3 * 16 + colInPage
            val encoded = BACK_SCENERY_DATA[dataIdx]
            if (encoded == 0) continue

            val shapeIndex = (encoded and 0x0F) - 1
            if (shapeIndex !in 0..11) continue
            val shapeBase = shapeIndex * 3
            var row = (encoded ushr 4) and 0x0F

            for (i in 0 until 3) {
                if (row >= 0x0B || row >= rows) break
                val code = BACK_SCENERY_METATILES[shapeBase + i]
                val metatile = decodeBackSceneryMetatile(code)
                if (metatile != null && grid[row][col] == 0x00) {
                    grid[row][col] = metatile
                }
                row++
            }
        }
    }

    private fun addForegroundScenery(
        grid: Array<IntArray>,
        columnState: List<ColumnRenderState>,
        rows: Int,
        cols: Int
    ) {
        for (col in 0 until cols) {
            val fg = columnState[col].fgScenery
            if (fg !in 1..3) continue
            val base = FORE_SCENE_DATA_OFFSETS[fg - 1]
            for (row in 0 until rows) {
                val code = FORE_SCENERY_DATA[base + row]
                if (code == 0) continue
                val metatile = decodeForeSceneryMetatile(code) ?: continue
                if (grid[row][col] == 0x00) {
                    grid[row][col] = metatile
                }
            }
        }
    }

    private fun decodeForeSceneryMetatile(code: Int): Int? = when (code and 0xFF) {
        0x86 -> 0x18 // water surface
        0x87 -> 0x19 // water body
        0x69 -> 0x01 // cracked ground
        0x45 -> 0x0F // castle top accent
        0x47 -> 0x10 // castle wall
        else -> null
    }

    private fun addTerrain(
        grid: Array<IntArray>,
        header: LevelHeader,
        columnState: List<ColumnRenderState>,
        rows: Int,
        cols: Int
    ) {
        if (rows <= 0 || cols <= 0) return

        for (col in 0 until cols) {
            val tc = columnState[col].terrainControl and 0x0F
            val bitsIdx = tc * 2
            val bitsTop = TERRAIN_RENDER_BITS[bitsIdx]
            var bitsBottom = TERRAIN_RENDER_BITS[bitsIdx + 1]

            // Cloud-style levels only keep the floor bit in the lower terrain byte.
            if (header.cloudTypeOverride) {
                bitsBottom = bitsBottom and 0x08
            }

            var row = 0
            for (bit in 0 until 8) {
                if (row >= rows) break
                if ((bitsTop and (1 shl bit)) != 0) {
                    grid[row][col] = terrainMetatile(header.areaType, row)
                }
                row++
            }
            for (bit in 0 until 8) {
                if (row >= rows) break
                if ((bitsBottom and (1 shl bit)) != 0) {
                    grid[row][col] = terrainMetatile(header.areaType, row)
                }
                row++
            }
        }
    }

    private fun terrainMetatile(areaType: Int, row: Int): Int {
        // Underground uses cracked ground for the final two rows.
        if (areaType == 2 && row >= 11) return 0x01
        return when (areaType) {
            0 -> 0x19 // water body
            1 -> 0x01 // overground cracked ground
            2 -> 0x03 // underground brick fill
            3 -> 0x10 // castle wall
            else -> 0x01
        }
    }

    private fun decodeBackSceneryMetatile(code: Int): Int? = when (code and 0xFF) {
        0x00 -> 0x00
        0x80 -> METATILE_CLOUD_TL
        0x81 -> METATILE_CLOUD_TM
        0x82 -> METATILE_CLOUD_TR
        0x83 -> METATILE_CLOUD_BL
        0x84 -> METATILE_CLOUD_BM
        0x85 -> METATILE_CLOUD_BR
        0x02 -> METATILE_BUSH_LEFT
        0x03 -> METATILE_BUSH_MID
        0x04 -> METATILE_BUSH_RIGHT
        0x05 -> METATILE_HILL_LEFT
        0x06 -> METATILE_HILL_LEFT_BOTTOM
        0x07 -> METATILE_HILL_TOP
        0x08 -> METATILE_HILL_RIGHT
        0x09 -> METATILE_HILL_RIGHT_BOTTOM
        0x0A -> METATILE_HILL_MIDDLE_BOTTOM
        0x4D -> METATILE_FENCE
        0x0D -> METATILE_TREE_TOP_TALL_UPPER
        0x0F -> METATILE_TREE_TOP_TALL_LOWER
        0x0E -> METATILE_TREE_TOP_SHORT
        0x4E -> METATILE_TREE_TRUNK
        else -> null
    }

    private val BACK_SCENE_DATA_OFFSETS = intArrayOf(0x00, 0x30, 0x60)

    private val FORE_SCENE_DATA_OFFSETS = intArrayOf(0x00, 0x0D, 0x1A)

    private val FORE_SCENERY_DATA = intArrayOf(
        // Foreground scenery 1 (water)
        0x86, 0x87, 0x87, 0x87, 0x87, 0x87, 0x87,
        0x87, 0x87, 0x87, 0x87, 0x69, 0x69,
        // Foreground scenery 2 (wall)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x45, 0x47,
        0x47, 0x47, 0x47, 0x47, 0x00, 0x00,
        // Foreground scenery 3 (over-water)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x86, 0x87
    )

    private val TERRAIN_RENDER_BITS = intArrayOf(
        0b00000000, 0b00000000, // no ceiling or floor
        0b00000000, 0b00011000, // floor 2
        0b00000001, 0b00011000, // ceiling 1, floor 2
        0b00000111, 0b00011000, // ceiling 3, floor 2
        0b00001111, 0b00011000, // ceiling 4, floor 2
        0b11111111, 0b00011000, // ceiling 8, floor 2
        0b00000001, 0b00011111, // ceiling 1, floor 5
        0b00000111, 0b00011111, // ceiling 3, floor 5
        0b00001111, 0b00011111, // ceiling 4, floor 5
        0b10000001, 0b00011111, // ceiling 1, floor 6
        0b00000001, 0b00000000, // ceiling 1, no floor
        0b10001111, 0b00011111, // ceiling 4, floor 6
        0b11110001, 0b00011111, // ceiling 1, floor 9
        0b11111001, 0b00011000, // ceiling 1, middle 5, floor 2
        0b11110001, 0b00011000, // ceiling 1, middle 4, floor 2
        0b11111111, 0b00011111  // full solid
    )

    private val BACK_SCENERY_DATA = intArrayOf(
        // bgScenery=1 (clouds)
        0x93, 0x00, 0x00, 0x11, 0x12, 0x12, 0x13, 0x00,
        0x00, 0x51, 0x52, 0x53, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x02, 0x02, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x91, 0x92, 0x93, 0x00,
        0x00, 0x00, 0x00, 0x51, 0x52, 0x53, 0x41, 0x42,
        0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x91, 0x92,

        // bgScenery=2 (mountains + bushes)
        0x97, 0x87, 0x88, 0x89, 0x99, 0x00, 0x00, 0x00,
        0x11, 0x12, 0x13, 0xA4, 0xA5, 0xA5, 0xA5, 0xA6,
        0x97, 0x98, 0x99, 0x01, 0x02, 0x03, 0x00, 0xA4,
        0xA5, 0xA6, 0x00, 0x11, 0x12, 0x12, 0x12, 0x13,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x02, 0x03,
        0x00, 0xA4, 0xA5, 0xA5, 0xA6, 0x00, 0x00, 0x00,

        // bgScenery=3 (trees + fences)
        0x11, 0x12, 0x12, 0x13, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x9C, 0x00, 0x8B, 0xAA, 0xAA,
        0xAA, 0xAA, 0x11, 0x12, 0x13, 0x8B, 0x00, 0x9C,
        0x9C, 0x00, 0x00, 0x01, 0x02, 0x03, 0x11, 0x12,
        0x12, 0x13, 0x00, 0x00, 0x00, 0x00, 0xAA, 0xAA,
        0x9C, 0xAA, 0x00, 0x8B, 0x00, 0x01, 0x02, 0x03
    )

    private val BACK_SCENERY_METATILES = intArrayOf(
        0x80, 0x83, 0x00, // cloud left
        0x81, 0x84, 0x00, // cloud middle
        0x82, 0x85, 0x00, // cloud right
        0x02, 0x00, 0x00, // bush left
        0x03, 0x00, 0x00, // bush middle
        0x04, 0x00, 0x00, // bush right
        0x00, 0x05, 0x06, // mountain left
        0x07, 0x06, 0x0A, // mountain middle
        0x00, 0x08, 0x09, // mountain right
        0x4D, 0x00, 0x00, // fence
        0x0D, 0x0F, 0x4E, // tall tree
        0x0E, 0x4E, 0x4E  // short tree
    )

    /**
     * Draw a large hill (5 metatiles wide at base, 4 rows tall).
     * Shape from SMB: triangle with peak at center.
     * Bottom at row 10, peak at row 7.
     */
    private fun drawLargeHill(
        grid: Array<IntArray>,
        startCol: Int,
        rows: Int,
        cols: Int
    ) {
        fun setIfSky(x: Int, y: Int, metatile: Int) {
            if (x in 0 until cols && y in 0 until rows && grid[y][x] == 0x00) {
                grid[y][x] = metatile
            }
        }

        // Row 10: base
        setIfSky(startCol + 0, 10, METATILE_HILL_LEFT_BOTTOM)
        setIfSky(startCol + 1, 10, METATILE_HILL_BODY)
        setIfSky(startCol + 2, 10, METATILE_HILL_BODY)
        setIfSky(startCol + 3, 10, METATILE_HILL_BODY)
        setIfSky(startCol + 4, 10, METATILE_HILL_RIGHT_BOTTOM)

        // Row 9: shoulders
        setIfSky(startCol + 0, 9, METATILE_HILL_LEFT)
        setIfSky(startCol + 1, 9, METATILE_HILL_LEFT_BOTTOM)
        setIfSky(startCol + 2, 9, METATILE_HILL_BODY)
        setIfSky(startCol + 3, 9, METATILE_HILL_RIGHT_BOTTOM)
        setIfSky(startCol + 4, 9, METATILE_HILL_RIGHT)

        // Row 8-7: peak
        setIfSky(startCol + 1, 8, METATILE_HILL_LEFT)
        setIfSky(startCol + 2, 8, METATILE_HILL_TOP)
        setIfSky(startCol + 3, 8, METATILE_HILL_RIGHT)
        setIfSky(startCol + 2, 7, METATILE_HILL_TOP)
    }

    /**
     * Draw a small hill (3 metatiles wide at base, 2 rows tall).
     * Bottom at row 10, peak at row 9.
     */
    private fun drawSmallHill(
        grid: Array<IntArray>,
        startCol: Int,
        rows: Int,
        cols: Int
    ) {
        fun setIfSky(x: Int, y: Int, metatile: Int) {
            if (x in 0 until cols && y in 0 until rows && grid[y][x] == 0x00) {
                grid[y][x] = metatile
            }
        }

        // Row 10: base
        setIfSky(startCol + 0, 10, METATILE_HILL_LEFT_BOTTOM)
        setIfSky(startCol + 1, 10, METATILE_HILL_BODY)
        setIfSky(startCol + 2, 10, METATILE_HILL_RIGHT_BOTTOM)

        // Row 9-8: peak
        setIfSky(startCol + 0, 9, METATILE_HILL_LEFT)
        setIfSky(startCol + 1, 9, METATILE_HILL_TOP)
        setIfSky(startCol + 2, 9, METATILE_HILL_RIGHT)
        setIfSky(startCol + 1, 8, METATILE_HILL_TOP)
    }

    /**
     * Draw a cloud at the given position.
     * width = number of middle sections (total cloud = width + 2 with caps)
     * Actually clouds are drawn as: left-cap, N middle tiles, right-cap (top row)
     * and bottom-left, N bottom-middle, bottom-right (bottom row)
     */
    private fun drawCloud(
        grid: Array<IntArray>,
        startCol: Int,
        row: Int,
        width: Int,
        rows: Int,
        cols: Int
    ) {
        if (row < 0 || row >= rows - 1 || width <= 0) return
        // Top row
        for (i in 0 until width) {
            val c = startCol + i
            if (c in 0 until cols && grid[row][c] == 0x00) {
                grid[row][c] = when {
                    width == 1 -> METATILE_CLOUD_TM
                    i == 0 -> METATILE_CLOUD_TL
                    i == width - 1 -> METATILE_CLOUD_TR
                    else -> METATILE_CLOUD_TM
                }
            }
        }
        // Bottom row
        for (i in 0 until width) {
            val c = startCol + i
            if (c in 0 until cols && row + 1 < rows && grid[row + 1][c] == 0x00) {
                grid[row + 1][c] = when {
                    width == 1 -> METATILE_CLOUD_BM
                    i == 0 -> METATILE_CLOUD_BL
                    i == width - 1 -> METATILE_CLOUD_BR
                    else -> METATILE_CLOUD_BM
                }
            }
        }
    }

    /**
     * Draw a bush at the given position (sits on ground).
     * width = total width in metatiles.
     */
    private fun drawBush(
        grid: Array<IntArray>,
        startCol: Int,
        row: Int,
        width: Int,
        rows: Int,
        cols: Int
    ) {
        if (row < 0 || row >= rows) return
        for (i in 0 until width) {
            val c = startCol + i
            if (c in 0 until cols && grid[row][c] == 0x00) {
                grid[row][c] = when {
                    width == 1 -> METATILE_BUSH_MID
                    i == 0 -> METATILE_BUSH_LEFT
                    i == width - 1 -> METATILE_BUSH_RIGHT
                    else -> METATILE_BUSH_MID
                }
            }
        }
    }

    // Metatile indices for scenery (defined in MetatileTable)
    private const val METATILE_HILL_BODY = 0x20
    private const val METATILE_HILL_TOP = 0x21
    private const val METATILE_HILL_LEFT = 0x27
    private const val METATILE_HILL_LEFT_BOTTOM = 0x28
    private const val METATILE_HILL_RIGHT = 0x29
    private const val METATILE_HILL_RIGHT_BOTTOM = 0x2A
    private const val METATILE_HILL_MIDDLE_BOTTOM = 0x2B
    private const val METATILE_CLOUD_TL = 0x0B   // cloud top-left
    private const val METATILE_CLOUD_TM = 0x22   // cloud top-middle
    private const val METATILE_CLOUD_TR = 0x0C   // cloud top-right
    private const val METATILE_CLOUD_BL = 0x0D   // cloud bottom-left
    private const val METATILE_CLOUD_BM = 0x23   // cloud bottom-middle
    private const val METATILE_CLOUD_BR = 0x0E   // cloud bottom-right
    private const val METATILE_BUSH_LEFT = 0x24   // bush left
    private const val METATILE_BUSH_MID = 0x25    // bush middle
    private const val METATILE_BUSH_RIGHT = 0x26  // bush right
    private const val METATILE_FENCE = 0x2C
    private const val METATILE_TREE_TOP_TALL_UPPER = 0x2D
    private const val METATILE_TREE_TOP_TALL_LOWER = 0x2E
    private const val METATILE_TREE_TOP_SHORT = 0x2F
    private const val METATILE_TREE_TRUNK = 0x17

    /**
     * Place a single object into the grid.
     */
    private fun placeObject(
        grid: Array<IntArray>,
        obj: LevelObject,
        absCol: Int,
        rows: Int,
        cols: Int,
        areaType: Int,
        areaStyle: Int
    ) {
        val row = obj.row
        val type = obj.type

        when {
            // Row 0x0C: SMB special row-12 objects.
            row == 0x0C -> {
                val specialId = (type shr 4) and 0x07
                val length = (type and 0x0F) + 1
                when (specialId) {
                    0x00 -> { // Hole_Empty
                        for (c in absCol until minOf(absCol + length, cols)) {
                            if (11 < rows) grid[11][c] = 0x00
                            if (12 < rows) grid[12][c] = 0x00
                        }
                    }
                    0x02, 0x03, 0x04 -> { // Bridge_High/Middle/Low
                        val bridgeRow = when (specialId) {
                            0x02 -> 6
                            0x03 -> 7
                            else -> 8
                        }
                        for (c in absCol until minOf(absCol + length, cols)) {
                            if (bridgeRow in 0 until rows) grid[bridgeRow][c] = 0x1C
                        }
                    }
                    0x05 -> { // Hole_Water
                        for (c in absCol until minOf(absCol + length, cols)) {
                            if (11 < rows) grid[11][c] = 0x18
                            if (12 < rows) grid[12][c] = 0x19
                        }
                    }
                    0x06, 0x07 -> { // QuestionBlockRow high/low
                        val blockRow = if (specialId == 0x06) 3 else 7
                        for (c in absCol until minOf(absCol + length, cols)) {
                            if (blockRow in 0 until rows) grid[blockRow][c] = 0x04
                        }
                    }
                    else -> {
                        // Pulley rope and other style objects are not fully modeled yet.
                    }
                }
            }

            // Row 0x0D: scenery change or area pointer (no visual effect on grid)
            row == 0x0D -> {
                // Row 13 special commands (d6-set objects in SMB parser)
                val specialId = type and 0x3F
                when (specialId) {
                    0x01 -> drawFlagpole(grid, absCol, rows, cols) // FlagpoleObject
                    else -> {
                        // Scroll locks, warp commands, intro pipe, etc. are logic-only here.
                    }
                }
            }

            // Row 0x0E: additional terrain/scenery (treat like 0x0D for now)
            row == 0x0E -> {
                // No visual effect on grid
            }

            // Row 0x0F: page skip command (no visual effect on grid)
            row == 0x0F -> {
                // Row 15 special objects: id in upper nibble, parameter in lower nibble.
                val specialId = (type shr 4) and 0x07
                val param = type and 0x0F
                when (specialId) {
                    0x02 -> drawCastle(grid, absCol, rows, cols)                  // CastleObject
                    0x03 -> drawSpecialStaircase(grid, absCol, param + 1, rows, cols) // StaircaseObject
                    else -> {
                        // Endless rope, balance rope, exit pipe, etc. not yet visualized.
                    }
                }
            }

            // Regular objects (row 0x00-0x0B)
            row < 0x0C -> {
                val classId = (type shr 4) and 0x07
                val param = type and 0x0F

                if (classId == 0) {
                    val smallId = param
                    val metatile = when (smallId) {
                        0x00, 0x01 -> 0x04 // question block
                        0x02, 0x03 -> 0x00 // hidden blocks (not shown)
                        0x04, 0x05, 0x06, 0x07, 0x08 -> 0x03 // brick variants
                        0x09 -> 0x06 // small water/pipe object
                        0x0A -> 0x05 // used block
                        0x0B -> 0x05 // jumpspring fallback visual
                        else -> 0x03
                    }
                    if (row < rows && absCol < cols) {
                        grid[row][absCol] = metatile
                    }
                    return
                }

                val length = param + 1
                when (classId) {
                    0x01 -> {
                        drawAreaStyleObject(
                            grid = grid,
                            startCol = absCol,
                            row = row,
                            param = param,
                            rows = rows,
                            cols = cols,
                            areaStyle = areaStyle
                        )
                    }
                    0x02 -> { // RowOfBricks
                        for (c in absCol until minOf(absCol + length, cols)) {
                            if (row < rows) grid[row][c] = 0x03
                        }
                    }
                    0x03 -> { // RowOfSolidBlocks
                        val mt = solidMetatile(areaType)
                        for (c in absCol until minOf(absCol + length, cols)) {
                            if (row < rows) grid[row][c] = mt
                        }
                    }
                    0x04 -> { // RowOfCoins
                        for (c in absCol until minOf(absCol + length, cols)) {
                            if (row < rows) grid[row][c] = 0x0A
                        }
                    }
                    0x05 -> { // ColumnOfBricks
                        for (r in row until minOf(row + length, rows)) {
                            if (absCol < cols) grid[r][absCol] = 0x03
                        }
                    }
                    0x06 -> { // ColumnOfSolidBlocks
                        val mt = solidMetatile(areaType)
                        for (r in row until minOf(row + length, rows)) {
                            if (absCol < cols) grid[r][absCol] = mt
                        }
                    }
                    0x07 -> { // VerticalPipe (decor/warp)
                        val pipeHeight = maxOf(2, (type and 0x07) + 1)
                        drawVerticalPipe(grid, absCol, row, pipeHeight, rows, cols)
                    }
                }
            }
        }
    }

    private fun drawAreaStyleObject(
        grid: Array<IntArray>,
        startCol: Int,
        row: Int,
        param: Int,
        rows: Int,
        cols: Int,
        areaStyle: Int
    ) {
        when (areaStyle) {
            // Tree/cloud ledge style
            0 -> {
                val width = (param + 1).coerceAtLeast(1)
                for (i in 0 until width) {
                    val c = startCol + i
                    if (c !in 0 until cols || row !in 0 until rows) continue
                    grid[row][c] = when {
                        width == 1 -> 0x15
                        i == 0 -> 0x14
                        i == width - 1 -> 0x16
                        else -> 0x15
                    }
                    // Draw supporting trunk under middle sections.
                    if (i in 1 until (width - 1)) {
                        for (r in (row + 1) until rows) {
                            if (grid[r][c] == 0x00) grid[r][c] = 0x17
                        }
                    }
                }
            }
            // Mushroom platform style
            1 -> {
                val width = (param + 1).coerceAtLeast(1)
                for (i in 0 until width) {
                    val c = startCol + i
                    if (c !in 0 until cols || row !in 0 until rows) continue
                    grid[row][c] = when {
                        width == 1 -> 0x15
                        i == 0 -> 0x14
                        i == width - 1 -> 0x16
                        else -> 0x15
                    }
                }
                val stemCol = (startCol + width / 2).coerceIn(0, cols - 1)
                for (r in (row + 1) until rows) {
                    if (grid[r][stemCol] == 0x00) grid[r][stemCol] = 0x17
                }
            }
            // Bullet-bill cannon style
            2 -> {
                val height = (param + 1).coerceAtLeast(1)
                if (row in 0 until rows && startCol in 0 until cols) {
                    grid[row][startCol] = 0x1D
                }
                for (i in 1 until height) {
                    val r = row + i
                    if (r !in 0 until rows || startCol !in 0 until cols) continue
                    grid[r][startCol] = 0x1E
                }
            }
            else -> {
                // Unknown style: fallback to a single solid block.
                if (row in 0 until rows && startCol in 0 until cols) {
                    grid[row][startCol] = 0x03
                }
            }
        }
    }

    private fun solidMetatile(areaType: Int): Int = when (areaType) {
        2 -> 0x03
        3 -> 0x10
        else -> 0x03
    }

    private fun drawVerticalPipe(
        grid: Array<IntArray>,
        col: Int,
        topRow: Int,
        height: Int,
        rows: Int,
        cols: Int
    ) {
        val h = height.coerceAtLeast(2)
        for (r in topRow until minOf(topRow + h, rows)) {
            if (r == topRow) {
                if (col < cols) grid[r][col] = 0x06
                if (col + 1 < cols) grid[r][col + 1] = 0x07
            } else {
                if (col < cols) grid[r][col] = 0x08
                if (col + 1 < cols) grid[r][col + 1] = 0x09
            }
        }
    }

    private fun drawFlagpole(
        grid: Array<IntArray>,
        col: Int,
        rows: Int,
        cols: Int
    ) {
        if (col !in 0 until cols) return
        val topRow = 2
        val bottomRow = minOf(10, rows - 1)
        if (topRow in 0 until rows) grid[topRow][col] = 0x12 // ball
        for (r in (topRow + 1)..bottomRow) {
            if (r in 0 until rows) grid[r][col] = 0x13 // shaft
        }
        // Approximate background pieces near the goal pole.
        if (bottomRow in 0 until rows && col + 1 < cols) {
            grid[bottomRow][col + 1] = 0x03 // small brick at pole base
        }
        val flagRow = (topRow + 2).coerceAtMost(bottomRow - 1)
        if (flagRow in 0 until rows && col + 1 < cols) {
            grid[flagRow][col + 1] = 0x1C // small marker to suggest flag cloth
        }
    }

    private fun drawSpecialStaircase(
        grid: Array<IntArray>,
        startCol: Int,
        steps: Int,
        rows: Int,
        cols: Int
    ) {
        val clampedSteps = steps.coerceIn(1, 16)
        for (step in 0 until clampedSteps) {
            val c = startCol + step
            if (c !in 0 until cols) continue
            val top = (11 - step).coerceAtLeast(0)
            for (r in top until minOf(rows, 13)) {
                grid[r][c] = 0x1B
            }
        }
    }

    private fun drawCastle(
        grid: Array<IntArray>,
        startCol: Int,
        rows: Int,
        cols: Int
    ) {
        val width = 5
        val height = 5
        val startRow = 7

        for (dy in 0 until height) {
            val r = startRow + dy
            if (r !in 0 until rows) continue
            for (dx in 0 until width) {
                val c = startCol + dx
                if (c !in 0 until cols) continue

                val mt = when {
                    dy == 0 -> if (dx == 0 || dx == width - 1) 0x0F else 0x10
                    dy == 2 && (dx == 1 || dx == width - 2) -> 0x11
                    else -> 0x10
                }
                grid[r][c] = mt
            }
        }
    }
}
