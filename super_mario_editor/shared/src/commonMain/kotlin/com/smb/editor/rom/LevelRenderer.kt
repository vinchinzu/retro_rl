package com.smb.editor.rom

/**
 * Renders a metatile grid to an ARGB pixel buffer.
 */
object LevelRenderer {
    const val METATILE_SIZE = 16  // 16x16 pixels per metatile
    const val TILE_SIZE = 8      // 8x8 pixels per CHR tile

    /**
     * Render a level tile grid to an ARGB pixel buffer.
     *
     * @param grid 2D array [row][col] of metatile indices
     * @param metatiles Metatile definitions
     * @param chrTiles Decoded CHR tiles (list of 64-int arrays)
     * @param palettes 4 sub-palettes of 4 ARGB colors each
     * @return ARGB pixel array with dimensions
     */
    fun renderGrid(
        grid: Array<IntArray>,
        metatiles: List<Metatile>,
        chrTiles: List<IntArray>,
        palettes: List<IntArray>
    ): RenderResult {
        val gridRows = grid.size
        val gridCols = if (grid.isNotEmpty()) grid[0].size else 0
        val width = gridCols * METATILE_SIZE
        val height = gridRows * METATILE_SIZE
        val pixels = IntArray(width * height)

        // Fill with background color (first color of first palette)
        val bgColor = if (palettes.isNotEmpty() && palettes[0].isNotEmpty()) {
            palettes[0][0]
        } else {
            0xFF000000.toInt()
        }
        pixels.fill(bgColor)

        for (row in 0 until gridRows) {
            for (col in 0 until gridCols) {
                val metatileIdx = grid[row][col]
                if (metatileIdx < 0 || metatileIdx >= metatiles.size) continue

                val mt = metatiles[metatileIdx]
                val palette = if (mt.paletteIndex in palettes.indices) {
                    palettes[mt.paletteIndex]
                } else {
                    continue
                }

                val px = col * METATILE_SIZE
                val py = row * METATILE_SIZE

                // SMB metatile tables store tiles in column-major order:
                // [top-left, bottom-left, top-right, bottom-right].
                drawTile(pixels, width, px, py, chrTiles, mt.topLeft, palette)
                drawTile(pixels, width, px + TILE_SIZE, py, chrTiles, mt.bottomLeft, palette)
                drawTile(pixels, width, px, py + TILE_SIZE, chrTiles, mt.topRight, palette)
                drawTile(pixels, width, px + TILE_SIZE, py + TILE_SIZE, chrTiles, mt.bottomRight, palette)
            }
        }

        return RenderResult(pixels, width, height)
    }

    private fun drawTile(
        pixels: IntArray,
        stride: Int,
        x: Int,
        y: Int,
        chrTiles: List<IntArray>,
        tileIdx: Int,
        palette: IntArray
    ) {
        if (tileIdx < 0 || tileIdx >= chrTiles.size) return
        val tile = chrTiles[tileIdx]

        for (row in 0 until TILE_SIZE) {
            for (col in 0 until TILE_SIZE) {
                val colorIdx = tile[row * TILE_SIZE + col]
                if (colorIdx == 0) continue  // transparent (color 0 = background)

                val px = x + col
                val py = y + row
                if (px < stride && py >= 0) {
                    val offset = py * stride + px
                    if (offset in pixels.indices) {
                        pixels[offset] = if (colorIdx < palette.size) {
                            palette[colorIdx]
                        } else {
                            0xFFFF00FF.toInt()
                        }
                    }
                }
            }
        }
    }

    /**
     * Convenience: render a full level from ROM.
     *
     * NES CHR ROM has two 4KB pattern tables.
     * SMB background metatiles reference pattern table at $1000
     * (CHR bank 1: tile IDs 0x100-0x1FF in raw CHR ROM order).
     */
    fun renderLevel(rom: NesRom, level: com.smb.editor.data.LevelData): RenderResult {
        val grid = LevelDecoder.expandToGrid(level)
        val metatiles = MetatileTable.getMetatiles()
        val allTiles = ChrDecoder.decodeAllTiles(rom.chrRom)
        // Background tiles are in CHR bank 1 (tiles 256-511).
        val bgTiles = if (allTiles.size >= 512) allTiles.subList(256, 512) else allTiles
        val palettes = NesColorPalette.getAreaPalettes(level.header.areaType)
        return renderGrid(grid, metatiles, bgTiles, palettes)
    }
}

data class RenderResult(
    val pixels: IntArray,
    val width: Int,
    val height: Int
)
