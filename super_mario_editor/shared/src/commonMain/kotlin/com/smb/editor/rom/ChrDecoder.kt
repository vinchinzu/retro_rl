package com.smb.editor.rom

/**
 * Decodes NES CHR (Character ROM) tiles.
 * Each tile is 2bpp, 16 bytes, 8x8 pixels.
 * Bytes 0-7 = bitplane 0, bytes 8-15 = bitplane 1.
 * Pixel value = (bit1 << 1) | bit0, giving 0-3.
 */
object ChrDecoder {
    const val TILE_SIZE = 16  // bytes per tile
    const val TILE_WIDTH = 8
    const val TILE_HEIGHT = 8

    /** Decode a single 8x8 tile from CHR data. Returns 64 ints (values 0-3). */
    fun decodeTile(chrData: ByteArray, tileIndex: Int): IntArray {
        val offset = tileIndex * TILE_SIZE
        val pixels = IntArray(64)
        for (row in 0 until 8) {
            val plane0 = chrData[offset + row].toInt() and 0xFF
            val plane1 = chrData[offset + row + 8].toInt() and 0xFF
            for (col in 0 until 8) {
                val bit0 = (plane0 shr (7 - col)) and 1
                val bit1 = (plane1 shr (7 - col)) and 1
                pixels[row * 8 + col] = (bit1 shl 1) or bit0
            }
        }
        return pixels
    }

    /** Decode all tiles from CHR ROM. Returns list of 64-int arrays. */
    fun decodeAllTiles(chrRom: ByteArray): List<IntArray> {
        val tileCount = chrRom.size / TILE_SIZE
        return (0 until tileCount).map { decodeTile(chrRom, it) }
    }
}
