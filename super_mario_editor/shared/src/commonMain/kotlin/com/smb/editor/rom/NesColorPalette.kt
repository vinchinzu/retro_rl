package com.smb.editor.rom

/**
 * Standard NES 2C02 NTSC palette and SMB area-type background palettes.
 * Uses the FCEUX default palette which closely matches the classic NES look.
 */
object NesColorPalette {
    /** FCEUX default NES 2C02 palette - 64 entries as 0xAARRGGBB */
    val PALETTE: IntArray = intArrayOf(
        // Row 0: 0x00-0x0F
        0xFF748484.toInt(), 0xFF2438EC.toInt(), 0xFF0000D8.toInt(), 0xFF4828D8.toInt(),
        0xFF9018A0.toInt(), 0xFFA00010.toInt(), 0xFF8C1000.toInt(), 0xFF5C2C00.toInt(),
        0xFF002C44.toInt(), 0xFF003C00.toInt(), 0xFF003814.toInt(), 0xFF143400.toInt(),
        0xFF143424.toInt(), 0xFF000000.toInt(), 0xFF000000.toInt(), 0xFF000000.toInt(),
        // Row 1: 0x10-0x1F
        0xFFBCBCBC.toInt(), 0xFF0070EC.toInt(), 0xFF2038EC.toInt(), 0xFF8000F0.toInt(),
        0xFFBC00BC.toInt(), 0xFFE4003C.toInt(), 0xFFD82000.toInt(), 0xFFC04C08.toInt(),
        0xFF886400.toInt(), 0xFF009400.toInt(), 0xFF008C04.toInt(), 0xFF008044.toInt(),
        0xFF008088.toInt(), 0xFF000000.toInt(), 0xFF000000.toInt(), 0xFF000000.toInt(),
        // Row 2: 0x20-0x2F
        0xFFFCFCFC.toInt(), 0xFF3CBCFC.toInt(), 0xFF5C94FC.toInt(), 0xFFC888FC.toInt(),
        0xFFF878F8.toInt(), 0xFFF05C68.toInt(), 0xFFF87858.toInt(), 0xFFFCA044.toInt(),
        0xFFF8B800.toInt(), 0xFFB8F818.toInt(), 0xFF5CC858.toInt(), 0xFF44D888.toInt(),
        0xFF48C4D8.toInt(), 0xFF787878.toInt(), 0xFF000000.toInt(), 0xFF000000.toInt(),
        // Row 3: 0x30-0x3F
        0xFFFCFCFC.toInt(), 0xFFA4E4FC.toInt(), 0xFFB8B8F8.toInt(), 0xFFD8B8F8.toInt(),
        0xFFF8B8F8.toInt(), 0xFFF8A4C0.toInt(), 0xFFF0D0B0.toInt(), 0xFFFCE0A8.toInt(),
        0xFFF8D878.toInt(), 0xFFD8F878.toInt(), 0xFFB8F8B8.toInt(), 0xFFB8F8D8.toInt(),
        0xFF00FCFC.toInt(), 0xFFF8D8F8.toInt(), 0xFF000000.toInt(), 0xFF000000.toInt()
    )

    // From smbdis.asm Ground/Water/Underground/CastlePaletteData (BG sub-palettes only).
    private val AREA_BG_PALETTE_INDICES: Map<Int, List<IntArray>> = mapOf(
        0 to listOf(
            intArrayOf(0x0F, 0x15, 0x12, 0x25),
            intArrayOf(0x0F, 0x3A, 0x1A, 0x0F),
            intArrayOf(0x0F, 0x30, 0x12, 0x0F),
            intArrayOf(0x0F, 0x27, 0x12, 0x0F)
        ),
        1 to listOf(
            intArrayOf(0x0F, 0x29, 0x1A, 0x0F),
            intArrayOf(0x0F, 0x36, 0x17, 0x0F),
            intArrayOf(0x0F, 0x30, 0x21, 0x0F),
            intArrayOf(0x0F, 0x27, 0x17, 0x0F)
        ),
        2 to listOf(
            intArrayOf(0x0F, 0x29, 0x1A, 0x09),
            intArrayOf(0x0F, 0x3C, 0x1C, 0x0F),
            intArrayOf(0x0F, 0x30, 0x21, 0x1C),
            intArrayOf(0x0F, 0x27, 0x17, 0x1C)
        ),
        3 to listOf(
            intArrayOf(0x0F, 0x30, 0x10, 0x00),
            intArrayOf(0x0F, 0x30, 0x10, 0x00),
            intArrayOf(0x0F, 0x30, 0x16, 0x00),
            intArrayOf(0x0F, 0x27, 0x17, 0x00)
        )
    )

    // smbdis.asm BackgroundColors table.
    private val BACKGROUND_COLORS = intArrayOf(0x22, 0x22, 0x0F, 0x0F, 0x0F, 0x22, 0x0F, 0x0F)

    /** Convert palette index array to ARGB colors */
    fun resolveColors(paletteIndices: IntArray): IntArray =
        IntArray(paletteIndices.size) { PALETTE[paletteIndices[it] and 0x3F] }

    /**
     * Get resolved ARGB palettes for an area type and background-color control.
     *
     * `bgColorCtrl` mirrors SMB BackgroundColorCtrl behavior:
     * - 0..3 => use area-type default background color
     * - 4..7 => use explicit background color entry
     */
    fun getAreaPalettes(areaType: Int, bgColorCtrl: Int = 0): List<IntArray> {
        val palettes = AREA_BG_PALETTE_INDICES[areaType] ?: AREA_BG_PALETTE_INDICES[1]!!
        val bgIndex = if (bgColorCtrl in 4..7) bgColorCtrl else areaType.coerceIn(0, 3)
        val bgColor = BACKGROUND_COLORS[bgIndex]
        return palettes.map { intArrayOf(bgColor, it[1], it[2], it[3]) }.map(::resolveColors)
    }
}
