package com.smb.editor.rom

/**
 * Parses iNES format NES ROMs.
 * iNES header: 16 bytes
 *   Bytes 0-3: "NES\x1A"
 *   Byte 4: PRG ROM banks (x16KB)
 *   Byte 5: CHR ROM banks (x8KB)
 *   Bytes 6-15: flags (mapper, mirroring, etc.)
 *
 * SMB uses NROM (mapper 0): flat 32KB PRG + 8KB CHR
 * PRG ROM: CPU $8000-$FFFF -> file offset 16 + (addr - $8000)
 */
class NesRom private constructor(
    val prgRom: ByteArray,
    val chrRom: ByteArray,
    val prgBanks: Int,
    val chrBanks: Int
) {
    /** Read a byte from PRG ROM using CPU address space ($8000-$FFFF) */
    fun readPrg(addr: Int): Int {
        require(addr in 0x8000..0xFFFF) { "PRG address out of range: ${addr.toString(16)}" }
        return prgRom[addr - 0x8000].toInt() and 0xFF
    }

    /** Read little-endian 16-bit word from PRG ROM */
    fun readPrgWord(addr: Int): Int {
        val lo = readPrg(addr)
        val hi = readPrg(addr + 1)
        return (hi shl 8) or lo
    }

    /** Read a range of bytes from PRG ROM */
    fun readPrgBytes(addr: Int, count: Int): ByteArray {
        val offset = addr - 0x8000
        return prgRom.copyOfRange(offset, offset + count)
    }

    /** Read a byte from CHR ROM by absolute offset */
    fun readChr(offset: Int): Int = chrRom[offset].toInt() and 0xFF

    companion object {
        fun fromBytes(data: ByteArray): NesRom {
            require(data.size >= 16) { "File too small for iNES header" }
            require(
                data[0] == 0x4E.toByte() && data[1] == 0x45.toByte() &&
                data[2] == 0x53.toByte() && data[3] == 0x1A.toByte()
            ) {
                "Not a valid iNES ROM (missing NES\\x1A header)"
            }

            val prgBanks = data[4].toInt() and 0xFF
            val chrBanks = data[5].toInt() and 0xFF
            val prgSize = prgBanks * 16384
            val chrSize = chrBanks * 8192

            require(data.size >= 16 + prgSize + chrSize) {
                "ROM file too small: expected ${16 + prgSize + chrSize} bytes, got ${data.size}"
            }

            val prgRom = data.copyOfRange(16, 16 + prgSize)
            val chrRom = if (chrSize > 0) {
                data.copyOfRange(16 + prgSize, 16 + prgSize + chrSize)
            } else {
                ByteArray(0)
            }

            return NesRom(prgRom, chrRom, prgBanks, chrBanks)
        }
    }
}
