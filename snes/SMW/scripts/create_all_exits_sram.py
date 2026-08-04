"""Create a checksum-valid Super Mario World all-exits SRAM fixture.

The vanilla SMW save format stores three 141-byte save buffers plus a
little-endian checksum, then stores mirrored copies at +0x1AD.  The checksum
is valid when sum(save_buffer) + checksum == 0x5A5A.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SRAM_SIZE = 0x800
SAVE_BUFFER_SIZE = 0x8D
CHECKSUM_TARGET = 0x5A5A
SLOT_BASES = (0x000, 0x08F, 0x11E)
MIRROR_DELTA = 0x1AD

# SMWDisX bank_00.asm:InitPlayerOverworldData.
# Starts both players at Yoshi's House on Yoshi's Island.
INITIAL_OVERWORLD_PLAYER_DATA = bytes.fromhex(
    "01 01"  # SaveDataBufferSubmap
    " 02 00 02 00"  # SaveDataBufferAni
    " 68 00 78 00 68 00 78 00"  # SaveDataBufferXPos/YPos
    " 06 00 07 00 06 00 07 00"  # SaveDataBufferXPosPtr/YPosPtr
)


def build_save_buffer() -> bytearray:
    """Return a 141-byte fully unlocked save buffer."""
    buf = bytearray(SAVE_BUFFER_SIZE)

    # SaveDataBuffer / OWLevelTileSettings: all overworld paths and level
    # completion flags open. This is intentionally broad for editor/testing use.
    buf[0x00:0x60] = bytes([0xFF]) * 0x60

    # SaveDataBufferEvents / OWEventsActivated: activate every event bit.
    buf[0x60:0x6F] = bytes([0xFF]) * 0x0F

    # Player overworld position/animation data.
    buf[0x6F:0x85] = INITIAL_OVERWORLD_PLAYER_DATA

    # SaveDataBufferSwitches: all four switch palaces active.
    buf[0x85:0x89] = bytes([0xFF]) * 4

    # SaveDataBufferExits / ExitsCompleted: vanilla full-completion display.
    buf[0x8C] = 0x60
    return buf


def checksum_for(save_buffer: bytes) -> int:
    return (CHECKSUM_TARGET - sum(save_buffer)) & 0xFFFF


def write_slot(sram: bytearray, base: int, save_buffer: bytes) -> None:
    checksum = checksum_for(save_buffer)
    for offset in (base, base + MIRROR_DELTA):
        sram[offset : offset + SAVE_BUFFER_SIZE] = save_buffer
        sram[offset + SAVE_BUFFER_SIZE] = checksum & 0xFF
        sram[offset + SAVE_BUFFER_SIZE + 1] = checksum >> 8


def validate_slot(sram: bytes, base: int) -> None:
    for offset in (base, base + MIRROR_DELTA):
        save_buffer = sram[offset : offset + SAVE_BUFFER_SIZE]
        checksum = sram[offset + SAVE_BUFFER_SIZE] | (sram[offset + SAVE_BUFFER_SIZE + 1] << 8)
        total = (sum(save_buffer) + checksum) & 0xFFFF
        if total != CHECKSUM_TARGET:
            raise ValueError(
                f"invalid checksum at 0x{offset:03X}: got 0x{total:04X}, "
                f"expected 0x{CHECKSUM_TARGET:04X}"
            )


def create_sram() -> bytes:
    sram = bytearray(SRAM_SIZE)
    save_buffer = build_save_buffer()
    for base in SLOT_BASES:
        write_slot(sram, base, save_buffer)
    for base in SLOT_BASES:
        validate_slot(sram, base)
    return bytes(sram)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="SMW/custom_integrations/SuperMarioWorld-Snes-v0/rom.srm",
        help="SRAM path to write",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(create_sram())
    print(f"Wrote {output} ({SRAM_SIZE} bytes)")
    print("Slots: 3 valid mirrored saves, 96 exits, all events, all switches")


if __name__ == "__main__":
    main()
