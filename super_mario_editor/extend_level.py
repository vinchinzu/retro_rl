#!/usr/bin/env python3
"""
Extend SMB World 1-1 to be twice as long.

Strategy:
1. Read the original 1-1 object data (verified at PRG $A68E, 100 bytes)
2. Duplicate every object with page numbers shifted by the original max page + 1
3. Strip the flagpole/castle/end-sequence from the first copy
4. Write extended data to unused ROM space
5. Patch the area data pointer to use the new extended data
6. Save as a new .nes ROM file
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ROM_PATH = (
    REPO_ROOT / "super_mario_bros" / "custom_integrations" / "SuperMarioBros-Nes-v0" / "rom.nes"
)
DEFAULT_OUT_PATH = SCRIPT_DIR / "smb_extended_1_1.nes"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create extended SMB 1-1 ROM variant.")
    p.add_argument("--rom", type=Path, default=DEFAULT_ROM_PATH, help="Input SMB ROM path")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT_PATH, help="Output ROM path")
    return p.parse_args()


args = _parse_args()
ROM_PATH = args.rom.resolve()
OUT_PATH = args.output.resolve()
rom = bytearray(ROM_PATH.read_bytes())
prg_offset = 16  # iNES header size

def prg_to_file(addr):
    return addr - 0x8000 + prg_offset

def rb(addr):
    return rom[prg_to_file(addr)]

# === 1. Read original 1-1 object data ===
OBJ_START = 0xA68E
obj_addr = OBJ_START
header = bytes([rb(obj_addr), rb(obj_addr + 1)])

# Parse all objects
raw_objects = []  # list of (b0, b1) pairs
addr = OBJ_START + 2  # skip 2-byte header
while rb(addr) != 0xFD:
    raw_objects.append((rb(addr), rb(addr+1)))
    addr += 2
    if len(raw_objects) > 200:
        break

print(f"Original 1-1: {len(raw_objects)} objects, {addr - OBJ_START + 1} bytes")

# Compute page numbers for each object
pages = []
current_page = 0
for b0, b1 in raw_objects:
    if b1 & 0x80:
        current_page += 1
    pages.append(current_page)

max_page = max(pages)
print(f"Original max page: {max_page}")

# Find where the end-of-level objects are (flagpole, castle, staircase)
# Objects on pages 11-12 with row >= 13 are likely end-sequence
# Keep everything up to the ascending staircase/flagpole area
# Actually, let's find the staircase block that comes before the flagpole
# Page 12 objects: vertical bricks and horizontal blocks (ground extension + castle)

# Split: pages 0-10 = "gameplay", pages 11-12 = "ending"
SPLIT_PAGE = 10  # everything through page 10 is gameplay

first_half = []
end_section = []
for i, (b0, b1) in enumerate(raw_objects):
    if pages[i] <= SPLIT_PAGE:
        first_half.append((b0, b1, pages[i]))
    else:
        end_section.append((b0, b1, pages[i]))

print(f"Gameplay objects (pages 0-{SPLIT_PAGE}): {len(first_half)}")
print(f"Ending objects (pages {SPLIT_PAGE+1}+): {len(end_section)}")

# === 2. Build extended level ===
# Structure: header + first_half + shifted_first_half + shifted_end_section + FD

PAGE_SHIFT = SPLIT_PAGE  # shift second copy so it starts right after page SPLIT_PAGE
# SMB page flags are relative (+1), so gaps must be exactly 1 between last/first objects

def rebuild_objects_with_page_reset(objects_with_pages):
    """Rebuild object bytes, re-encoding page advance flags.

    SMB page flags are RELATIVE: each flag means 'advance 1 page'.
    If we need to skip >1 page, insert invisible padding objects
    (row 15 col 0 = below screen, harmless) to bridge the gap.
    """
    result = []
    prev_page = 0  # page counter starts at 0
    for b0, b1_orig, target_page in objects_with_pages:
        b1_base = b1_orig & 0x7F
        gap = target_page - prev_page
        if gap > 1:
            # Insert (gap-1) padding objects to bridge the gap
            for _ in range(gap - 1):
                # Row 15, col 0 with page flag = invisible off-screen object
                result.append((0xF0, 0x80))  # page advance + harmless type 0
            # Now the actual object needs one more page advance
            b1_new = b1_base | 0x80
        elif gap == 1:
            b1_new = b1_base | 0x80  # set page advance
        else:
            b1_new = b1_base  # same page, no flag
        result.append((b0, b1_new))
        prev_page = target_page
    return result

# First copy: pages 0 through SPLIT_PAGE (gameplay section)
first_copy = [(b0, b1, p) for b0, b1, p in first_half]

# Second copy: same gameplay but with pages shifted
second_copy = [(b0, b1, p + PAGE_SHIFT) for b0, b1, p in first_half]

# End section: shift by PAGE_SHIFT
shifted_end = [(b0, b1, p + PAGE_SHIFT) for b0, b1, p in end_section]

# Combine
all_objects = first_copy + second_copy + shifted_end

print(f"Extended level: {len(all_objects)} objects")
new_max = max(p for _, _, p in all_objects)
print(f"Extended max page: {new_max}")

# Rebuild with correct page flags
rebuilt = rebuild_objects_with_page_reset(all_objects)

# Build the raw byte stream
new_data = bytearray(header)  # 2-byte header
for b0, b1 in rebuilt:
    new_data.append(b0)
    new_data.append(b1)
new_data.append(0xFD)  # terminator

print(f"New data size: {len(new_data)} bytes")

# === 3. Find free space in ROM ===
# The SMB ROM has some padding at the end of the PRG ROM.
# Let's look for a block of 0xFF bytes large enough.
prg_size = 32768
min_space = len(new_data) + 16  # some margin

best_start = None
best_len = 0
run_start = None
run_len = 0

for i in range(prg_size):
    byte = rom[prg_offset + i]
    if byte == 0xFF or byte == 0x00:
        if run_start is None:
            run_start = i
            run_len = 1
        else:
            run_len += 1
        if run_len > best_len:
            best_len = run_len
            best_start = run_start
    else:
        run_start = None
        run_len = 0

print(f"\nBest free space: {best_len} bytes at PRG ${best_start + 0x8000:04X}")

if best_len < min_space:
    # Alternative: overwrite the original data area + following levels
    # Since we're making a ROM hack, we can overwrite adjacent area data
    print(f"WARNING: Not enough free space ({best_len} < {min_space})")
    print("Will overwrite original 1-1 data and adjacent areas")
    write_addr = OBJ_START
else:
    write_addr = best_start + 0x8000
    
print(f"Writing extended data at PRG ${write_addr:04X}")

# === 4. Patch the ROM ===
# Write the new level data
file_offset = prg_to_file(write_addr)
for i, byte in enumerate(new_data):
    rom[file_offset + i] = byte

# Now we need to patch the pointer that points to 1-1's object data.
# From our analysis, the pointer table at $9D28/$9D48 entry [12] had Lo=$8E.
# With hi offset adjustment, Lo[12]+Hi[13] gives $A68E.
# 
# The actual pointer structure uses:
# L_AreaData at $9D28 (low bytes) and H_AreaData offset by area type
# For ground areas, if the offset is such that Lo[12] Hi[12+offset] = $A68E,
# then we need Lo[12] = lo(write_addr), Hi[12+offset] = hi(write_addr)
#
# Simpler approach: just scan the ROM for bytes that encode $A68E and patch them.
# The pointer is stored as separate Lo and Hi bytes in the tables.

new_lo = write_addr & 0xFF
new_hi = (write_addr >> 8) & 0xFF

# Find and patch the pointer (Lo=$8E at known table positions)
# We know Lo[12] in the area data low table at $9D28 = byte at $9D28+12 = $9D34
# And the high byte is at $9D48+13 = $9D55 (with area type offset of +1)
lo_file_offset = prg_to_file(0x9D34)
hi_file_offset = prg_to_file(0x9D55)

old_lo = rom[lo_file_offset]
old_hi = rom[hi_file_offset]
print(f"\nPointer table: Lo at $9D34={old_lo:02X}, Hi at $9D55={old_hi:02X}")
print(f"Expected: Lo=$8E Hi=$A6 -> ${old_hi:02X}{old_lo:02X}")

if old_lo == 0x8E and old_hi == 0xA6:
    rom[lo_file_offset] = new_lo
    rom[hi_file_offset] = new_hi
    print(f"Patched pointer: ${new_hi:02X}{new_lo:02X}")
elif write_addr == OBJ_START:
    print("Writing in-place, no pointer patch needed")
else:
    print(f"WARNING: Pointer bytes don't match expected values!")
    print(f"Got Lo={old_lo:02X} Hi={old_hi:02X}, trying alternate locations...")
    # Try the original table interpretation: $9D28[12] and $9D48[12]
    for lo_off, hi_off in [(0x9D34, 0x9D55), (0x9D34, 0x9D54), (0x9D34, 0x9D56)]:
        lo_val = rom[prg_to_file(lo_off)]
        hi_val = rom[prg_to_file(hi_off)]
        if lo_val == 0x8E and hi_val == 0xA6:
            rom[prg_to_file(lo_off)] = new_lo
            rom[prg_to_file(hi_off)] = new_hi
            print(f"Patched at Lo=${lo_off:04X} Hi=${hi_off:04X}")
            break

# === 5. Save ===
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("wb") as f:
    f.write(rom)
print(f"\nSaved extended ROM to: {OUT_PATH}")
print(f"ROM size: {len(rom)} bytes")

# === 6. Summary ===
print(f"\n=== SUMMARY ===")
print(f"Original 1-1: {len(raw_objects)} objects, max page {max_page}")
print(f"Extended 1-1: {len(all_objects)} objects, max page {new_max}")
print(f"Level is now ~{(new_max+1)/(max_page+1):.1f}x longer")
print(f"Original data: ${OBJ_START:04X} ({addr - OBJ_START + 1} bytes)")
print(f"Extended data: ${write_addr:04X} ({len(new_data)} bytes)")
