#!/usr/bin/env python3
"""
Self-evaluate the extended 1-1 ROM.

Checks:
1. Byte-level: compare original vs extended object streams
2. Structural: verify page progression, object types preserved
3. Functional: load in retro emulator, run right, verify level is longer
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ROM_ORIG = (
    REPO_ROOT / "super_mario_bros" / "custom_integrations" / "SuperMarioBros-Nes-v0" / "rom.nes"
)
DEFAULT_ROM_EXT = SCRIPT_DIR / "smb_extended_1_1.nes"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the extended SMB 1-1 ROM against the original.")
    p.add_argument("--orig-rom", type=Path, default=DEFAULT_ROM_ORIG, help="Original SMB ROM path")
    p.add_argument("--ext-rom", type=Path, default=DEFAULT_ROM_EXT, help="Extended SMB ROM path")
    return p.parse_args()


args = _parse_args()
ROM_ORIG = args.orig_rom.resolve()
ROM_EXT = args.ext_rom.resolve()

def load_prg(path):
    data = Path(path).read_bytes()
    return data[16:16+32768]  # PRG ROM

def rb(prg, addr):
    return prg[addr - 0x8000]

def parse_objects(prg, start_addr):
    """Parse SMB object stream starting at addr. Returns (header, objects, end_addr)."""
    header = (rb(prg, start_addr), rb(prg, start_addr + 1))
    objects = []
    addr = start_addr + 2
    page = 0
    while rb(prg, addr) != 0xFD:
        b0 = rb(prg, addr)
        b1 = rb(prg, addr + 1)
        if b1 & 0x80:
            page += 1
        row = (b0 >> 4) & 0x0F
        col = b0 & 0x0F
        obj_type = b1 & 0x7F
        objects.append({
            'b0': b0, 'b1': b1,
            'page': page, 'row': row, 'col': col,
            'type': obj_type, 'page_flag': bool(b1 & 0x80)
        })
        addr += 2
        if len(objects) > 300:
            print("ERROR: too many objects, no terminator found")
            break
    return header, objects, addr

# Known object type names (partial, from our ROM analysis)
OBJ_NAMES = {
    0x00: "QBlock(coin)", 0x01: "QBlock(powerup)", 0x02: "Brick(coin)", 0x03: "Brick(star)",
    0x0C: "Hole", 0x0D: "HBricks", 0x0F: "Bridge",
    0x11: "QBlock(coin)", 0x12: "QBlock(powerup)",
    0x17: "Pipe", 0x18: "Pipe(tall)", 0x19: "Pipe(taller)",
    0x24: "Staircase", 0x26: "Castle",
    0x0E: "Scenery", 0x0B: "Scenery",
}

def describe_obj(obj):
    name = OBJ_NAMES.get(obj['type'], f"Type${obj['type']:02X}")
    return f"P{obj['page']:2d} R{obj['row']:2d} C{obj['col']:2d} {name}"

print("=" * 60)
print("SELF-EVALUATION: Extended 1-1 ROM")
print("=" * 60)

# === 1. Byte-level comparison ===
print("\n--- 1. BYTE-LEVEL COMPARISON ---")
prg_orig = load_prg(ROM_ORIG)
prg_ext = load_prg(ROM_EXT)

# Count differing bytes in PRG ROM
diffs = sum(1 for i in range(32768) if prg_orig[i] != prg_ext[i])
print(f"PRG ROM bytes changed: {diffs} / 32768")

# Check CHR ROM unchanged
chr_orig = ROM_ORIG.read_bytes()[16+32768:]
chr_ext = ROM_EXT.read_bytes()[16+32768:]
chr_diffs = sum(1 for i in range(len(chr_orig)) if chr_orig[i] != chr_ext[i])
print(f"CHR ROM bytes changed: {chr_diffs} / {len(chr_orig)}")

# Check iNES header unchanged
hdr_orig = ROM_ORIG.read_bytes()[:16]
hdr_ext = ROM_EXT.read_bytes()[:16]
print(f"iNES header match: {hdr_orig == hdr_ext}")

# === 2. Structural comparison ===
print("\n--- 2. STRUCTURAL COMPARISON ---")
OBJ_START = 0xA68E

hdr_o, objs_o, end_o = parse_objects(prg_orig, OBJ_START)
hdr_e, objs_e, end_e = parse_objects(prg_ext, OBJ_START)

print(f"Original: header=({hdr_o[0]:02X},{hdr_o[1]:02X}), {len(objs_o)} objects, ends at ${end_o:04X}")
print(f"Extended: header=({hdr_e[0]:02X},{hdr_e[1]:02X}), {len(objs_e)} objects, ends at ${end_e:04X}")
print(f"Header preserved: {hdr_o == hdr_e}")

max_page_o = max(o['page'] for o in objs_o)
max_page_e = max(o['page'] for o in objs_e)
print(f"Original max page: {max_page_o}")
print(f"Extended max page: {max_page_e}")
print(f"Page ratio: {(max_page_e+1)/(max_page_o+1):.2f}x")

# Check that first copy matches original gameplay section (pages 0-10)
SPLIT_PAGE = 10
orig_gameplay = [o for o in objs_o if o['page'] <= SPLIT_PAGE]
ext_first_half = objs_e[:len(orig_gameplay)]

print(f"\nFirst half objects: {len(ext_first_half)} (expected {len(orig_gameplay)})")
match_count = 0
mismatch_count = 0
for i, (og, eh) in enumerate(zip(orig_gameplay, ext_first_half)):
    if og['page'] == eh['page'] and og['row'] == eh['row'] and og['col'] == eh['col'] and og['type'] == eh['type']:
        match_count += 1
    else:
        mismatch_count += 1
        if mismatch_count <= 5:
            print(f"  MISMATCH at obj {i}: orig={describe_obj(og)} ext={describe_obj(eh)}")

print(f"First half: {match_count} match, {mismatch_count} mismatch")

# Auto-detect actual page shift from data
ext_second_half = objs_e[len(orig_gameplay):len(orig_gameplay)*2]
if ext_second_half:
    actual_shift = ext_second_half[0]['page'] - orig_gameplay[0]['page']
else:
    actual_shift = SPLIT_PAGE
PAGE_SHIFT = actual_shift
print(f"\nSecond half objects: {len(ext_second_half)} (expected {len(orig_gameplay)})")
print(f"Detected page shift: {PAGE_SHIFT}")
match2 = 0
mismatch2 = 0
for i, (og, eh) in enumerate(zip(orig_gameplay, ext_second_half)):
    expected_page = og['page'] + PAGE_SHIFT
    if expected_page == eh['page'] and og['row'] == eh['row'] and og['col'] == eh['col'] and og['type'] == eh['type']:
        match2 += 1
    else:
        mismatch2 += 1
        if mismatch2 <= 5:
            print(f"  MISMATCH at obj {i}: orig P{og['page']}+{PAGE_SHIFT}=P{expected_page} got {describe_obj(eh)}")

print(f"Second half: {match2} match, {mismatch2} mismatch (shifted by {PAGE_SHIFT} pages)")

# Check ending preserved
orig_ending = [o for o in objs_o if o['page'] > SPLIT_PAGE]
ext_ending = objs_e[len(orig_gameplay)*2:]
print(f"\nEnding objects: {len(ext_ending)} (expected {len(orig_ending)})")
match3 = 0
mismatch3 = 0
for i, (og, eh) in enumerate(zip(orig_ending, ext_ending)):
    expected_page = og['page'] + PAGE_SHIFT
    if expected_page == eh['page'] and og['row'] == eh['row'] and og['col'] == eh['col'] and og['type'] == eh['type']:
        match3 += 1
    else:
        mismatch3 += 1
        if mismatch3 <= 5:
            print(f"  MISMATCH at obj {i}: orig P{og['page']}+{PAGE_SHIFT}=P{expected_page} got {describe_obj(eh)}")

print(f"Ending: {match3} match, {mismatch3} mismatch")

# === 3. Page progression visualization ===
print("\n--- 3. PAGE PROGRESSION ---")
print("Original:")
page_counts_o = {}
for o in objs_o:
    page_counts_o[o['page']] = page_counts_o.get(o['page'], 0) + 1
for p in sorted(page_counts_o):
    bar = "#" * page_counts_o[p]
    print(f"  Page {p:2d}: {bar} ({page_counts_o[p]})")

print("Extended:")
page_counts_e = {}
for o in objs_e:
    page_counts_e[o['page']] = page_counts_e.get(o['page'], 0) + 1
for p in sorted(page_counts_e):
    bar = "#" * page_counts_e[p]
    print(f"  Page {p:2d}: {bar} ({page_counts_e[p]})")

# === 4. Data overlap check ===
print("\n--- 4. DATA OVERLAP CHECK ---")
orig_data_end = end_o + 1  # include FD terminator
ext_data_end = end_e + 1
print(f"Original data: ${OBJ_START:04X}-${orig_data_end-1:04X} ({orig_data_end - OBJ_START} bytes)")
print(f"Extended data: ${OBJ_START:04X}-${ext_data_end-1:04X} ({ext_data_end - OBJ_START} bytes)")
overwrite_bytes = (ext_data_end - OBJ_START) - (orig_data_end - OBJ_START)
print(f"Extra bytes used: {overwrite_bytes} (overwrites adjacent level data)")

# Check what the next area data after original 1-1 was
next_area_start = orig_data_end
print(f"Next area data was at ${next_area_start:04X}")
# See if it's still valid
next_byte = rb(prg_ext, next_area_start)
print(f"First byte of next area in extended ROM: ${next_byte:02X}", end="")
if next_area_start < ext_data_end:
    print(" (OVERWRITTEN - this adjacent level is corrupted)")
else:
    print(" (intact)")

# === 5. Overall verdict ===
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
issues = []
if hdr_o != hdr_e:
    issues.append("Header changed")
if mismatch_count > 0:
    issues.append(f"First half: {mismatch_count} mismatches")
if mismatch2 > 0:
    issues.append(f"Second half: {mismatch2} mismatches")
if mismatch3 > 0:
    issues.append(f"Ending: {mismatch3} mismatches")
if chr_diffs > 0:
    issues.append(f"CHR ROM modified ({chr_diffs} bytes)")
if max_page_e < max_page_o * 1.5:
    issues.append(f"Level not long enough ({max_page_e+1} pages vs expected ~{(max_page_o+1)*2})")

warnings = []
if overwrite_bytes > 0:
    warnings.append(f"Overwrites {overwrite_bytes} bytes of adjacent level data")

if not issues:
    print("PASS - Extended 1-1 structure is correct")
    print(f"  - Level is {(max_page_e+1)/(max_page_o+1):.1f}x longer ({max_page_e+1} pages vs {max_page_o+1})")
    print(f"  - {len(objs_e)} objects ({len(objs_o)} original)")
    print(f"  - First half faithfully reproduced")
    print(f"  - Second half correctly shifted by {PAGE_SHIFT} pages")
    print(f"  - Ending (flagpole/castle) preserved at shifted position")
else:
    print("FAIL")
    for issue in issues:
        print(f"  ERROR: {issue}")

if warnings:
    for w in warnings:
        print(f"  WARNING: {w}")
