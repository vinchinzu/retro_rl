# Plan — Zelda I (NES)

## Goal

Advance from M5 (boot → Level 1 room 0x54 cleared) toward a verified
continuous clear of The Legend of Zelda using the shared `adventure_common`
route graph.

## Next milestones

1. **M3 Level 1 room 0x52** — clear or route through the six-Keese west
   branch; room 0x54's east doorway is blocked after clear.
2. **M4/M5 Level 1 chain** — extend the verified power-on chain from the
   0x53 key toward the map and Aquamentus.
3. **M6 route graph** — every required milestone (8 dungeons + Ganon path) has an owner and stop predicate.
4. **M7–M8** — continuous dry run + verified capture.

## Bottleneck

The room 0x52 west branch and onward map route. Room 0x54 is cleared but its
east doorway is blocked; it produces no known inventory change
(`RoomItemId=0x16`). Overworld combat robustness remains thin (periodic sword
swings only).

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Graph package: `adventure_common` (first consumer; second consumer later for promotion of richer APIs).
- Sword cave geometry (probe-stable): approach ~(60,100) on 0x77, cave mode 11, align x=120, walk up to sword, exit down.
- Level 1 overworld path (probe-stable 2026-07-28): east-then-north via 0x78/68/58/48/38 → 0x37; door enter UP at x≈112 from y≈140.
- Dungeon prefix (probe-stable 2026-07-28):
  `0x73→E 0x74→first key→W 0x73→unlock N→0x63→clear→N 0x53→clear/key`.
- Cleared 0x53 branches `W→0x52` (six Keese, item `0x03`) and `E→0x54`
  (eight Keese, item `0x16`); north is closed.
- Room 0x54 clear is 2/2 isolated + 2/2 natural; west returns to 0x53 and east
  is blocked.
- Use `scripts/dungeon_lab.py` and `docs/DUNGEON_LAB.md` for future rooms.
