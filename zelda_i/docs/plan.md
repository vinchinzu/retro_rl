# Plan — Zelda I (NES)

## Goal

Advance from M5 (Clean power-on → Level 1 Triforce shard 1) toward a verified
continuous clear of The Legend of Zelda using the shared `adventure_common`
route graph.

## Next milestones

1. **M3 Level 2 entry** — finish 0x4A→0x3C and enter the Moon; define a
   RAM-backed room-ready stop.
2. **M4/M5 Level 2 chain** — build its isolated rooms, then pass each from the
   natural Level 1 completion predecessor.
3. **M6 route graph** — every required milestone (8 dungeons + Ganon path) has an owner and stop predicate.
4. **M7–M8** — continuous dry run + verified capture.

## Bottleneck

Door-path geometry to **0x3C** is probe-mapped (via 0x5A/5B/5C maze/5D@x52;
see `LEVEL2_ROUTE.md` + `LEVEL2_DOOR_HOPS`). Timed Clean attempt (2026-07-29)
dies on **0x5C with 0 hearts** after draining 3→0 across 0x48–0x5C; maze never
starts. Remaining: **heart-safe farm before 0x5A**, then maze controller, then
Level 2 interior (Ropes 0x28 → keys → Dodongo bombs → `triforce & 0x02`).
Full-game graph still needs seven more dungeons, inventory items, and Ganon.

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
- Level 1 completion is 2/2 isolated + 2/2 Clean power-on natural. See
  `docs/LEVEL1_ROUTE.md`; the required suffix ends on `triforce & 0x01`.
- Level 2 approach: engine settle to 0x37, walk prefix to 0x4A (see
  `docs/LEVEL2_ROUTE.md`). Avoid 0x79 dead-end; do not rely on mid-fanfare
  `Level1Complete.state` reloads.
- External walkthroughs/maps are approved planning accelerators. Keep their
  claims source-linked and separate from live emulator verification.
- Use `scripts/dungeon_lab.py` and `docs/DUNGEON_LAB.md` for future rooms.
