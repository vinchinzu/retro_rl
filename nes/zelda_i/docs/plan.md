# Plan — Zelda I (NES)

## Goal

Advance from M5 (Clean power-on → Level 1 Triforce shard 1) toward a verified
continuous clear of The Legend of Zelda using the shared `retro_harness.adventure`
route graph.

Tracker: **`bd ready -l zelda_i`**. Process: `docs/tasks/PROCESS.md`.

## Strategy (SM lessons + ~100 beads)

1. **Assisted first pass** — `--infinite-life` Survival assist unlocks
   overworld corridors and dungeon interiors without heart starvation.
   Evidence is dual-track (not Clean STATUS).
2. **Pure-first rooms** — isolated controller → natural-entry → graph promote.
3. **Expand beads at the tip** — epics for L2–L9 + OW prep + Death Mountain;
   spawn room children when that dungeon is active (~80–120 total by credits).
4. **Clean pass later** — heart farm / combat harden per segment after geometry
   is known; never demote assisted greens into Clean rows.
5. **Adventure harness** — keep RAM/combat local; `RouteGraph`, `NamedRoute`,
   legs, waypoints stay on `retro_harness.adventure`.

## Next milestones

1. **L2 tip (serial)** — rooms → Dodongo → `triforce & 0x02` (assisted then Clean).
2. **Parallel pure** — isolated pure from L3/L5/L6 checkpoints while L2 tip runs
   (`docs/tasks/QUEUE.md`, `bd ready -l zelda_i`):
   - L3 west key **Clean** (`Level3WestKey`) → Raft → Manhandla → TF `0x04`
   - L5 0x66 clear **Clean** (`Level5Cleared66`) → whistle → Digdogger → TF `0x10`
   - L6 east key **assisted** (`Level6EastKey`) → Rod → Gohma → TF `0x20`
   - L8 bush `0x6D` live; candle shop residual → enter
3. **Clean door path L2** — heart-safe farm before 0x5A (parallel; not tip-blocking).
4. **M6 route graph** — milestones in adventure; `routes_later.py` stubs exist.
5. **M7–M8** — continuous dry run + verified capture (assisted then Clean).

## Bottleneck

**L2 interior tip** remains the serial path to full clear (other agents).
Parallel pure tracks L3/L5/L6/L8 from checkpoints without blocking L2.
See `docs/tasks/QUEUE.md`.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Graph package: `retro_harness.adventure` (first consumer; second consumer later for promotion of richer APIs).
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
