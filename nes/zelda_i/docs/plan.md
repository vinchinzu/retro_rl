# Plan — Zelda I (NES)

## Goal

Advance from M5 (Clean power-on → Level 1 Triforce shard 1) toward a verified
continuous clear of The Legend of Zelda using the shared `retro_harness.adventure`
route graph.

Tracker: **`bd ready -l zelda_i`**. Process: `docs/tasks/PROCESS.md`.

## Next pass — Level 5 east key (2026-08-14)

Verified assisted stopping point (Survival refill only; not Clean STATUS):

- `Level4Complete` → real overworld return `0x45` → Lost Hills → settled L5
  entry `0x76`: **1/1**, 5,031 path frames. Inventory stayed bombs=7,
  Raft=1, Stepladder=1, Triforce=`0x0c`; assist reported zero progression and
  capacity writes. Runner: `scripts/run_l4_to_l5.py`; checkpoint:
  `Level5EntranceFromL4`.
- `Level5EntranceFromL4` → north `0x66` → clear three Gibdos → fixed key:
  **1/1**, 1,254 frames, keys 0→1. Runner: `scripts/run_level5_clear66.py`;
  checkpoint: `Level5Cleared66`.

Do not rerun those predecessors first. Resume with exactly one diagnostic run:

```bash
uv run python nes/zelda_i/scripts/run_level5_east_key.py \
  --from-state Level5Cleared66 --keep-keys --infinite-life --save-state --trials 1
```

The last attempt returned `0x66→0x76` with keys=1, then stalled before `0x77`.
The runner now emits `prefix_trail` samples every 250 frames; inspect those and
`recordings/l5_east_key_t0_isolated.png` before changing geometry. Expected
route: finish the `0x66` ladder crossing DOWN, align x≈120, exit SOUTH to
`0x76`, approach the east wall on y≈157, align to door channel y≈141, then
RIGHT through the key door. Predecessors keep keys by default; `--keep-keys`
is optional explicit safety. Do not poke doors or keys, add random jitter, or
write inventory. After this is green, continue serially toward the Whistle; do
not detour into the known `0x67` Bubble residual.

## Strategy (finish easy → then tune)

**Order of work (agents):** pathfinding and puzzle solving first → full-game
route under Survival assist → Clean combat/heart harden using damage heatmaps.

1. **Infinite life + damage tracking** — `--infinite-life` Survival assist
   (`UnlimitedHealthAssist`) keeps agents alive. Telemetry records
   `total_damage`, `damage_by_location`, samples (see ASSIST_CONTRACT). Not
   Clean STATUS.
2. **Path + puzzles first** — overworld hops, door geometry, keys, bomb walls,
   push-blocks, item gates. Do **not** block route progress on sword kiting.
3. **Pure-first rooms** — isolated controller → natural-entry → graph promote
   (geometry + stop predicates; combat only as needed to open doors).
4. **Expand beads at the tip** — epics for L2–L9 + OW prep + Death Mountain;
   spawn room children when that dungeon is active (~80–120 total by credits).
5. **Clean pass later** — rank rooms by assist `damage_by_location`; heart farm /
   combat harden only after geometry is known. Never demote assisted greens
   into Clean STATUS rows.
6. **Adventure harness** — keep RAM/combat local; `RouteGraph`, `NamedRoute`,
   legs, waypoints stay on `retro_harness.adventure`.

## Next milestones

1. **L5 tip (serial)** — east key → Whistle → Digdogger → `triforce & 0x10`.
2. **Later dungeons** — continue L6–L8 under Survival assist before returning
   to Clean combat hardening.
3. **M6 route graph** — compose the assisted checkpoint chain into reusable
   route legs, then run a continuous dry run.
4. **M7–M8** — verified full-game capture (assisted first, Clean later).

## Bottleneck

**L5 east key geometry** is the current serial boundary. The predecessor chain
through L4 and the first `0x66` key are verified; the next pass starts from
`Level5Cleared66` and must use the new sampled trace instead of broad probes.

## Video / watchability (2026-08-06)

Hitbox-gated sword + faster boot landed (not a STATUS promote):

- `combat.py` sword rectangle + `should_swing_at`; dungeon + L1 early rooms
  slash only in blade range / contact (patrol walks clean).
- OW `walk_or_swing` — no air-swings on empty screens (`nav_common` /
  `overworld_nav` / `ow_path`).
- Boot `BOOT_PERIOD=50` (~565f ready vs ~1749); YouTube intro 90f; cave
  dialog idle 180f.

Residual room-by-room combat polish only if a clear regresses under hitbox gate.
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

- Dungeon door-graph template: `door_graph.py` (`LEVEL_2_DOOR_GRAPH` seed). See `docs/DUNGEON_LAB.md` § Door graph (`rr-mhl`).

### Item gates (`rr-iri`)
- ### ZOW — early item gates (rr-iri pathing; rr-38p residual)
- Planned hop tables in `item_gate_hops.py` (geometry only, assisted OK):
- Probe: `scripts/probe_item_gate_hops.py --route all --infinite-life`.
