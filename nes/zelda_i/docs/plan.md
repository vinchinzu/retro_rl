# Plan — Zelda I (NES)

## Goal

Advance from M5 (Clean power-on → Level 1 Triforce shard 1) toward a verified
continuous clear of The Legend of Zelda using the shared `retro_harness.adventure`
route graph.

Tracker: **`bd ready -l zelda_i`**. Process: `docs/tasks/PROCESS.md`.

## Next pass — move backward from final Patra to candidate room 0x62 (2026-08-14)

Verified recon endpoint (explicit fixture; not Clean or Survival route STATUS):

- `Level9FinalPatraReconFixture` starts with live body type `0x47` HP `0xB0`
  plus eight orbiting eyes type `0x25` HP `0x60`, and both north-door fields
  closed. The south-stand controller naturally removes all eyes and the body,
  then the engine raises `CurOpenedDoors & 0x08`.
- `0x52` → Ganon `0x42` → Zelda `0x32` → rolling credits → final page is
  **2/2 exact** in one emulator session: Patra clears at controller frame
  1,883, credits begin at total frame 5,342, final page at 6,542.
- The Patra segment preserves the fixture inventory. Runtime controller writes
  are object/room/door/inventory/progression/capacity = 0; Survival restores
  four filled-heart units (two in Patra, two in Ganon), with zero deaths.
- Runner/evidence: `scripts/run_level9_patra.py`,
  `recordings/l9_patra_credits_recon.json`; checkpoints and screenshot names
  are listed in `LEVEL9_ROUTE.md`.

Next, inspect the **candidate southern predecessor room `0x62`** under
`rr-sz8.3`. Use the game room loader to materialize its untouched contents,
record a room-entry screenshot/RAM snapshot, solve its live door contract with
controller input, then naturally enter `0x52` and reuse the proven Patra suffix.
Treat `0x62` as a geometry hypothesis until the live transition confirms it;
do not clear objects or write room/door state in the accepted run.

Exact handoff command from the repository root:

```bash
bd show rr-sz8.3
uv run python nes/zelda_i/scripts/run_level9_patra.py \
  --from-state Level9FinalPatraReconFixture --infinite-life \
  --save-state --trials 2 --tag l9_patra_credits_recon
```

The second command replays the accepted suffix while the `0x62` fixture/probe
is added. The forward L5 boundary remains preserved at `Level5Cleared66`.

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

1. **L9 backward tip** — inspect/solve candidate predecessor `0x62`, naturally
   enter final Patra `0x52`, then reuse the 2/2 Patra→credits suffix.
2. **Forward route (preserved)** — L5 east key → Whistle → Digdogger →
   `triforce & 0x10`, then L6–L8 under Survival assist.
3. **M6 route graph** — compose the assisted checkpoint chain into reusable
   route legs, then run a continuous dry run.
4. **M7–M8** — verified full-game capture (assisted first, Clean later).

## Bottleneck

**Candidate predecessor room `0x62`** is the active backward boundary. Final
Patra `0x52` through the final ending page is verified 2/2 with controller
input after the disclosed start fixture. The next pass must live-verify 0x62's
contents and earn its transition into 0x52. The L5 east key is the paused
forward boundary, not erased or superseded as route evidence.

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
