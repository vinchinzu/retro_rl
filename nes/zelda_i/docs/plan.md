# Plan — Zelda I (NES)

## Goal

Advance from M5 (Clean power-on → Level 1 Triforce shard 1) toward a verified
continuous clear of The Legend of Zelda using the shared `retro_harness.adventure`
route graph.

Tracker: **`bd ready -l zelda_i`**. Process: `docs/tasks/PROCESS.md`.

## Parked L7/L8 boundary (2026-08-14)

`rr-dnp` now has a deterministic Survival-assisted pond controller from
`PostSwordStart`. The live walk reaches `0x53` through
`77→78→68→58→57→56→55→65→64→54→53`. The `0x64` east-ledge escape and
`0x54→0x53` transition are encoded and unit-tested. The last trial stopped on
`0x53` at `(224,173)` while trying to align DOWN for the west hop to `0x52`;
it had zero deaths and zero progression/capacity writes. It did **not** reach
the pond, so `OW_L7Pond` was not saved.

Exact continuation command:

```bash
PYTHONPATH=nes uv run python nes/zelda_i/scripts/probe_level7_entry.py \
  --allow-missing-caps --infinite-life --save-state --max-frames 10000 \
  --tag l7_dnp_pond_assisted_v10
```

Before rerunning, add one `level7_overworld.py` micro for `0x53`: move LEFT
inland from the east edge before descending toward the lower west gap, then
push LEFT to `0x52`. Evidence to compare:
`recordings/l7_dnp_pond_assisted_v9.json` and its `_final.png`.

Level 8 is parked at the existing `0x6D` bush/candle boundary. Do not reopen
the old poke-burn result as a route claim. After L7 yields the natural Red
Candle, the smallest L8 boundary is Red Candle + `Level8BushOW` → burn `0x6D`
→ live entry room; otherwise the residual is natural 60R farm → Blue Candle
buy → burn. No new L8 checkpoint or claim was made in this pass.

## Next pass — predecessor of blade-trap/Like-Like room 0x41 (2026-08-14)

Verified backward recon remains an explicit fixture, not Clean or Survival
route STATUS:

- Blade-trap/Like-Like room `0x41` settles with four traps and four
  Like-Likes. The north mask is visible, but live enemies block the walk.
  Controller-only clear followed by north lands east-bomb `0x31`; no door or
  next-room poke is used.
- Continuous fixture compose `0x41→0x31→0x30→0x67→0x04→0x03→0x52→credits`
  is **1/1**: credits 25,858, final page 27,058, total 27,148 frames. Runtime
  object/room/door/inventory/progression/capacity writes are all zero.
- Evidence: `recordings/l9_room41_dump.json`,
  `recordings/l9_play41_north_patra_credits_recon.json`, and
  `Level9Room41NorthReconFixture`. The start still inherits fixture inventory
  and loader setup, so `route_eligible=false`.
- Next live question: materialize the real predecessor south of `0x41`
  (candidate `0x51`) without staging `0x41` doors, then determine how its
  north shutter is earned. Keep `0x40` out of this predecessor chain.

Exact replay commands from the repository root:

```bash
uv run python nes/zelda_i/scripts/run_level9_stairs.py \
  --dump-41 --tag l9_room41_dump
uv run python nes/zelda_i/scripts/run_level9_stairs.py \
  --compose-41 --infinite-life --save-state --trials 1 \
  --tag l9_play41_north_patra_credits_recon
```

The forward route now has a continuous Whistle basement `0x04` → Digdogger
`0x24` → L5 Triforce `0x14` reel. Its missing predecessor seam is East Key
Pols Voice `0x77` → natural Recorder acquisition → Whistle basement.

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

1. **L9 backward tip** — real predecessor south of blade-trap/Like-Like room
   `0x41`; its clear+north → east-bomb `0x31` suffix is accepted.
2. **Forward route (preserved)** — East Key Pols Voice `0x77` → natural
   Recorder acquisition → Whistle basement `0x04`; the Digdogger/TF suffix is
   accepted. Then continue Levels 6–8 under Survival assist.
3. **M6 route graph** — compose the assisted checkpoint chain into reusable
   route legs, then run a continuous dry run.
4. **M7–M8** — verified full-game capture (assisted first, Clean later).

## Bottleneck

**Real entry into blade-trap/Like-Like room `0x41`** is the active backward
boundary. Its clear+north → east-bomb `0x31` → block-stairs `0x30` → cellar
`0x67` → Patra stairs `0x03` → credits suffix is accepted but fixture-started.
The forward boundary is East Key Pols Voice `0x77` → Whistle basement `0x04`.

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
