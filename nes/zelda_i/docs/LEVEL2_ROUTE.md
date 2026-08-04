# Level 2 route — The Moon

Planning sources:

- [Zelda Dungeon — Level 2: The Moon](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-2-the-moon/)
- [IGN — Dungeon Two](https://www.ign.com/wikis/the-legend-of-zelda/Dungeon_Two)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)

Walkthrough claims below that are emulator-verified are marked; source-only
claims stay labeled.

## Post-Triforce return (verified)

Collecting shard 1 sets `ADDR_TRIFORCE & 0x01` and enters **mode 18** (fanfare).
After ~535 idle frames the engine transitions (modes 2→3→4) and places Link on
**overworld screen 0x37** at ~(112, 125) around frame **704**.

- Prefer `Level1ExitOverworld.state` or a live settle after collection.
- Reloading `Level1Complete.state` mid-fanfare can freeze mode 18.

## Verified walk prefix (0x37 → 0x4A)

```text
0x37 ─E@y≈140─► 0x38 ─S@x≈120─► 0x48 ─S@x≈112─► 0x58
  ─E@y148–162─► 0x59 ─N@x≈112─► 0x49 ─E@y≈141─► 0x4A
```

Stop: `level2_path_prefix_success` — overworld play, screen 0x4A, sword ≥ 1,
triforce & 0x01. Evidence: `recordings/level2_prefix_isolated.json` (3/3).

Per-screen hop timing (emulator frames, `RoomTimer`, 2026-07-29 1/1):
`recordings/room_timings/level2_prefix_isolated_timing.json` — six hops,
slowest `0x49→0x4A` (~539 location frames); transitions ~83–104f.

## Full door path (probe-verified geometry; health not Clean yet)

Walkthrough target door: overworld **0x3C**. Naive “right four from start”
hits rocky dead-end **0x79**. North-entry `0x4B→0x5B` **seals east** (BFS max
x≈144). Correct corridor enters **0x5B from the west via 0x5A**.

```text
0x37 E@y140 → 0x38 S → 0x48 S → 0x58 E → 0x59
  E@y120–145 → 0x5A E@y130–150 → 0x5B
  E@y80–95 (north bush corridor) → 0x5C
  [maze] E@y≈88 to x≈184, down to y≈128, E → 0x5D
  N@x≈48–56 → 0x4D W@y120–170 → 0x4C N@x112 → 0x3C (Moon door UP)
```

Hop tables: `overworld.LEVEL2_DOOR_HOPS`, maze pixels
`LEVEL2_5C_MAZE_WAYPOINTS`. Fixture states (dev): `Level2DoorOW.state`,
`Level2Entrance.state`, `Level2EntryFresh.state`.

### 0x5C maze (required)

BFS path cells (cell size 8): east along row y≈88 to gx≈23, up/down channel at
gx≈24, then east at y≈128 into 0x5D. Plain `ScreenHop` RIGHT with a single
y-band is **not** enough.

### 0x5D north

North exit only near **x≈48–56** (not center). East also opens to 0x5E but is
not on the door route.

## Interior (walkthrough + first live probes)

| Claim | Source | Live |
|-------|--------|------|
| OW door 0x3C | walkthrough + probe | verified entry |
| Entry room | — | **0x7d** (south mouth; mode 16 settle → play) |
| North of entry | walkthrough UP | **0x6d**, Ropes type **0x28** (spawn ~100f after enter) |
| Clear opens left door bit | walkthrough | `cur_opened_doors` bit1 often sets; physical LEFT still flaky |
| Magical Boomerang | walkthrough | not yet |
| Dodongo needs bombs | walkthrough | not yet |
| Triforce bit 0x02 | walkthrough | not yet |

Speed-route sketch (source): N ropes → W key → return → E key → N/E with keys
→ optional Compass/Map/bomb shortcuts → Magical Boomerang → Moldorm key →
Ropes unlock → Goriya bombs → **Dodongo** (2 mouths) → Heart → Triforce shard 2.

## Traps

| Trap | Detail |
|------|--------|
| 0x79 rocky dead-end | No east exit from 0x78 east@y≈180. |
| 0x37 east lane | Only **y≈140** exits east; y≈125 re-enters Level 1. |
| 0x4B→0x5B north entry | East of 0x5B unreachable; use **0x5A→0x5B**. |
| 0x5C north pocket | Entering only at y≈93 without maze cannot reach 0x5D. |
| 0x5A damage corridor | Arrives low on hearts; Clean farm/heal still open. |
| 0x5D north x | Must align **x≈52**, not x112. |
| Room-ready | After dungeon room transition wait for enemy types (Ropes ~100f). |

## Controllers / runner

```bash
uv run python zelda_i/scripts/run_to_level2_prefix.py --trials 2
uv run python zelda_i/scripts/run_to_level2_prefix.py --room-timing --trials 1
uv run python zelda_i/scripts/probe_level2_suffix.py --help
```

- `level2_overworld.PostTriforceSettleController`
- `level2_overworld.OverworldToLevel2Controller` (default stop 0x4A)
- Door path + maze not yet promoted to a 2/2 Clean natural runner
- Opt-in hop timing: `chain.run_controller_stage(..., room_timer=)` /
  runner `--room-timing` → `recordings/room_timings/`

## Measured door-path fail (not route progress)

From `Level1ExitOverworld` with `LEVEL2_DOOR_HOPS` + `require_level2_screen`
(2/2, Clean): reaches **0x5C** then **dies** (mode 17) at ~(16,93) with
**0 filled hearts**. Health drain along the way: 3→2 on 0x48/58, 2→1 on
0x59/5A, 1→0 on 0x5B/5C. Slowest hops: `0x5B→0x5C` (~718f), `0x59→0x5A`
(~659f). Maze hop 0x5C→0x5D never starts. Timing artifact:
`recordings/room_timings/level2_door_path_probe_timing.json`.

**Next experiment:** farm hearts on 0x4A to ≥2 filled before entering the
0x5A corridor; only then wire `LEVEL2_5C_MAZE_WAYPOINTS` into the controller.

## Acceptance (not yet)

- [ ] 2/2 isolated walk 0x37→0x3C without health poke
- [ ] 2/2 enter Level 2 (`level==2`, room-ready 0x7d)
- [ ] 2/2 clear Moon → `triforce & 0x02` isolated + natural-entry
