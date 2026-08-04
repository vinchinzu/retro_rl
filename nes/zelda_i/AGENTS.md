# Agent Instructions — zelda_i

Scripted NES completion agent for **The Legend of Zelda** (graph_navigation
track; maturity **M5** — Clean power-on → Level 1 Triforce shard 1).

## Identity

| Field | Value |
|-------|-------|
| Status | chained Level 1 completion (M5; `triforce & 0x01`) |
| Integration | `LegendOfZelda-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Legend of Zelda, The.zip` |
| Local ROM | `zelda_i/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python zelda_i/scripts/setup_rom.py
uv run python zelda_i/scripts/boot_probe.py
uv run python zelda_i/scripts/run_sword_cave.py              # isolated Level1
uv run python zelda_i/scripts/run_sword_cave.py --natural-entry
uv run python zelda_i/scripts/run_to_level1.py               # isolated → dungeon
uv run python zelda_i/scripts/run_to_level1.py --natural-entry
uv run python zelda_i/scripts/run_to_level1.py --screen-only # stop on 0x37
uv run python zelda_i/scripts/run_level1_first_key.py
uv run python zelda_i/scripts/run_level1_first_key.py --natural-entry
uv run python zelda_i/scripts/run_level1_north.py
uv run python zelda_i/scripts/run_level1_north.py --natural-entry
uv run python zelda_i/scripts/run_level1_clear63.py
uv run python zelda_i/scripts/run_level1_clear63.py --natural-entry
uv run python zelda_i/scripts/run_level1_clear53.py
uv run python zelda_i/scripts/run_level1_clear53.py --natural-entry
uv run python zelda_i/scripts/run_level1_clear54.py
uv run python zelda_i/scripts/run_level1_clear54.py --natural-entry
uv run python zelda_i/scripts/run_level1_complete.py --trials 2
uv run python zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2
uv run python zelda_i/scripts/run_to_level2_prefix.py --trials 2
uv run python zelda_i/scripts/run_to_level2_prefix.py --from-heart --trials 2
uv run python zelda_i/scripts/run_to_level2_prefix.py --room-timing --trials 1
uv run python zelda_i/scripts/run_level1_complete.py --room-timing --trials 1
uv run python zelda_i/scripts/dungeon_lab.py --help
uv run python zelda_i/scripts/probe_room_timer.py self-check
uv run python zelda_i/scripts/probe_room_timer.py offline -i samples.json
uv run pytest zelda_i/tests retro_harness/adventure/tests -q
```

## Layout

| Path | Role |
|------|------|
| `ram.py` | Snapshots, readiness, capabilities |
| `overworld.py` | 16×8 grid, `ScreenHop` path tables, early route graph |
| `overworld_nav.py` | Sword → Level 1 overworld/door controller |
| `level1.py` | First key + locked north-door + room 0x63/0x53 controllers |
| `dungeon.py` | Data-driven room specs + generic dungeon combat controller |
| `level1_finish.py` | Switch/hint routing, backtrack, Aquamentus, and Triforce controllers |
| `level2_overworld.py` | Post-Triforce settle + walk prefix toward Level 2 (stop 0x4A) |
| `nav_common.py` | Shared overworld swing/stuck/align helpers |
| `dungeon_lab.py` | Parallel sweeps, traces/diffs, RAM deltas, exit probes, provenance |
| `docs/DUNGEON_LAB.md` | Lab commands, artifacts, and acceptance boundary |
| `docs/LEVEL1_ROUTE.md` | Walkthrough correlation + verified Eagle speed route |
| `docs/LEVEL2_ROUTE.md` | Post-Triforce settle + walk prefix / traps toward Moon |
| `chain.py` | Shared natural power-on → Level 1 live prefix |
| `routes.py` | Named routes / milestones |
| `sword_cave.py` | Sword segment controller |
| `room_timer.py` | Confirmed OW-screen / dungeon-room hop timing (emulator frames) |
| `retro_harness/adventure/` | Shared capability-aware `RouteGraph` |

## Traps

- Sword cave door is **NW** of spawn on screen 0x77, not north edge (north edge is 0x67).
- Cave play is **mode 11**, not 5. Do not mash A during mode 16 enter animation.
- Sword pickup needs **x≈120** alignment then UP; x=112 leaves you short of the item.
- Hold only idle through dialog (~280 frames), then align and collect.
- After sword exit at ~(64,77), **DOWN first** out of the cave pocket before routing.
- **Do not go straight north on col 7** for Level 1: 0x67 is a dead-end grove; 0x47 is a lake.
- Level 1 path: `0x77→E→0x78→N→0x68→N→0x58→N@x112→0x48→N→0x38→W→0x37`, door UP at x≈112.
- On 0x58 bush grid, horizontal travel needs **y≈150–160**.
- Level 1 entry is room **0x73**; its north door is locked. Take the open east
  door to **0x74** for the first key.
- Room 0x74's block clusters require lane routing: use y≈181 to return west,
  then x≈48/y≈141 into room 0x73.
- From the 0x73 east doorway, step left to x≈208, descend to y≈149, move to
  x≈120, then go north to unlock the door into **0x63**.
- Room 0x63 clear: hybrid nearest-Stalfos engage + box patrol; wait for
  `RoomAllDead>=20`. No key drop. North door to **0x53** opens at x≈120.
- From the 0x63 clear endpoint, route `(64,101)→(120,101)→N` into 0x53.
  Preserve the one-frame waypoint idles: starting its patrol two frames late
  produces a deterministic death.
- Room 0x53's key is the fixed room-clear item at **(128,109)**. Do not chase
  transient type `0x60` green-rupee drops as though they were the key.
- Cleared 0x53 branches west to the required 0x52 route and east to optional
  0x54 (eight Keese / Compass); north is closed.
- Keese (`0x1B`) have HP=0 while alive. Use type-only liveness; HP-positive
  predicates false-clear the room.
- Room 0x42's switch block is pushed north from x≈112/y≈149. The accepted
  route visits hint room 0x41 before crossing east into 0x43.
- Room 0x44's maze requires the upper corridor to reach 0x45. The speed route
  skips its Boomerang pickup.
- Dormant Wallmasters remain at x=0 after the first wave. Use dominant-axis
  engagement so Link aligns vertically and slashes left into the wall.
- Aquamentus is north of 0x45 in 0x35. Use the fixed stance/fireball dodge
  controller; the natural RNG stream needs its recorded 109-frame entry
  alignment.
- The Triforce is east in 0x36. Route down the left perimeter, east along the
  bottom, then north through the x≈112–128 opening.
- After triforce: **idle** mode 18 (~704f) → overworld **0x37**. Do not reload
  `Level1Complete` mid-fanfare (can freeze). Use `Level1ExitOverworld` or live
  settle.
- Level 2 walk prefix: `0x37→38→48→58→59→49→4A` (controller stop). **Never**
  route through rocky dead-end **0x79**. On 0x37 only y≈140 exits east.
- Item IDs `0x16` Compass, `0x17` Map, and `0x1D` Boomerang are
  walkthrough-correlated; the speed route does not collect them.
- Lab checkpoints are development fixtures until the same spec passes the
  power-on natural-entry runner.

## Next milestone

Heart-safe door path 0x37→0x3C (maze on 0x5C; N@x52 on 0x5D), enter Moon room
0x7d, clear Level 2 (Ropes `0x28`, Dodongo bombs) to `triforce & 0x02`.

Door-path traps: never `0x4B→0x5B` north-entry (east sealed); use `0x5A→0x5B`;
0x5C needs `LEVEL2_5C_MAZE_WAYPOINTS`; wait ~100f for Rope spawn after room
enter. Walkthroughs: `docs/research/DUNGEON_WALKTHROUGHS.md`.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
