# Agent Instructions — zelda_i

Scripted NES completion agent for **The Legend of Zelda** (graph_navigation
track; maturity **M5** — boot → Level 1 room 0x54 cleared).

## Identity

| Field | Value |
|-------|-------|
| Status | chained early suffix (M5; east branch room 0x54 clear) |
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
uv run python zelda_i/scripts/dungeon_lab.py --help
uv run pytest zelda_i/tests adventure_common/tests -q
```

## Layout

| Path | Role |
|------|------|
| `ram.py` | Snapshots, readiness, capabilities |
| `overworld.py` | 16×8 grid + early route graph (sword + Level 1 path) |
| `overworld_nav.py` | Sword → Level 1 overworld/door controller |
| `level1.py` | First key + locked north-door + room 0x63/0x53 controllers |
| `dungeon.py` | Data-driven room specs + generic dungeon combat controller |
| `dungeon_lab.py` | Parallel sweeps, traces/diffs, RAM deltas, exit probes, provenance |
| `docs/DUNGEON_LAB.md` | Lab commands, artifacts, and acceptance boundary |
| `chain.py` | Shared natural power-on → Level 1 live prefix |
| `routes.py` | Named routes / milestones |
| `sword_cave.py` | Sword segment controller |
| `adventure_common/` | Shared capability-aware `RouteGraph` (repo root) |

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
- Cleared 0x53 branches west to 0x52 (six Keese) and east to 0x54 (eight
  Keese, RoomItemId=0x16); north is closed.
- Keese (`0x1B`) have HP=0 while alive. Use type-only liveness; HP-positive
  predicates false-clear the room.
- Room 0x54 clear: attack phase 0, engage distance 48, center patrol. West
  returns to 0x53; the east doorway probe is blocked. Item `0x16` causes no
  known inventory change and remains symbolically unknown.
- Lab checkpoints are development fixtures until the same spec passes the
  power-on natural-entry runner.

## Next milestone

Take the west branch through room 0x52 and continue toward the map/Aquamentus.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
