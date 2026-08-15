# Agent Instructions — zelda_i

NES Legend of Zelda (graph nav; **M5** Clean power-on → Level 1 Triforce).
Shared: `retro_harness.adventure`, `retro_harness.nes`.
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/HYGIENE.md`,
`docs/ASSIST_CONTRACT.md`. Tracker: **`bd ready -l zelda_i`** (prefix `rr-`).

## Commands

```bash
uv run python zelda_i/scripts/setup_rom.py
uv run python zelda_i/scripts/boot_probe.py

# Clean M5 (do not overwrite)
uv run python zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2

# Survival spine (rr-4d53; does not overwrite Clean M5)
uv run python nes/zelda_i/scripts/run_level1_complete.py \
  --natural-entry --infinite-life --video --trials 1

uv run pytest zelda_i/tests retro_harness/adventure/tests -q
bd ready -l zelda_i
```

Segment CLIs (L2–L9, TAS, lab): `docs/plan.md`.

## Layout

| Path | Role |
|------|------|
| `anchors.py` | Canonical L3–L9 door/entry/TF constants |
| `ram.py`, `overworld.py`, `overworld_nav.py` | Snapshots + OW graph / L1 path |
| `ow_path.py` | Shared `OverworldPathController` (L2–L8 hop engine) |
| `dungeon.py` + `dungeon_ids.py` | Combat engine + enemy/item IDs |
| `level*_dungeon.py` | **Room specs + stop predicates only** |
| `bomb_wall_path.py`, `level2_bomb_path.py` | Parameterized bomb-wall (`make_*`) |
| `level4_path.py` / `level4_maze_path.py` / `level4_stepladder.py` | L4 path controllers (dungeon is specs only) |
| `level*_path.py` (L5 facade + west/whistle/cellar/tf), `level*_boss_*` | Path controllers + timing knobs |
| `level*_overworld.py` | Hop tables + thin `ow_path` subclasses |
| `runner.py` | Shared script env/assist/report helpers |
| `docs/HYGIENE.md` | Architecture rules (do not re-expand phase machines) |

## Dual track

- **Clean** (default): STATUS-eligible; no health writes.
- **Assisted first pass** (`--infinite-life`): infinite hearts + damage heatmap.
  Do not promote as Clean. Contract: `docs/ASSIST_CONTRACT.md`.

**Agent order:** pathfinding + puzzles → assisted full clear → Clean harden
from damage heatmaps. Do not block tip progress on combat polish.

## Traps (burned once)

- Sword cave door is **NW** of spawn on 0x77 (not north edge 0x67). Cave = mode **11**.
- Sword pickup: x≈120 then UP; idle ~280f through dialog.
- After cave exit ~(64,77): **DOWN first**. Level 1 path:
  `0x77→E→78→N→68→N→58→N@x112→48→N→38→W→37`, door UP @x≈112.
- L1 entry room **0x73** north locked → east **0x74** first key. Keese HP=0 while
  alive (type-only liveness). Room 0x53 key fixed at **(128,109)**.
- After triforce: idle mode 18 (~704f) → OW 0x37; do not reload
  `Level1Complete` mid-fanfare. L2 prefix: `37→38→48→58→59→49→4A`; never 0x79.
- Lab checkpoints are dev fixtures until natural-entry runner passes same spec.
- `$0656` B-item: **1=bombs, 2=arrows, 4=candle**. `dungeon_ops.B_ITEM_*`.
- Old `Level5Entrance` lacks Raft/Stepladder/bombs/TF. Use `Level5EntranceFromL4`.
- L5 `0x76→0x77` is a key door: clear north `0x66` first. Fixed key can leave
  Link on the river ladder x≈56,y≈117; finish DOWN before horizontal align.
  Do not poke doors or keys for a route claim.
- Lost Hills entry from `0x1C` settles on the east ledge x≈240,y≈141; alternate
  short LEFT/DOWN bursts before the four consecutive UP wraps.
- L9 final Patra `0x52`: body `0x47` + 8 eyes `0x25`; stand 30 px south, pulse
  UP+A. After clear, recenter x≈120 before UP into Ganon `0x42`.
- L9 play `0x13` north is a **wall**. `0x03` south is a wall; east is bomb.
  Live: **0x04 bomb-west lands 0x03**. Play **0x30** block-stairs @(208,96) →
  cellar **0x67** right lands **0x04**. Play **0x21** south shutter stays sealed
  after Patra. 0x40 stays dirty (`route_eligible=false`).
- L4: skip compass to keep spare key for KEY-UP; 0x11→0x12 is **BOMB_RIGHT**
  @(192,141); Gleeok Clean = south-stand not head kite.
- L3: 0x5a key door long y=141 push; 0x69 stairs **only y≈141**; 0x5c need
  **raw=3**; type **0x2b** invuln ≠ boss; TF is **0x3d UP of boss**.

## Next

```bash
bd ready -l zelda_i
```

Tip + parked work live in `docs/plan.md` (Survival spine `rr-4d53.1` L1
Wallmaster residual; L9 `0x51` dest-NO retarget). Do not start a new room
or route leaf unless that is the claimed bead.
