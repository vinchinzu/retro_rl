# Agent Instructions — zelda_i

NES Legend of Zelda (graph nav; **M5** Clean power-on → Level 1 Triforce).
Shared: `retro_harness.adventure`, `retro_harness.nes`. Docs: `docs/STATUS.md`,
`docs/plan.md`, `docs/HYGIENE.md`, `docs/ASSIST_CONTRACT.md`. `bd ready -l zelda_i`.

## Commands

```bash
uv run python zelda_i/scripts/setup_rom.py
uv run python zelda_i/scripts/boot_probe.py
uv run python zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2
uv run python nes/zelda_i/scripts/run_level1_complete.py \
  --natural-entry --infinite-life --video --trials 1
uv run pytest zelda_i/tests retro_harness/adventure/tests -q
bd ready -l zelda_i
```

Segment CLIs (L2–L9, TAS, lab): `docs/plan.md` · `docs/tasks/QUEUE.md`.

## Layout

| Path | Role |
|------|------|
| `anchors.py` | Canonical L3–L9 door/entry/TF constants |
| `ram.py`, `overworld*.py`, `ow_path.py` | Snapshots + OW graph / hop engine |
| `dungeon.py`, `level*_dungeon.py` | Combat + **room specs / stop predicates only** |
| `level*_path.py`, `level*_overworld.py` | Path controllers + hop tables |
| `docs/HYGIENE.md` | Architecture rules (do not re-expand phase machines) |

## Dual track

**Clean** (default, STATUS-eligible, no health writes) vs **Assisted**
(`--infinite-life`, heatmap only — do not promote). Pathfinding + puzzles →
assisted clear → Clean harden.

## Traps (burned once)

- Sword cave is **NW** of spawn on **0x77** (not north 0x67). Cave = mode **11**.
  Pickup: x≈120 then UP; idle ~280f. Exit ~(64,77): **DOWN first**. L1:
  `77→E→78→N→68→N→58→N@x112→48→N→38→W→37`, door UP @x≈112.
- L1 entry **0x73** north locked → east **0x74** first key. Keese HP=0 while
  alive. Room 0x53 key **(128,109)**. After TF: idle mode 18 (~704f) → OW 0x37;
  do not reload `Level1Complete` mid-fanfare. L2: `37→38→48→58→59→49→4A`; never 0x79.
- `$0656` B-item: **1=bombs, 2=arrows, 4=candle**. Lab checkpoints are fixtures
  until natural-entry passes the same spec.
- Old `Level5Entrance` lacks Raft/Stepladder/bombs/TF — use `Level5EntranceFromL4`.
  L5 `0x76→0x77` is a key door: clear north `0x66` first. Lost Hills from `0x1C`
  settles east ledge x≈240,y≈141; LEFT/DOWN bursts before four UP wraps.
- L9 Patra `0x52`: body `0x47` + 8 eyes `0x25`; stand 30 px south, pulse UP+A;
  recenter x≈120 before UP into Ganon `0x42`. Play `0x13` north is a **wall**.
  Live: **0x04 bomb-west lands 0x03**; `0x30` stairs @(208,96) → cellar `0x67`
  right lands **0x04**. `0x21` south shutter stays sealed after Patra. `0x40` dirty.
- L4: skip compass (spare KEY-UP); `0x11→0x12` is **BOMB_RIGHT** @(192,141);
  Gleeok Clean = south-stand not head kite.
- L3: 0x5a key door long y=141 push; 0x69 stairs **only y≈141**; 0x5c need
  **raw=3**; type **0x2b** invuln ≠ boss; TF is **0x3d UP of boss**.

Topology essays: `docs/LEVEL*_ROUTE.md`, `docs/OVERWORLD_DOORS.md`. Parked: `docs/plan.md`.
