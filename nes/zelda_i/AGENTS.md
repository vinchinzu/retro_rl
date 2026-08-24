# Agent Instructions — zelda_i

NES Legend of Zelda (graph nav; **M5** Clean power-on → Level 1 Triforce).
Shared: `retro_harness.adventure`, `retro_harness.nes`.
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/HYGIENE.md`,
`docs/ASSIST_CONTRACT.md`, `docs/tasks/PROCESS.md`.
Tracker: **`bd ready -l zelda_i`** (prefix `rr-`).

## Commands

```bash
uv run python zelda_i/scripts/setup_rom.py
uv run python zelda_i/scripts/boot_probe.py

# Clean M5 (do not overwrite)
uv run python zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2

# Survival spine — one continuous session (does not overwrite Clean M5)
# First file slot / first quest. Records MP4 unless --no-video.
uv run python nes/zelda_i/scripts/run_survival_spine.py --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level2 --trials 1

uv run pytest zelda_i/tests retro_harness/adventure/tests -q
bd ready -l zelda_i
```

Segment CLIs (L2–L9, TAS, lab): `docs/plan.md` and `docs/tasks/QUEUE.md`.

## Layout

| Path | Role |
|------|------|
| `anchors.py` | Canonical L3–L9 door/entry/TF constants |
| `ram.py`, `overworld.py`, `overworld_nav.py` | Snapshots + OW graph / L1 path |
| `ow_path.py` | Shared `OverworldPathController` (L2–L8 hop engine) |
| `walk_physics.py`, `predict.py` | OccupancyWalker + RAM claims (`retro_harness.predict`) |
| `level3_spine.py` | Exact predicate stops for the continuous L3 spine |
| `dungeon.py` + `dungeon_ids.py` | Combat engine + enemy/item IDs |
| `level*_dungeon.py` | **Room specs + stop predicates only** |
| `bomb_wall_path.py`, `level2_bomb_path.py` | Parameterized bomb-wall (`make_*`) |
| `level4_path.py` / `level4_maze_path.py` / `level4_stepladder.py` | L4 path controllers (dungeon is specs only) |
| `level*_path.py` (L5 facade + west/whistle/cellar/tf), `level*_boss_*` | Path controllers + timing knobs |
| `level*_overworld.py` | Hop tables + thin `ow_path` subclasses |
| `runner.py` | Shared script env/assist/report helpers |
| `docs/HYGIENE.md` | Architecture rules (do not re-expand phase machines) |

## Dual track

- **Assisted first pass** (segment-script default, `--infinite-life`):
  infinite hearts + damage heatmap. Opt out with `--no-infinite-life`.
  Do not promote as Clean. Contract: `docs/ASSIST_CONTRACT.md`.
- **Clean**: STATUS-eligible; no health writes. M5 stays
  `run_level1_complete` without `--infinite-life`.

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
  `$066F` low nibble is **whole hearts**, not `0xF` full. Writing `0xF`
  makes `World_FillHearts` `INC` extra containers. Full is `lo==hi`
  (`0x22`=3/3) plus `$0670=$FF`. After L1 TF expect **5**, after L2 TF **7**.
- Stuck nav: stand still (`*_wait`). Do not loop LEFT/RIGHT/DOWN wiggle.
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

Tip + parked work live in `docs/plan.md`. Spine is continuous only
(`run_survival_spine.py`); no seamed compose. The live power-on spine now
clears L4 `0x50`: `l4_clear50_continuous_v1`, TF=0x07, keys=5, bombs=15,
zero deaths/progression/capacity writes and zero state restores. Survival bomb
count top-ups are operator-approved through the assisted full-game clear;
record every verified gate and never write capacity. L4 entry bombs=0, so the
0x61 wall gets 0→16 and consumes one. Continuous v7 reaches `0x40` through
coordinate gates, clears the Zols, and naturally raises keys 5→6. Continuous
`l4_room31_continuous_v1` clears `0x30` Vires from `(120,205)` (ignore
invuln `0x2b`) then KEY-RIGHT @y141 into `0x31` at `(16,141)` in 667f.
Continuous `l4_clear31_continuous_v7` RIGHT+UP-clips the west alcove,
waypoints the maze, and clears 5× Vire in 4,818f; leftover `(112,141)`.
Continuous `l4_room32_continuous_v11` UP-to-y113, RIGHT+DOWN-clips the east
column, then south-U waypoints into `0x32` at `(16,141)` in 376f.
Continuous `l4_clear32_continuous_v1` clears 2× Zol + 2× LikeLike from
leftover `(16,141)` in 3,812f (ignore `0x2b`/`0x68`); leftover `(80,109)`.
`--through level4-stepladder` is wired (`clear_first=False`) but live-blocked
v1–v11: push enters `0x60` mode-9 leftover `(48,133)`; west-aisle RIGHT
solid, south-corridor UP solid, RIGHT+UP/DOWN clips miss, token
`MAZE_60_TO_LADDER` hits the `0x32` exit. Isolated BFS from `(48,69)` is
banned. Next is a coordinate causeway onto the island / `ADDR_LADDER`;
do not use checkpoint-mediated/emulator-state BFS.
L2 entry bombs=0; Survival count top-up `poke_bombs=16` until farm
`rr-doua`. Isolated `Level3*` pins cannot close spine beads
(`docs/LEVEL3_ROUTE.md` § Spine attach). L9 / hygiene / isolated L4 parked.
