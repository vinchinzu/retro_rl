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
| `level4_path.py` / `level4_maze_path.py` / `level4_stepladder.py` / `level4_exit60.py` / `level4_west31.py` / `level4_keyup20.py` / `level4_map21.py` / `level4_mappick.py` / `level4_bomb11.py` / `level4_key01.py` / `level4_clear12.py` / `level4_spine.py` | L4 path controllers + spine stages (dungeon is specs only) |
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
- L4: skip compass to keep spare key for KEY-UP; 0x11 north is **BOMB_UP**
  @(120,105) not free UP; 0x11→0x12 is **BOMB_RIGHT** @(192,141);
  Gleeok Clean = south-stand not head kite.
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
Continuous `l4_stepladder_continuous_v34` 1/1: east grey dock UP at x=175
y=189, y-first to y=141, LEFT onto `(136,141)`; `ADDR_LADDER` set; 118,292f.
Continuous `l4_exit60_continuous_v2` 1/1: item freeze 150f, reverse dock
DOWN at x=175/176 (v1 LEFT at `(176,173)` mid-dock solid), west-aisle UP;
0x32 play leftover `(192,189)`; 118,806f hop 514f.
Continuous `l4_west31_continuous_v1` 1/1: south-U around pushed 0x68
`(48,189)→(48,141)→(16,141)`; leftover `(208,141)` play `0x31`; hop 405f.
Continuous `l4_keyup20_continuous_v1` 1/1: reverse 0x31 east-U + LEFT+UP
clip + inland west, then KEY-UP @x120; leftover `(120,205)` play `0x20`;
keys 5→4; hop 868f. 0x20 Vire clear is on the tape (v7–v22, 1249f, ignore
`0x2b`). Continuous `l4_room21_continuous_v22` 1/1: north-around to
`(200,96)`, RIGHT+DOWN clip into x=208, y=141 then RIGHT; leftover
`(16,141)` play `0x21`; 121,775f hop 447f. Continuous
`l4_map_continuous_v15` 2/2: spawn RIGHT+UP to `(48,93)`, then
RIGHT+DOWN clips east of the vestibule; `ADDR_MAP|0x08` at `(208,181)`
in 297f (map=0x0A). Continuous `l4_bomb11_continuous_v2` 2/2: UP the
east column to y=93, LEFT to the north bomb stand `(120,105)` (v1 LEFT
at y=109 is a 16px pillar); bomb-UP → play `0x11` `(120,189)` in 435f.
Continuous `l4_key01_continuous_v3` 2/2: v1 hold-UP leftover `(120,93)`
is the north wall; bomb-UP `(120,105)` 377f then pickup `(120,141)`
819f; leftover play `0x01` `(120,133)` keys 4→5 bombs 15→14.
Continuous `l4_clear12_continuous_v1` 2/2: DOWN 0x01→0x11 244f, bomb-RIGHT
`(192,141)` 392f, Vire clear 654f (ignore `0x68`); leftover play `0x12`
`(128,117)` bombs 14→13. Isolated BFS is still banned. Next: 0x12 push
+ Gleeok approach (`PATH_12_TO_GLEEOK`). Do not close `.6` until TF `0x08`.
L2 entry bombs=0; Survival count top-up `poke_bombs=16` until farm
`rr-doua`. Isolated `Level3*` pins cannot close spine beads
(`docs/LEVEL3_ROUTE.md` § Spine attach). L9 / hygiene / isolated L4 parked.
