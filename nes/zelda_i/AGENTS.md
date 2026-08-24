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
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level5 --no-video --trials 1

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
| `level4_path.py` / `level4_maze_path.py` / `level4_stepladder.py` / `level4_exit60.py` / `level4_west31.py` / `level4_keyup20.py` / `level4_map21.py` / `level4_mappick.py` / `level4_bomb11.py` / `level4_key01.py` / `level4_clear12.py` / `level4_gleeok13.py` / `level4_spine.py` | L4 path controllers + spine stages (dungeon is specs only) |
| `level5_spine.py` | L4 TF settle → L5 entry → 0x66 → east 0x77 → Recorder 0x04 → TF `0x10` |
| `level6_spine.py` | L5 TF settle → L6 entry → east key 0x7a → west 0x78 → 0x18 Gleeok → 0x19 |
| `level*_path.py` (L5 facade + west/whistle/cellar/tf; L6 north 0x68 / 0x18 settle), `level6_gleeok18.py`, `level6_stairs18.py`, `level6_room19.py`, `level*_boss_*` | Path controllers + timing knobs |
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
- L5 Recorder cellar `0x04` leftover `(135,141)`: ladder x=176 DOWN, pit y=189,
  mouth x=48 UP. 0x06 return is `take_block_stairs_06` RIGHT onto `(128,141)`
  after 0x68 push; center idle at `(120,141)` never warps. 0x65 north shutter
  is one-way; bomb-east to 0x66.
- Lost Hills entry from `0x1C` settles on the east ledge x≈240,y≈141; alternate
  short LEFT/DOWN bursts before the four consecutive UP wraps.
- L6 0x58 north leftover PNG shutter is **walkable** (clear58 v1 wrong belief;
  occupancy long-UP is free, keys unchanged). 0x48 blade traps: run UP, no clear.
  0x38 Bubble `0x40` is sword-immune residual (not a clear blocker); ignore
  invuln `0x2b`. Left 0x68 at `(96,144)` must actually move UP (v2 shutter
  looks open, is sealed; v3 200f UP is not a push; v5 center-UP @ x=120
  hits the pair). After y-move, west aisle x=64 then north door.
  0x28 leftover `(120,181)` UP is solid; LEFT+UP clips y=181, hold-UP to
  `(96,109)`, RIGHT+UP at y=109 enters play `0x18`. Cardinal RIGHT at
  y=173 and y=109 is solid. 0x18 enter leftover has no Gleeok; idle
  census is type **`0x44`** (not L4 `0x43`) at `(124,111)` HP160 +
  fireball `0x56`. y=189 UP is solid; LEFT+UP slides west then inland.
  South-stand kills `0x44`; east shutter still closed after body-gone.
  Post-Gleeok: no `0x46`. PNG-black east + `open_doorway_mask` 0 is **not**
  sealed: occupancy y=141 RIGHT enters play `0x19` (v1 1/1). Do not RIGHT
  at leftover y=133. North hole is decorative: y=109 `0x76`, y=101/95 `0x77`,
  hold-UP `(120,93)` not mode 9 (`--through level6-stairs18` stays red).
  0x19 live is 2× Zol `0x13` + 2× Like-Like `0x17` (enter PNG beam is the
  Like-Like, not a wizzrobe). Map sprite at south-center `(120,181)` idle
  is **not** `ADDR_MAP|0x20`.
- Post-L5 0x1B west exit is **y=141 LEFT** after south-around the x≈72 rock
  (v25 north-edge LEFT solid; v31 leftover `(24,149)` is mountain dither not
  a free walk; v32/v33 diagonal clips yo-yo). 0x14/0x23 south mouths are the
  SE blue paths (align_x 160 / 208), not x=112. 0x15 Lynels: inland then
  south band y=165–189. Do not grant Whistle.
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
(`run_survival_spine.py`); no seamed compose. The live power-on spine holds
L6 cleared `0x19` (`l6_clear19_continuous_v1` 1/1, 208,845f hop 4,213f
leftover `(176,158)`, keys=5, bombs=8, TF=`0x1F`, map=`0x0A`,
deaths/progression/capacity 0, no state load). Census 2× Zol `0x13` +
2× Like-Like `0x17`; RoomItemId `0x17` Map on floor not collected
(map19 v2 stood on `(120,181)` sprite, bit still off). North hole
decorative. Rod / Gohma / TF `0x20` residual. Do not grant Map/Rod.
Do not poke doors/keys. Ignore 0x2b.
L2 entry bombs=0; Survival count top-up `poke_bombs=16` until farm
`rr-doua`. Isolated `Level3*` pins cannot close spine beads
(`docs/LEVEL3_ROUTE.md` § Spine attach). L9 / hygiene / isolated L4 parked.
