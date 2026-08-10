# rr-fpd6 — Lightning Lord spawn decode (2026-08-10)

## Acceptance: MET

1. **LL object type IDs** (`$0400` = `aobject_pointer`, exist bit `$0420` bit7)
2. **Air Man per-screen placement** for prog≥1000 (mapset 4+)
3. **Live probe** shows LL present under Clean play from `AirFanPlatform` / `AirScreen2`

## Type IDs (lsmmega/mm2 disassembly + LaserEyes sprdb)

| ID | Enum / name | Role |
|----|-------------|------|
| `0x3D` (61) | `objects_kaminari_goro_move` / Lightning Lord (move) | Upper part / move state (y≈12–16) |
| `0x3E` (62) | `objects_kaminari_goro` / Thunder Chariot body | Main LL+cloud body (y≈32–36) |
| `0x3F` (63) | `objects_lightning_bolt` | Projectile |
| `0x40`/`0x41` | `objects_goblin_1/2` | Air Tikki (was mislabeled “type36” in night3–5) |
| `0x37` | `objects_pipi` | Pipi controller |

Night3–5 “types only {1,2,35,36}” was a **false negative**: probes under-reported
`$0400` (missed 0x40 goblin and never watched 0x3D/0x3E). Live re-read shows
goblin **0x40**, LL **0x3D/0x3E** while 0x23/0x24 still appear (projectiles/other).

## Air Man enemy table (checkpoint 0, mapsets 0–9)

Source: `stages/airman_wily2/airman_wily2_objects_set.asm` (lsmmega/mm2).

| idx | mapset | x | y | type |
|-----|--------|---|---|------|
| 0 | 1 | 0x48 | 0x9B | goblin_1 |
| 1 | 1 | 0xF8 | 0xAB | goblin_2 |
| 2 | 2 | 0x98 | 0x6B | goblin_1 |
| 3 | 3 | 0x48 | 0x8B | goblin_2 |
| 4 | 3 | 0xE8 | 0x8B | goblin_1 |
| **5** | **4** | **0xC0** | **0x20** | **kaminari_goro** |
| 6 | 5 | 0x60 | 0x60 | kaminari_goro |
| 7–8,10 | 6 | 0x00/0x40/0xD0 | 0x38/0x40 | kaminari_goro |
| 9 | 6 | 0x80 | 0x08 | pipi_remove |
| 11 | 7 | 0x90 | 0x30 | pipi |

First segment: mapsets **0–9** horizontal, then scroll down (`scrolling_airman_wily2_00`).
`screen_id` / camera screen matches mapset index in this stretch.

## Spawn gates (live)

From `AirFanPlatform` (prog~949, sy=84):

| Event | Frame (edge walk) | prog | cam | notes |
|-------|-------------------|------|-----|-------|
| LL enters `$0400` | ~16 | **~961** | scr3 cam_x~193 | types 0x3D+0x3E, object scr=4, x≈191–192 y≈12/32 |
| Still present through death | … | ~1047 | scr4 | never despawns before pit death |

Gates observed:

- **Scroll load**: when camera approaches mapset 4 (right side of scr3), enemy
  indices `len_idx/ren_idx` advance (4→5/6); free slots get LL.
- **No Y-band gate** for spawn: LL spawns while player is grounded at sy=84.
- **No prior-kill flag** needed: `aenemies_flag` shows active indices; first LL
  is enemy index 5 in the stage set.
- Cloud altitude **y≈32–36** (body); pure jump min_sy **~34** reaches altitude
  but closest approach was **~28px short in X** (player sx≈135 vs LL x≈163 at
  prog~1031). `tile_feet` never 1 past prog 984 (cloud is object-solid after
  kill, not a tile).

## ROM / disasm pins

- Disasm repo: https://github.com/lsmmega/mm2 (Megaman II (U))
- Object enum: `constants/objects.asm` → `objects_kaminari_goro = 0x3E`
- Placement: `stages/airman_wily2/airman_wily2_objects_set.asm`
  - enemy mapset list + xcoord/ycoord/object parallel arrays (256-byte each)
- Checkpoints / scroll: `airman_wily2_checkpoints.asm`, `airman_wily2_scrolling.asm`
- RAM: `aobject_pointer=$400`, `aobject_flag=$420` (bit7 exist),
  `aobject_screen=$440`, `aobject_x/y=$460/$4A0`, `aenemies_flag=$100`,
  `zscreen_id=$20`, `zleft/right_checkpoint_enemies_index=$48/$49`

## Live evidence

- `recordings/air_post4_fpd6/probe_*.json` — LL events >50/run; types include 61/62
- `recordings/air_post4_fpd6/summary.json`
- `recordings/air_post4_fpd6/land_grid.json` / `kill_land.json` — no cloud land yet;
  max air prog still ~1070–1071 class

## Residual for rr-54ui (not fpd6)

- **Land on Thunder Chariot**: rider kill is **Clean** (pulse B on `0x3D`,
  20→13→6→despawn). X gap closable to ~5–10px at Y-meet after kill, but
  empty cloud still freefall (`ft=0`). Object-solid / feet-on-top geometry
  residual — see `recordings/air_post4_cloud/RED_PIN.md`.
- Then chain mapset 5–6 LLs → camera≥5 → boss door.

## Do not re-run

Goblin-as-solid, pure-RIGHT grids without LL watch, “types only 35/36” camps,
hold-B without pulse.
