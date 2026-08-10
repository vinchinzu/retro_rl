# RAM map — Mega Man 2 (NES)

Verified against Data Crystal + Level1 Air Man probes (2026-08-08).
fceumm exposes system WRAM `0x0000–0x07FF`.

## Camera & position

| Addr | Name | Notes |
|------|------|-------|
| `$001B` | Camera state | `$00` idle, `$01` nametable scroll, `$02` freeze, `$80` vertical |
| `$001F` | Camera X | Fine scroll |
| `$0020` | Camera X screen | Increments when fine X wraps |
| `$0022` | Camera Y | |
| `$0460` | Mega Man X | Screen-relative; object slot 0 |
| `$04A0` | Mega Man Y | Screen-relative; fall death ≈ `Y ≥ 200` |

Progress helper: `camera_progress_x = screen * 256 + camera_x`.

## Life & combat

| Addr | Name | Notes |
|------|------|-------|
| `$06C0` | Mega Man HP | Full bar often `28` |
| `$06C1` | Boss HP | |
| `$06C2–$06E1` | Enemy HP slots | |
| `$00A8` | Lives | |
| `$00A7` | E-tanks | |
| `$004B` | Invuln timer | Counts down 1/frame |
| `$0032` | Tile under feet | `0` air, `1` ground, `2` ladder, `3` instadeath, … |
| `$003D` | Is shooting | |

## Weapons / stages

| Addr | Name | Notes |
|------|------|-------|
| `$009A` | Unlocked weapons | Bitfield (Air Shooter `$02`, …) |
| `$009C–$00A6` | Weapon / item ammo | |
| `$002A` | Stage-select cursor | Menu; may linger into play |

## Readiness

`is_level1_ready` in `ram.py`: health in `(0, 28]`, lives in `(0, 10)`, optional
`obs_mean > 50` to reject dark title frames.

## Object slots (fpd6)

fceumm WRAM; Mega Man is index 0. Parallel arrays of 32 slots.

| Addr | Name | Notes |
|------|------|-------|
| `$0400+i` | Object type (`aobject_pointer`) | Behavior/ID (see DECODE.md) |
| `$0420+i` | Flags | bit7 = exist (`objects_exist`) |
| `$0440+i` | Object screen | Matches `zscreen_id` when on-camera |
| `$0460+i` | Object X | Screen-relative |
| `$04A0+i` | Object Y | Screen-relative |
| `$0100+i` | `aenemies_flag` | Spawn/kill tracking |

Air Man LL: type **`0x3E`** (and `0x3D` move). Goblin/Air Tikki: **`0x40`/`0x41`**.
Pipi: **`0x37`**. Spawn list: stage bank objects_set (mapset + x + y + type).
