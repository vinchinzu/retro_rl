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
| `$009A` | Unlocked weapons | `$01` Atomic Fire (Heat), `$02` Air Shooter, `$04` Leaf, `$08` Bubble, `$10` Quick, `$20` Time, `$40` Metal, `$80` Crash |
| `$009B` | Unlocked items | `$01` Item-1 (Heat), `$02` Item-2 (Air), `$04` Item-3 (Flash) |
| `$009C–$00A6` | Weapon / item ammo | |
| `$002A` | Stage-select cursor | `$00` Wily; `$01`–`$08` clockwise from Bubble (Heat=`$08`, Air=`$02`) |

Stage-select grid:

```
1 Bubble   2 Air     3 Quick
8 Heat     0 Wily    4 Wood
7 Metal    6 Flash   5 Crash
```

## Readiness

`is_level1_ready` in `ram.py`: health in `(0, 28]`, lives in `(0, 10)`, optional
`obs_mean > 50` to reject dark title frames.

## Object slots (fpd6)

fceumm WRAM; Mega Man is index 0. Parallel arrays of 32 slots.

| Addr | Name | Notes |
|------|------|-------|
| `$0400+i` | Object type (`aobject_pointer`) | Behavior/ID (see DECODE.md) |
| `$0420+i` | Flags | bit7 exist, bit6 right, bit5 invis, bit4 appearing_block |
| `$0440+i` | Object screen | Matches `zscreen_id` when on-camera |
| `$0460+i` | Object X | Screen-relative |
| `$04A0+i` | Object Y | Screen-relative |
| `$04E0+i` | `aobject_tsa` | Appearing-block solid type **or** enemy AI timer (LL: countdown) |
| `$0600/$0640+i` | Object X/Y speed | Signed motion |
| `$0100+i` | `aenemies_flag` | Spawn/kill tracking |

Flag constants (lsmmega/mm2 `constants/flags.asm`): `objects_exist=$80`,
`objects_right=$40`, `objects_invisible=$20`, `objects_appearing_block=$10`.

Air Man LL: type **`0x3E`** body + **`0x3D`** move/rider. On rider death: type **`6`**
(`objects_killed`) ~12f. Empty body stays `0x3E`; `aobject_tsa` cycles as AI timer
(not solid). Body AI (lsmmega bank14) has no solid-arm rewrite; appearing_block
`$10` is the only decoded object-solid path and is never set on empty chariot.
Cloud OAM top ≈ body y−16. Goblin/Air Tikki: **`0x40`/`0x41`**. Pipi: **`0x37`**.

| Addr | Name | Notes |
|------|------|-------|
| `$002C` | `zmegaman_status` | Pose SM: 3/6/7 air/fall variants; no cloud-stand lock observed |
| `$0110+i` | Body AI child index | LL body tracks child slot; not a solid enable |
| `$0120+i` | Parent link | Rider stores body slot |
