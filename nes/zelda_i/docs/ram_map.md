# RAM map — Zelda I (NES)

Verified against Data Crystal and live fceumm probes (2026-07-27). Used for M2+
instrumentation and the sword-cave segment.

```text
ADDR_LEVEL          = 0x0010  # 0 = overworld; 1-9 = dungeon
ADDR_IS_UPDATING_MODE=0x0011  # 0=mode initialization, nonzero=update loop
ADDR_MODE           = 0x0012  # 5=play, 6/7=scroll, 11=cave play, 16=cave enter
ADDR_SUBMODE        = 0x0013  # mode-local phase (see Level 9 ending note)
ADDR_DIALOG_TIMER   = 0x0029  # dialog countdown
ADDR_LINK_X         = 0x0070  # 0..240 screen X
ADDR_LINK_Y         = 0x0084  # ~61..221 screen Y
ADDR_LINK_FACING    = 0x0098  # $08 N, $04 S, $01 E, $02 W
ADDR_SCREEN         = 0x00EB  # overworld: (row<<4)|col  (16x8); start=$77
ADDR_NEXT_SCREEN    = 0x00EC
ADDR_COLLIDING_TILE = 0x049E  # 0x26 empty ground (probe)

ADDR_ROOM_ITEM_ID       = 0x00AB
ADDR_CUR_OPENED_DOORS   = 0x00EE  # bit0=R bit1=L bit2=D bit3=U
ADDR_OPEN_DOORWAY_MASK  = 0x033F
ADDR_ROOM_ALL_DEAD      = 0x034D  # engine room-clear counter / flag
ADDR_ROOM_OBJ_COUNT     = 0x034E
ADDR_OBJ_TYPE           = 0x034F  # 16 object slots
ADDR_OBJ_HP             = 0x0485  # gameplay object HP slots

ADDR_SELECTED_ITEM  = 0x0656  # B slot: 1=bombs, 2=arrows, 4=candle
ADDR_SWORD          = 0x0657  # 0=none, 1=wooden, 2=white, 3=magical
ADDR_BOMBS          = 0x0658
ADDR_RUPEES         = 0x066D
ADDR_KEYS           = 0x066E
ADDR_HEALTH         = 0x066F  # hi nibble = containers-1, lo = filled hearts
                              # full refill (assist): (health & 0xF0) | 0x0F
ADDR_TRIFORCE       = 0x0671
ADDR_BOOMERANG      = 0x0674  # wooden; 0=false, 1=true
ADDR_MAGIC_BOOMERANG= 0x0675  # magical; overrides wooden when set
ADDR_MAGIC_SHIELD   = 0x0676
ADDR_MAX_BOMBS      = 0x067C
```

Survival assist (opt-in only): `zelda_i.assist.UnlimitedHealthAssist` writes
`health` via `data.set_value`; see `docs/ASSIST_CONTRACT.md`.

## Mode notes

| mode | meaning (probe) |
|------|-----------------|
| 5 | Normal overworld/dungeon play |
| 6 | Preparing scroll |
| 7 | Scrolling |
| 11 | Cave / underworld item-room play (sword cave) |
| 16 | Cave enter animation |
| 17 | Link death |
| 18 | Triforce collection / dungeon-complete animation |
| 19 (`0x13`) | Zelda ending / credits |

Level 9 ending stops must also require `ADDR_IS_UPDATING_MODE != 0` because
initialization reuses the same submode values: update submode 3 is rolling
staff credits and update submode 4 is the final Press Start page.

Screen/room hop timing (`room_timer.py`) treats mode **5** as settled play and
modes **6/7/16** (plus cave **11**) as non-destination transition noise. See
[ROOM_TIMER.md](ROOM_TIMER.md).

## Readiness

`is_level1_ready` — mode 5, level 0, health > 0, optional obs_mean > 50.

## Sword segment stop

`sword_segment_success` — sword >= 1, overworld play, screen 0x77.

## Level 1 stop

`level1_entrance_success` — `level == 1` (inside dungeon).

`level1_screen_reached` — sword >= 1, screen 0x37, still overworld.

## Level 1 room stops

`level1_first_key_success` — level 1 with `keys >= 1`.

`level1_north_room_success` — room 0x63, mode 5, and all three initial Stalfos
spawned.

`level1_room_63_cleared` — room 0x63, mode 5, zero live Stalfos, and
`RoomAllDead >= 20`. Clear yields **no inventory reward** (RoomItemId stays
`0x03`). Door probe after clear: south open to `0x73`, north open to `0x53`
(approach x≈120), west/east closed.

`level1_room_53_cleared` — room 0x53, mode 5, zero live Stalfos,
`RoomAllDead >= 20`, and keys >= 1. The `0x19` room key appears at the fixed
coordinate `(128,109)` after all five Stalfos die. Type `0x60` is a transient
green-rupee drop and must not be used as the key target.

`dungeon_room_cleared(..., ROOM_54_SPEC)` — room 0x54, mode 5, zero type
`0x1B` objects, and `RoomAllDead >= 20`. Keese HP remains zero while alive,
so this predicate intentionally uses object type only. Clear causes no known
inventory delta because the optional Compass pickup (`RoomItemId=0x16`,
walkthrough-correlated) is skipped.

`level1_complete` — `ADDR_TRIFORCE & 0x01`. The natural runner additionally
records room `0x36`, mode 18, Heart Container health `0x31`, and every
room-stage result.

```text
0x73 entrance → east 0x74 (RoomItemId=0x19, five Stalfos)
  → first key → west 0x73 → spend key north → 0x63 (three Stalfos)
  → clear 0x63 (no drop) → north door → 0x53 (five Stalfos)
  → clear + fixed key (RoomItemId=0x19)
  → west 0x52 → north 0x42 → hint 0x41 → east 0x43
  → north 0x33 → north 0x23 → backtrack to 0x43
  → east 0x44 → east 0x45 → north 0x35 → east 0x36
  → collect Triforce shard 1 (`ADDR_TRIFORCE & 0x01`)
```

Stalfos have object type `0x2A`; the controllers read slots 1–10 from the
`ObjType`, `ObjX`, `ObjY`, and `ObjHP` arrays.

Keese in rooms 0x52/0x54 have object type `0x1B`. Their `ObjHP` bytes remain
zero while alive, so future Keese stop predicates must count `ObjType` rather
than requiring positive HP.

## Overworld path notes

Verified Level 1 approach (2026-07-28): east-then-north via screens
`0x77→0x78→0x68→0x58→0x48→0x38→0x37`, enter tree door UP at x≈112 from y≈140.
Mode 8 appears after hits; treat as brief freeze.

Post-Triforce: mode 18 fanfare → idle ~704 frames → overworld **0x37** (engine
return). Walk prefix toward Level 2: `0x37→38→48→58→59→49→4A`. Avoid `0x79`
(rocky dead-end). Walkthrough Level 2 door screen: **0x3C**.
