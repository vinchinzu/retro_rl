# RAM map — Super Mario Bros. (NES)

M2 instrumentation used by `smb/ram.py`, `retro_harness.platformer.levels.smb`, and
the 1-1 autobot segment.

```text
ADDR_PLAYER_STATE   = 0x000E  # 0x08 walk/stand, 0x0B dying
ADDR_PLAYER_FACING  = 0x0033  # 1=right, 2=left
ADDR_X_SPEED        = 0x0057  # signed horizontal speed
ADDR_X_PAGE         = 0x006D  # 256-pixel horizontal page
ADDR_PLAYER_X       = 0x0086  # offset within page
ADDR_Y_SPEED        = 0x009F  # signed vertical speed
ADDR_PLAYER_Y       = 0x00CE
ADDR_PLAYER_SCREEN_X= 0x03AD  # on-screen X
ADDR_AREA_POINTER   = 0x0750  # venue within multi-area levels (8-4)
ADDR_PLAYER_STATUS  = 0x0756  # 0=small, 1=big, 2=fire
ADDR_LIVES          = 0x075A
ADDR_LEVEL_LO       = 0x075C
ADDR_WORLD          = 0x075F  # 0-indexed world
ADDR_LEVEL          = 0x0760  # 0-indexed level within world
ADDR_OPER_MODE      = 0x0770  # 0=demo/title, 1=playing, 2=end, 3=game over
ADDR_SCREEN_PAGE    = 0x071A  # camera page
ADDR_SCREEN_X       = 0x071C  # camera left X within page
ADDR_TIMER_HUNDREDS = 0x07F8  # 4 at level start (400)
ADDR_TIMER_TENS     = 0x07F9
ADDR_TIMER_ONES     = 0x07FA
```

## Computed

| Name | Formula | Use |
|------|---------|-----|
| `player_x` | `x_page * 256 + x_offset` | Progress / stall |
| `level_id` | `world * 4 + level` | Completion (leave 1-1 → id 1) |
| `timer` | `h*100 + t*10 + o` | Time pressure / obs |
| `screen_x` | `screen_page * 256 + screen_x_off` | Camera / obs |
| `x_speed` / `y_speed` | signed s8 at 0x57 / 0x9F | Physics / obs |
| `in_air` | `y_speed != 0` (with state filter) | Grounded flag |

## Readiness

`is_level1_ready`: oper_mode=1, alive player_state, timer hundreds in {3,4},
lives ≤ 98, optional frame mean > 40 (rejects black title).

## Death

- `player_state == 0x0B` (dying animation)
- lives drop vs start (primary for optimizers)

## Completion (1-1)

`level_id` changes away from 0 after `max_player_x ≥ 2500` (flagpole region)
and lives did not drop — see `segment_1_1_success`.

## Completion (1-2 secret warp → World 4)

| Signal | Value |
|--------|-------|
| `world` (`0x075F`) | **3** (World 4, 0-indexed) |
| `level_id` | **12** (`3*4+0` = 4-1) |
| Helper | `reached_world_4` / `segment_1_2_warp_success` |

Underground area of 1-2 uses `level_id=2` (alias of main level, **not** clear).

## Completion (8-4 ending)

`reached_ending` requires all of:

- `world == 7`
- `level == 3`
- `oper_mode == 2`
- no lives drop when `start_lives` is supplied

The warp-finish runner additionally holds idle for 120 frames and requires the
ending predicate to remain true throughout that settle window.
