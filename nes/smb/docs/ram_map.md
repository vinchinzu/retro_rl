# RAM map — Super Mario Bros. (NES)

M2 instrumentation used by `smb/ram.py`, `retro_harness.platformer.levels.smb`, and
the 1-1 autobot segment.

```text
ADDR_PLAYER_STATE   = 0x000E  # 0x08 walk/stand, 0x0B dying
ADDR_PLAYER_MOTION  = 0x001D  # 0=grounded, 1=air (smbdis Player_State)
ADDR_PLAYER_FACING  = 0x0033  # 1=right, 2=left
ADDR_X_SPEED        = 0x0057  # signed horizontal speed (high byte)
ADDR_X_PAGE         = 0x006D  # 256-pixel horizontal page
ADDR_PLAYER_X       = 0x0086  # offset within page
ADDR_Y_SPEED        = 0x009F  # signed vertical speed
ADDR_PLAYER_Y       = 0x00CE
ADDR_PLAYER_SCREEN_X= 0x03AD  # on-screen X
ADDR_PLAYER_X_FRAC  = 0x0400  # SprObject_X_MoveForce; X position frac (Oσ)
ADDR_PLAYER_Y_FRAC  = 0x0416  # Player_YMF_Dummy; Y position frac (Oσ)
ADDR_Y_MOVE_FORCE   = 0x0433  # Player_Y_MoveForce; gravity accumulator
ADDR_JUMP_ORIGIN_Y  = 0x0708  # Y pixel at takeoff
ADDR_VERTICAL_FORCE = 0x0709  # rising / current gravity
ADDR_VERTICAL_FORCE_DOWN = 0x070A  # fall gravity (A-release copy)
ADDR_AREA_POINTER   = 0x0750  # venue within multi-area levels (8-4)
ADDR_PLAYER_STATUS  = 0x0756  # 0=small, 1=big, 2=fire
ADDR_LIVES          = 0x075A
ADDR_LEVEL_LO       = 0x075C
ADDR_WORLD          = 0x075F  # 0-indexed world
ADDR_LEVEL          = 0x0760  # 0-indexed level within world
ADDR_OPER_MODE      = 0x0770  # 0=demo/title, 1=playing, 2=end, 3=game over
ADDR_RUNNING_SPEED  = 0x0703  # RunningSpeed; latched |vx| when >= $1C
ADDR_X_FORCE        = 0x0705  # Player_X_MoveForce; 16-bit X-speed low byte
ADDR_SCREEN_PAGE    = 0x071A  # camera page
ADDR_SCREEN_X       = 0x071C  # camera left X within page
ADDR_TIMER_HUNDREDS = 0x07F8  # 4 at level start (400)
ADDR_TIMER_TENS     = 0x07F9
ADDR_TIMER_ONES     = 0x07FA
ADDR_FRAME_COUNTER  = 0x0009  # free-running; lag tag
```

## Residual lattice

`Observation` is the RAM-readable lattice. Stepper state is `PlayerPhysics`
(`a_held` is tape memory). Floor is `World.ground_y`, not a player field.
Physics grounded is `$001D == 0` (`player_on_ground`); policy `is_in_air`
still uses `y_speed`. Speeds are first-differing-field only, not a second σ+.

| Name | RAM | Width | Lattice | Notes |
|------|-----|-------|---------|-------|
| `x` | `$006D` + `$0086` | u16 | Oπ | `page * 256 + offset` |
| `y` | `$00CE` | u8 | Oπ | 0 = top; floor ≈ 176 on 1-1 |
| `pose` | `$000E` | u8 | Oπ | `0x08` normal, `0x0B` dying, `0x04` flagpole |
| `room` | `$075F`, `$0760`, `$0750` | packed | Oπ | `(world<<16) \| (level<<8) \| area_pointer` |
| `sub_x` | `$0400` | u8 | Oσ | X **position** subpixel (not velocity) |
| `sub_y` | `$0416` | u8 | Oσ | Y **position** subpixel |
| `enemy0_active` | `$000F` | u8 | Oσ+ | first enemy slot flag |
| `enemy0_type` | `$0016` | u8 | Oσ+ | first enemy slot type |
| `energy` | `$075A` | u8 | O† | lives (SMB analog of SM energy) |
| `dead` | `$000E` / `$0770` / `y` | flag | O† | dying, game over, or `y ≥ 240` |
| `velocity_x` | `$0057` | s8 | field | first-diff only |
| `velocity_y` | `$009F` | s8 | field | first-diff only |
| `frame_counter` | `$0009` | u8 | lag | stop scoring later kinematics |
| `on_ground` | `$001D == 0` | flag | physics | `player_on_ground`; air/jump is `1` |
| `x_force` | `$0705` | u8 | physics | `Player_X_MoveForce` |
| `running_speed` | `$0703` | u8 | physics | `RunningSpeed`; brake `$D0` if set |
| `y_move_force` | `$0433` | u8 | physics | `Player_Y_MoveForce` |
| `vertical_force` | `$0709` | u8 | physics | rising / current gravity |
| `vertical_force_down` | `$070A` | u8 | physics | fall gravity |
| `jump_origin_y` | `$0708` | u8 | physics | A-release height gate |
| `a_held` | (tape) | flag | physics | not RAM; previous-frame A |
| `ground_y` | — | u8 | world | flat floor; not on the player |

See `docs/RESIDUAL.md` for planner rules and the first measurement segments.

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
