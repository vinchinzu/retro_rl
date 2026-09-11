# rr-bxzj residual — Clean L4 Entrance→TF heart-safe Gleeok

Stopped at fixture-live. `route_eligible=false`. Do not STATUS. Do not close
the bead. This sitting reached room 0x12 (pre-Gleeok); died in 0x12 Vire combat.

Isolated runner: `run_level4_entrance_tf.py --from-state Level4Entrance
--no-infinite-life --no-video --trials 1`. Pin `Level4Entrance` (play 0x71).

## Landed this sitting

- `level4_leave_0x31` blocker **resolved**: replaced broken north-gold RIGHT/UP
  attempt from `(80,109)` with south corridor waypoints:
  `(80,173) → (128,173) → (128,133) → (112,133) → (112,141)` on floor.
  Leave clears in 88f without stalls, yo-yos, or clips.
- `level4_east_0x32` lands in 369f (`(112,141)` JOIN UP → SE clip → waypoints → 0x32).
- Downstream hops verified contiguous on Clean:
  - `level4_clear_0x32` (1928f)
  - `level4_stepladder` 0x60 (1566f, `ADDR_LADDER=1`)
  - `level4_exit_0x60` (540f, return to 0x32)
  - `level4_west_0x31` (371f)
  - `level4_maze_west_0x30` (512f)
  - `level4_key_up_0x20` (324f, keys 1→0)
  - `level4_clear_0x20` (6870f)
  - `level4_map_0x21` (449f)
  - `level4_map_pickup_0x21` (297f, `ADDR_MAP|0x08`)
  - `level4_bomb_north_0x21` (433f)
  - `level4_bomb_north_0x11` (377f)
  - `level4_key_0x01` (954f, keys 0→1)
  - `level4_south_0x11` (251f)
  - `level4_bomb_east_0x12` (390f, bomb hole to 0x12)
- `level4_clear_0x12` blocker **resolved**: 5× Vire + split Keese combat clears
  cleanly in 714f (end_frame 38500) with **0 deaths** and **health=96** preserved.
  - `ROOM_12_SPEC.combat` tuned with `avoid_walls=True`, `contact_backstep=20`,
    `engage_dominant_axis=True`, `attack_phase=2`, and `DoorRoute("RIGHT", ...)`.
  - `Room12ViresController` overrides `_off_wall_step` to step east cleanly from
    west bomb hole onto open floor, and slashes in `_engage` when any live enemy
    is in blade reach.
- `level4_gleeok_enter_0x13` blocker **resolved**: 456f (end_frame 38956, phase `DONE`).
  - West-of-block leftover `(72, 150)` routes around block `(96, 144)` via north
    (`y <= 117`), steps east to `x = 112`, then south to `PUSH_12_STAND (112, 144)`.
  - Pushes block LEFT 70f (`PUSH_12_HOLD`) to open east door (doors 2→3).
  - Executes token path through door into room 0x13 cleanly at `(32, 141)`.
- Room 0x13 Gleeok approach geometry fixed:
  - Set `APPROACH_SOUTH_Y = 189` in `boss_combat.py` and `ROOM_13_SOUTH_Y = 189` in
    `occupancy.py`. Link drops south at `x=32` to `y=189` and walks east along the open
    corridor to `x=116` in 101f without hitting the interior wall at `(48, 165)`
    (`approach_south f=101 xy=(116,189) hp=96 dodge_thr=22`).

## Failed stage this sitting (1 trial)

`level4_gleeok_tf` failed in 206f (end_frame 39162, `death` at xy `(126, 133)`).
Link executed south approach to `(116, 189)` cleanly, then advanced to south stand
`(126, 133)` under Gleeok (`(124, 111) + STAND_DY=22`). At `health=96`
(`hearts_hi=6, hearts_lo=0`), Link has zero whole hearts; contact with Gleeok's
body hitbox or fireballs causes instant death without invulnerability flashing.

## Glance (trial 0 leftover)

room **0x13**, mode **17**, xy **(126, 133)**, tf **0x04**, keys **1**, bombs
**5**, health **96**, deaths **1**. PNG `recordings/l4_entrance_tf_t0_final.png`.

## Next sitting

- `level4_gleeok_tf`: Tune low-HP Gleeok engagement when Link enters with `health <= 96`.
  Evaluate increasing south-stand offset (`STAND_DY >= 26`) or evasive sword spacing so
  Link does not overlap the body hitbox, or refine earlier dungeon combat (rooms 0x50,
  0x32, 0x20, 0x12) to preserve `>= 106` health (the lab poke continuous floor).
- Complete Triforce `0x08` collection in room `0x03` on Clean.

