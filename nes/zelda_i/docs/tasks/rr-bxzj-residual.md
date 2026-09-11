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

## Failed stage this sitting (1 trial)

`level4_gleeok_enter_0x13` failed in 8000f (end_frame 46500, `timeout_208_149`).
Link reached east door at `(208, 149)` in play mode 5, but the door remained shut
(cur_opened_doors=2). Stand path approach to `PUSH_12_STAND (112, 144)` from clear
leftover `(72, 150)` collided with the block at `(96, 144)` before pushing.

## Glance (trial 0 leftover)

room **0x12**, mode **5**, xy **(208, 149)**, tf **0x04**, keys **1**, bombs
**5**, health **96**, deaths **0**. PNG `recordings/l4_entrance_tf_t0_final.png`.

## Next sitting

- `level4_gleeok_enter_0x13`: Approach to `PUSH_12_STAND (112, 144)` needs to route
  around block `(96, 144)` (north via y=117 or south via y=173) when Link finishes
  combat west of the block, then push block LEFT 70f to open east door (doors 2→3).
- Gleeok south-stand + low-HP approach dodge into Triforce 0x08.

