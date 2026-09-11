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

## Failed stage this sitting (1 trial)

`level4_clear_0x12` failed in 268f (end_frame 38054).
Link died in room 0x12 (`link_death`, mode 17) during 5× Vire combat.
Room 0x12 is the final room before Gleeok 0x13.

## Glance (trial 0 leftover)

room **0x12**, mode **17**, xy **(156, 165)**, tf **0x04**, keys **1**, bombs
**5**, health **96**, deaths **1**. PNG `recordings/l4_entrance_tf_t0_final.png`.

## Next sitting

- Room 0x12 Vire combat: Link enters with health ~96 from long continuous
  run; needs tighter combat tuning or dodge handling to avoid early death
  before pushing block 0x68 into Gleeok 0x13.
- Gleeok south-stand + low-HP approach dodge is unit-hardened but unproven
  on Clean continuous tape.

