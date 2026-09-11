# rr-npv.2 residual — Clean L5 Entrance→TF

Stopped at fixture-live. Not a STATUS claim. `route_eligible=false`.

## Landed

- `level5.path.RamWaitHop` / `wait_ram` / `drive_hop`: dest is RAM (room,
  mode, x/y band, object census, whistle bit). `max_frames` is the budget.
- `whistle_path.py`: `_ops().idle(n)` and `push_dir(frames=N)` folded onto
  those hops. Leftover-relative door bands use Wave 0 `door_band_goal`.
- Pin loader: `scripts/run_level5_whistle_tf.py --clean --no-video`
  (`Level5EntranceFromL4`). `runner.open_env` still ImportErrors (no
  `load_state`); script uses `make_env` + `resync_custom_state`.

## Unit

`QT_QPA_PLATFORM=offscreen uv run pytest nes/zelda_i/tests/test_level5*.py -q`
→ 25 passed.

## First red (one ROM trial, stopped)

`Level5EntranceFromL4` start: play 0x76 `(120,205)` mode 5, raft=1 ladder=1
bombs=7 keys=0 TF=`0x0c` whistle=0.

| stage | frames | ok |
|-------|--------|----|
| `level5_clear_0x66` | 1265 | yes |
| `level5_east_key_0x77` | 761 | yes |
| `level5_clear_0x77` | 325 | **no** |

Leftover: **room 0x77 mode 17 (death) `(48,182)`**, deaths=1, TF bit 0x10
missing (`TF=0x0c`), keys=1, bombs=7, health=112 (`0x70`), whistle=0.
Clean Pols Voice (`Level5PolsVoiceController`) without infinite-life.

Changes landed in this sitting:
- `_is_solid`: Strictly blocked right 2×3 cluster and underneath (`152 <= x <= 184 and y >= 109`), hold `x >= 185` only on `y > 185`. Completely eliminated the previous serial red trap at `(153, 189)`.
- North routing around cluster: Link paths `UP` to `y <= 109` to cross east/west.
- Occupancy tracking: Records cell ahead on >= 4 stuck frames into `blocked_cells`, increments `misses`, and replans; emits `stand_no_path` when blocked.
- Spacing & intercept: Intercept swings on column/row alignment (`<= 24 px`) or proximity (`<= 24 px`), backstep restricted to contact threat (`<= 14 px`).
- Unit tests: Added `test_pols_voice_controller_is_solid` and `test_pols_voice_controller_occupancy_miss_and_stand` (25 passed).

New failure analysis:
Link entered from west door `(32, 141)` into west pocket `(48, 141)`. Pols Voice in central aisle caused Link to route south into SW pocket `(48, 182)` where he was cornered against the south/west walls.

## Next

Survive 0x77 Pols Voice Clean: avoid southwest pocket `(48, 182)` routing, exit west doorway north to `y <= 109` across top aisle or into central aisle `x=96..144`. Then continue polled whistle path to TF 0x10.
