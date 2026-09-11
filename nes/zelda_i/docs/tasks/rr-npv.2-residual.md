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
→ 26 passed.

## Second red (one ROM trial, stopped)

`Level5EntranceFromL4` start: play 0x76 `(120,205)` mode 5, raft=1 ladder=1
bombs=7 keys=0 TF=`0x0c` whistle=0.

| stage | frames | ok |
|-------|--------|----|
| `level5_clear_0x66` | 1265 | yes |
| `level5_east_key_0x77` | 761 | yes |
| `level5_clear_0x77` | 855 | **no** |

Leftover: **room 0x77 mode 17 (death) `(80,105)`**, deaths=1, TF bit 0x10
missing (`TF=0x0c`), keys=1, bombs=7, health=112 (`0x70`), whistle=0.
Clean Pols Voice (`Level5PolsVoiceController`) without infinite-life. Total frames: 2881.

Changes landed in this sitting:
- `_is_solid`: Marked southwest dead-end pocket (`x < 88 and y > 145`) as solid so Link never retreats or paths south of the west doorway. Corrected left cluster bounds to `56 <= x <= 88 and 109 < y <= 164` to open the north aisle at `y=109`.
- Northward routing from west door: when `lx < 88 and ly > 109`, routes `UP` to `y <= 109` (`route_north_from_west_door`), completely eliminating the retreat into the SW pocket `(48, 182)`.
- North aisle traverse: when `lx < 88 and ly <= 109` and target is east, traverses `RIGHT` (`traverse_east_to_aisle`) across the open north aisle into the central aisle (`x=96..144`).
- Reward collection: added north/east routing in `_collect_reward` for `lx < 88`.
- Unit tests: updated `test_pols_voice_controller_is_solid` to verify SW dead-end pocket solidity and open north aisle; added `test_pols_voice_controller_west_door_routes_north` (26 passed).

Trial analysis:
- SW pocket `(48, 182)` retreat was completely eliminated.
- Link exited west door `(48, 141)` northward to `(48, 109)` and advanced east along the north cross-aisle to `(80, 105)`.
- Link survived 855 frames (vs 325 previously) and dealt heavy damage to multiple Pols Voices (one down to 64 HP, one to 96 HP, two to 144 HP) before dying to contact damage in the cross-aisle at `(80, 105)`.

## Next

Survive 0x77 Pols Voice Clean: refine spacing/evasion against multiple jumping Pols Voices in the north cross-aisle and central aisle `x=96..144`. Once 0x77 is cleared and key collected, continue polled whistle path to TF 0x10.
