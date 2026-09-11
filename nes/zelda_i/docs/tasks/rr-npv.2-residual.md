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

## Third red (one ROM trial, stopped)

`Level5EntranceFromL4` start: play 0x76 `(120,205)` mode 5, raft=1 ladder=1
bombs=7 keys=0 TF=`0x0c` whistle=0.

| stage | frames | ok |
|-------|--------|----|
| `level5_clear_0x66` | 1265 | yes |
| `level5_east_key_0x77` | 761 | yes |
| `level5_clear_0x77` | failed | **no** |

Leftover: **room 0x77 mode 17 (death) `(133,157)`**, deaths=1, TF bit 0x10
missing (`TF=0x0c`), keys=1, bombs=7, whistle=0.
Clean Pols Voice (`Level5PolsVoiceController`) without infinite-life.

Changes landed in this sitting:
- Implemented trajectory projection for jumping enemies (`projected`).
- Added tactical leap evasion: when a Pols Voice is leaping towards Link (`state == 1, dist <= 36`), Link computes perpendicular/safe evasive backsteps.
- Added strike timing and retreat against grounded enemies: attacks only when grounded (`state == 0`) and aligned, triggering a 6-frame safe retreat.
- Added arena re-entry: returns Link to central arena (`x=96..144, y=112..165`) if displaced.
- Condensed `dungeon.py` to 978 LOC (strictly under the ~1000 LOC soft max).

Trial analysis:
- Link advanced past the north cross-aisle `(80, 105)` all the way into the central arena at `(133, 157)`.
- Died at `(133, 157)` in the central arena to jumping contact damage.

## Next

Survive 0x77 Pols Voice Clean in central arena `(133, 157)`: fine-tune strike timing and safe spacing during multi-Pols-Voice groundings to finish all 5 kills and claim key drop.

