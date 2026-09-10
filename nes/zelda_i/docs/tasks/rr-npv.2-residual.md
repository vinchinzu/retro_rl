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

`QT_QPA_PLATFORM=offscreen uv run pytest nes/zelda_i/tests/test_level5_dungeon.py
nes/zelda_i/tests/test_level5_path.py nes/zelda_i/tests/test_level5_overworld.py -q`
→ 22 passed.

## First red (one ROM trial, stopped)

`Level5EntranceFromL4` start: play 0x76 `(120,205)` mode 5, raft=1 ladder=1
bombs=7 keys=0 TF=`0x0c` whistle=0.

| stage | frames | ok |
|-------|--------|----|
| `level5_clear_0x66` | 1265 | yes |
| `level5_east_key_0x77` | 761 | yes |
| `level5_clear_0x77` | 583 | **no** |

Leftover: **room 0x77 mode 17 (death) `(48,157)`**, deaths=1, TF bit 0x10
missing, whistle still 0. Clean Pols Voice (`Level5PolsVoiceController`)
without infinite-life. Whistle-path RAM waits never ran this sitting.

Do not poke doors/keys/Whistle. Do not retry from this leftover without a
combat policy change. `west_path` / `cellar_path` / `boss_path` still have
`idle(n)` (outside this hop's dest).

## Next

Survive 0x77 Pols Voice Clean (no health write), then continue the polled
whistle path to TF 0x10. Jitter the 0x77 leftover or name a new RAM pin
after a living clear.
