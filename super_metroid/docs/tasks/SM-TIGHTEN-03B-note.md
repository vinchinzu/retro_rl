# SM-TIGHTEN-03B: Terminator Exit Idle Trim

## Choice

Recipe B uses the safer bounded-timeout option. The Terminator exit timeout in
`play_parlor_to_main_shaft` is reduced from `900` to `600` frames. The existing
`_hold_until_room` loop still checks the room every frame and returns
immediately when Green Pirates (`0x99BD`) is observed, so directional input is
not held after the room transition is detected.

## Exact Control-Flow Change

Before:

```python
_hold_until_room(session, 0x99BD, 900, "LEFT", "A", "B", "X", ...)
```

After:

```python
_hold_until_room(session, 0x99BD, 600, "LEFT", "A", "B", "X", ...)
```

Only the `exit_terminator` timeout changed. The Terminator bomb-tunnel
`8 * (45 frames LEFT+X, 15 frames LEFT)` timing was not changed.

## Verification / Residual

Required unit verification:

```bash
uv run pytest super_metroid/tests/test_post_spore_controller.py super_metroid/tests/test_controller_common.py -q
```

Planner-only continuous verification remains:

```bash
uv run python super_metroid/scripts/record/continuous.py --to spore --no-video
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_spore.json --top 20
```

No frame savings are claimed until the planner re-records the continuous
prefix and compares the split dwell. The speculative `200`-`500` frame saving
range is not claimed. Rollback is the one-line timeout change from `600` back
to `900` if the continuous prefix fails or times out.

This task does not STATUS-promote the route and does not forge progression,
capacity, or boss-bit RAM.
