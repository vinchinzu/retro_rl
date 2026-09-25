# Plan: Super Double Dragon

Verified segments are in `docs/STATUS.md`. RAM is in `docs/ram_map.md`.

## Next

1. Natural Mission 3 stairs from area `0x19` to `0x1A`, then the gym floor and
   the Chin brothers at `0x1B`. Compare event and camera bytes at the archived
   stair entry with `Area19_Clear`. Replace the `Stage4` transition clone with
   that natural boss clear.
2. Mission 5 waves after `Stage5_FirstClear`, through area `0x1E`.
3. Missions 6 and 7, then the ending.
4. One title-to-ending run with `--no-dev-heal` and no transition clone.

Use `scripts/run_area.py` one area at a time. Add a policy branch only for a
collision or boss that the current policy misses. A development heal or a
cloned transition is not a natural clear.
