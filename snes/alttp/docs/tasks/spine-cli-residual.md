# Opening spine CLI residual (`rr-ccxt.16`)

Planner owns `docs/STATUS.md`. Did not STATUS-promote. Did not change
`VERIFIED_TIP_SEGMENT_IDS`. Did not overwrite `recordings/verified_tip_run.json`.

## This sitting

`--through room_72` fail-closed stub. Stairs (`room_01_down_to_0x72`) may
be isolated or even natural_entry from a sibling sitting, but they are
**not continuous**. CLI must not run stairs on a clean power-on until a
later sitting composes them into `full_tip`. `--through` choices:

| `--through` | Behavior |
|-------------|----------|
| `room_50` (default) | `full_tip.run_to_verified_tip` (already continuous). Leftover JSON: `recordings/opening_spine_room_50.json` |
| `room_01` | Fail-closed stub. Blocker: `room_01 not on continuous tip`. Graph edge `room_50_east_to_0x01` stays `natural_entry`; not composed, not claimed continuous. |
| `room_72` | Fail-closed stub. Blocker: `room_72 not on continuous tip; stairs not composed into full_tip`. Leftover JSON: `recordings/opening_spine_room_72.json`. `ok=False`, `leftover=null`, `continuous=False`. |
| `zelda` | Fail-closed stub. Blocker: `zelda not on continuous tip; $F3CC==1 not measured`. |

`--no-video` is the default (`BooleanOptionalAction` on `--video`, default
off). MP4 encode is not wired; leftover JSON only. SDL dummy in the
docstring / AGENTS command. Unit tests: `alttp/tests/test_opening_spine.py`
(argparse + fake dispatch, no ROM).

## Leftover glance

No ROM run this sitting. Fail-closed stubs write leftover JSON with
`leftover: null` and do not boot. A live `room_50` leftover glance is
room hex, module/submodule, xy, sword, `$F3CC` follower, keys — same
shape as `leftover_glance()`.

## Next

Do not promote `0x50→0x01` or stairs to continuous from this CLI. Compose
the natural-entry edge only after a clean power-on leave at `0x01` (drive
from ROOM_WORK_QUEUE + `rr-ccxt.4` / `rr-ccxt.1`). Compose stairs into
`full_tip` only after a clean power-on leave at `0x72`. Zelda stays
fail-closed until `$F3CC==1` is measured (`rr-ccxt.3`).
