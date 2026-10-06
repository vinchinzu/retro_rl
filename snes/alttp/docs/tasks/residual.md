# ALttP review fixes — 2026-10-06

`rr-ihr8`: the fresh controller rescue regression is fixed and verified by
forward power-on replay. The other three review findings are also corrected.

## Cause and changes

The boomerang chest updates its opened bit and inventory before the delayed
item message starts. `open_free_chests` returned while Link was frozen at
`(889,3976)` despite ordinary module/action bytes. It now waits for the actual
item/text handoff and settles the receipt. The existing mapped departure
works; no new geometry is needed.

Cell-entry combat candidates now step only the emulator core and restore the
root before replaying the selected inputs through the live environment.
Discarded candidates cannot charge the live ledger or tape. Forecast restores
use the existing rollout counter; an exception commits no inputs. If every
candidate fails, only the final attempt and its stand are committed/reported.

What-if saves reject matching canonical source/destination paths before RAM
writes or saving, including symlink aliases. Scan pins use the shared sidecar
schema with `kind="scan"`, retaining tape, frame, location and pin details.

## Leave proof

Fresh `NONE` power-on controller previously failed at frame **14069**, room
`0x71`, `(889,3976)`, HP20/24, key1, follower0. With the fixes it rescues Zelda
in **18548 committed frames**. Controller search remains development evidence:
904 disclosed restores (896 rollout restores), zero writes/refills.

`recordings/review_fixes/review_after_replay.json` plays those inputs forward
from power-on with blank SRAM prepared before frame 1. `route_goal_ok`,
`synchronization_ok`, and `clean_chain` are true; zero CRC desync, state loads,
RAM writes, refills or deaths. Leave RAM: room `0x80`, module/submodule
`0x07/0x00`, `(352,4168)`, HP24/24, sword/lamp1, follower `$F3CC == 1`, keys0.

The published C1 Sanctuary tape still replays **36078f** exactly, with zero
desync, state loads, writes or refills. Evidence:
`recordings/review_fixes/review_published_c1_replay.json`.

The natural `Full_room_81` regression also clears the entire rescue (7377f),
with the chest receipt followed by the original NW waypoint and alcove wrap.
The forced combat retry reproduces the old 21 committed / 22 observed frames
and discarded 8HP hit; the fix yields 21 / 21 and zero damage. Real-ROM combat
inputs replay to identical full RAM from both the guard pin and natural cell
entry. Source-preserving what-if saves and synchronized scan provenance are
also verified on the real ROM.

## Reproduce

From `snes/`, export `QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy
SDL_AUDIODRIVER=dummy`, then use new output names:

```bash
uv run python alttp/scripts/audit_power_on.py controller \
  --through zelda --no-survival \
  --out alttp/recordings/review_fixes/review_after_controller.json
uv run python alttp/scripts/audit_power_on.py replay \
  --tape alttp/recordings/review_fixes/review_after_controller.tape.npz \
  --out alttp/recordings/review_fixes/review_after_replay.json
uv run pytest alttp/tests/test_open_chests.py \
  alttp/tests/test_cell_entry_combat.py alttp/tests/test_pin_tools.py \
  alttp/tests/test_tape.py -m "not ml" -q
```

## Verification and remaining work

Focused checks including ROM tests: **23 passed**. Broader ALttP offline gate:
**377 passed**, excluding the pre-existing fake sword-exit fixture failure
`rr-mott`. Core: **551 passed**. Repository docs: **8 passed**. Compileall and
changed-file whitespace checks pass. The unfiltered ALttP offline gate still
fails `test_run_from_sword_exit_ok`; this failure predates the fixes.

The longer fresh controller also reaches Kakariko in **34714f**. Its separate
forward replay, `recordings/review_fixes/review_kakariko_replay.json`, passes
route goal, synchronization and Clean acceptance with zero CRC desync, state
loads, RAM writes, refills or deaths. Leave: outdoors screen `0x18`,
`(511,1584)`, HP32/32, sword/lamp1, follower0, progress/flags2/0x15.
The controller discloses 4681 restores (4673 rollout restores); only the
forward replay proves the committed path. Prior published Kakariko evidence
is preserved. Gear gathering (`rr-7cdd`) is the next action.

Non-claims: did not STATUS-promote; did not treat development pins as power-on
proof; did not write follower, sword, keys, capacity or progression during the
route; did not copy approach coordinates into Python; did not overwrite
published recordings. Inventory mutation is confined to the explicit what-if
copy test/tool.
