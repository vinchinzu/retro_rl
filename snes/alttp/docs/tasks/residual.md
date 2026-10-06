# ALttP route residual — 2026-10-06

`rr-f2mk` acceptance is complete: the Clean Sanctuary-to-Kakariko extension
collects the Heart Container, exits south, and walks west via
`0x13 -> 0x12 -> 0x11 -> 0x18`. Gear gathering remains `rr-7cdd`.

## Leave proof

`recordings/clean_kakariko_20261006/prefix_replay.json` verifies **38277f**
from an actual `NONE` reset and blank SRAM prepared before frame 1. Zero
full-RAM CRC desync, state loads, RAM writes, refill assists or deaths.
`route_goal_ok`, `synchronization_ok` and `clean_chain` are true.

Final outdoors: loaded screen `0x18`, standing quadrant `0x18`,
module/submodule `0x09/0x00`, `(511,1584)`, HP `32/32`, sword/lamp `1`,
follower `$F3CC == 0`, keys `0xFF`, progress/flags `2/0x15`.
The Heart Container increases capacity `24 -> 32` and heals through play.
Bombs, rupees, bottles and net have not been gathered.

| Suffix leg | Frames | End HP |
|------------|-------:|-------:|
| Sanctuary Heart Container | 236 | 32/32 |
| Sanctuary south exit | 534 | 32/32 |
| Westward Kakariko traversal | 1429 | 32/32 |

The first 36078 inputs are the published C1 tape, replayed and CRC-checked
in the same emulator before the new rows. No development pin is loaded.
The suffix controller discloses 454 rollout restores and zero writes;
only the separate forward replay proves Clean. The older prefix's search
history remains disclosed in `docs/STATUS.md`; its committed inputs contain
no restores. C2 (Eastern prize) remains unearned; STATUS is unchanged.

## Reproduce

From `snes/`, export `QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy
SDL_AUDIODRIVER=dummy`, then use the existing audit CLI:

```bash
uv run python alttp/scripts/audit_power_on.py controller \
  --prefix-tape alttp/recordings/clean_escape_20261002/controller.tape.npz \
  --through kakariko --no-survival \
  --out alttp/recordings/clean_kakariko_20261006/prefix_controller.json
uv run python alttp/scripts/audit_power_on.py replay \
  --tape alttp/recordings/clean_kakariko_20261006/prefix_controller.tape.npz \
  --capture 38277:DevClean_Kakariko \
  --out alttp/recordings/clean_kakariko_20261006/prefix_replay.json
```

To reconstruct the ROM-test predecessor, replay the published C1 tape with
`--capture 36078:DevClean_Sanctuary` and a new output path. This development
capture is not a substitute for the power-on replay above.

Verification: two live-ROM whole-suffix offset tests (0,7), each followed
by every-frame CRC forward replay; 369 offline cases; eight repository docs
checks; compileall and diff whitespace check. The known fake sword-exit
fixture `rr-mott` is excluded from the offline run; the unfiltered run still
fails that case. Logs: `rom_tests.log`, `offline_tests.log` in the evidence
folder. The legacy assisted Kakariko pin test is replaced by the Clean
composition tests, all marked `rom`.

## Cleanup and next work

Westward route composition lives in the existing segment registry; the
unused Sanctuary-to-Link's-house wrapper is deleted. `overworld.py` is
985 lines and `opening_route/segment.py` is 943. Heart receipt and door-push
frames are counted, including failures. Kakariko acceptance distinguishes
loaded screen from standing quadrant. Older tapes tolerate newly added
snapshot fields while every recorded field and full-RAM CRC must match.
The prefix CLI rejects assisted, incompatible, wrong-ROM or desynchronized
inputs before dispatching a suffix. No one-off runner remains.

`rr-ihr8`: the current fresh controller fails before Zelda rescue in room
`0x71` at `(889,3976)`, HP `20/24`, key 1, follower 0, at frame 14069.
Its faster courtyard predecessor differs from C1; inspect that exact arrival
before changing rescue navigation. `controller.json` preserves this red.
The published C1 replay remains exact.

Next action (`rr-7cdd`): map and enter the Well from the Clean Kakariko
arrival, earn actual chest resources, then compose Hideout/bottle/net legs
and require a new power-on replay. Keep gear research separate from earned
inventory evidence.

Non-claims: did not STATUS-promote; did not treat a pin as power-on; did
not write follower, sword, keys, capacity or progression; did not copy
approach coordinates into Python; did not overwrite the historical
`recordings/verified_tip_run.json` on a red.
