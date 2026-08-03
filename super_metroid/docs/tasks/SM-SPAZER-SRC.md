# TASK SM-SPAZER-SRC: Continuous-like Below Spazer source for Spazer pure

## Recipe step
source capture

## Model
Luna / Flash

## Wave type
implement

## Own files only
- Capture under `custom_integrations/SuperMetroid-Snes/scratch/`
  (e.g. `post_below_spazer_for_spazer_pure.state`) — **gitignored state OK**
- `docs/SOURCE_STATES.md` — one new row
- residual: `docs/tasks/SM-SPAZER-SRC-residual.md` (optional)

## Context
- Epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md)
- Need continuous-like entry in Below Spazer `0xA408` **before** Spazer collect,
  natural door from Bat (not door-warp spawn that grants Spazer).
- Preferred provenance: power-on continuous `--to below_spazer` tip end state,
  or pure handoff from existing `continuous_like` / bat chain.
- Loadout expected: early Red Brinstar (missiles + supers; **no** requirement
  for Hi-Jump / Varia / Speed). Walljump path must work in this window.

## Read first
- `docs/SOURCE_STATES.md`
- `docs/tasks/SPAZER_EARLY.md`
- `docs/STATUS.md` Below Spazer continuous section

## Do
1. Capture state at Below Spazer entry (or stable dwell just inside `0xA408`)
   with Spazer **not** yet collected (`collected_beams` lacks Spazer bit).
2. Record room id, x/y, beams, items, door history in residual + SOURCE_STATES.
3. Do not edit controllers beyond optional probe one-liner in residual notes.

## Do not
- Door-warp into Spazer room and call it natural
- Write beam/item RAM to fake collect
- STATUS promote

## Acceptance
- [ ] State path documented in `SOURCE_STATES.md`
- [ ] Room `0xA408`, Spazer not collected
- [ ] Residual points to `SM-SPAZER-PURE`

## Verify commands
```bash
# After capture — probe room id only (adjust path to actual state):
uv run python -c "
from pathlib import Path
# document-only card: residual must print room + beams from the new state
print('see residual for state path + room assert')
"
rg -n "spazer|0xA447|below_spazer_for_spazer" super_metroid/docs/SOURCE_STATES.md
```

## Done when
SOURCE_STATES row + residual with next = `SM-SPAZER-PURE`.
