# TASK SM-SPAZER-PURE: Pure Below Spazer → Spazer collect → return

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/spazer.py`
- `routes/kpdr/registry.py` (register pure segment name only if pattern requires)
- `scripts/probe/kpdr.py` (wire pure name if needed)
- `tests/test_spazer_pure.py` or extend scaffold tests
- residual: `docs/tasks/SM-SPAZER-PURE-residual.md`

## Context
- Epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md)
- Source: from `SM-SPAZER-SRC` (Below Spazer, Spazer not held)
- Goal: enter `0xA447`, collect real Spazer PLM, return to `0xA408` with beam
  collected. Prefer natural pedestal approach; use **walljumps** where red-room
  ledges require them (one residual phase if geometry stalls).
- Reuse walljump patterns from Bubble only as style reference — do not couple
  modules.
- Human ref: `docs/tasks/refs/early_spazer_red_room.png`

## Read first
- `docs/tasks/SPAZER_EARLY.md`
- `docs/SOURCE_STATES.md` (Spazer pure source row)
- `routes/kpdr/spazer.py` (scaffold)
- `routes/kpdr/bubble_mountain_mid.py` (walljump open-loop style — read only)
- `routes/kpdr/red_tower.py` (Bat / Below Spazer neighbor)

## Do
1. Implement `play_below_spazer_to_spazer` → `play_spazer_collect` →
   `play_spazer_return_to_below` (or one composed `play_spazer_detour`).
2. Success: room `0xA408` after return **and** Spazer present in
   `collected_beams` (natural PLM; no RAM poke).
3. On RED: residual with pin (x,y,pose,room) + **one** next residual card if
   walljump phase needed (`SM-SPAZER-PURE-R1` style — create residual only).
4. Wire pure probe name `spazer-detour` (or project-consistent kebab name).

## Do not
- Continuous compose / STATUS
- Progression writes or beam writes
- Force-pass from practice policy JSON
- Second interacting knob without residual

## Acceptance
- [ ] Pure probe green from listed source state
- [ ] Spazer collected naturally; return room `0xA408`
- [ ] Residual schema if RED (next ID + one change)
- [ ] Narrow unit/probe tests green

## Verify commands
```bash
# Replace SOURCE with SM-SPAZER-SRC path from SOURCE_STATES.md
uv run python super_metroid/scripts/probe/kpdr.py pure spazer-detour \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/<SRC>.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_detour_pure.state
uv run pytest super_metroid/tests/test_spazer_scaffold.py -q
```

## Done when
Pure GREEN residual → next `SM-SPAZER-GRAPH`. If RED, residual R1 one-knob only.
