# TASK SM-K4-06D: Diagnostic — weapon / missile door open attempt (no pure claim)

## Recipe step
diagnostics (supports door open hypothesis — **not** pure green requirement)

## Model
Luna

## Own files only
- `scripts/probe/kraid_door_weapon_recon.py` (**create**)
- `docs/tasks/SM-K4-06D-report.md` (**create**)

Do **not** edit varia_return production path unless you add a **dev-only**
flag path that default pure still uses current beam shots (prefer separate
probe script, not controller edit).

## Context
Standing beam shots + short hop still never set `door_transition`. Hypotheses
include wrong weapon, closed blue needing different hit, or geometry. This
card scripts **separate** attempts from the same source:
1. beam shots (baseline)
2. missiles if available in state
3. supers if available
Each attempt: approach lip + open sequence + brief spin; record whether room
or door_transition changes. Resource assist OK; no progression forges.

## Read first
- DOOR-PHASE / DOOR-BLUE / 06B residuals
- `ram.py` selected_item / ammo fields
- blue recon scripts for session pattern
- state ammo on post_varia_to_kraid_pure (may be limited)

## Do
1. CLI with `--mode beam|missile|super|all` (default all).
2. Sample door_transition / room / pose / x / y / selected_item / ammo.
3. Report table: mode → transition? room change? final pin.
4. Recommend production controller change **only if** a mode clearly opens
   door; else recommend next geometry/PLM card.

## Residual required
- Table of modes
- Ammo available on source
- Non-claims

## Do not
- Force pure green by warping
- continuous / STATUS
- Leave production controller on a failing weapon experiment without residual

## Acceptance
- [ ] Probe runs exit 0
- [ ] Report with mode table
- [ ] No STATUS

## Verify commands
```bash
uv run python super_metroid/scripts/probe/kraid_door_weapon_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state \
  --mode all
```
