# TASK SM-K4-R-CLIMB-REDESIGN: Planner redesign — Kihunter alcove climb

## Recipe step
1 pure controller (planner redesign — **not** one-knob cadence)

## Model
Planner (Grok)

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return` only)
- optional residual: `docs/tasks/SM-K4-R-CLIMB-REDESIGN-residual.md`
- optional diagnostic: `scripts/probe/kihunter_climb_redesign.py` (create if useful)

## Context
Wave 8c/d closed RED on kihunter→zeela:

| Evidence | Finding |
|----------|---------|
| R-02D..F | Cadence / right-cap / launch-band knobs all timeout `y≈395` |
| CLIMB-RECON | **108/108** trials `min_y=371`; no upper land; no Baby |
| Least-bad recon | left=32, cap=450, up_shot — still **91px** short of `y<280` |
| Forward hint | `play_kihunter_to_baby_kraid` **morph-bombs floor shot blocks** near **x≈350**, then drops |
| MapRando (room 81) | bottom-right door → break floor shot blocks (obstacle D) → Hi-Jump to junction → left vertical door |

**Planner call:** stop R-02G cadence spam. Redesign the maneuver class.

### Redesign thesis
Right-capped spinjump+shot from the baby-door alcove never sits under the
drop-shaft / shot-block hole. New controller should:

1. **Seek the shaft band** near the known forward bomb x (~340–370), without
   planting forever on the hard wall at x≈357.
2. **Clear first, climb second** — standing/crouch UP+X and/or morph bombs,
   then a **clean vertical Hi-Jump** (not simultaneous rightward sprint).
3. Keep Baby guard (`0xA521` fail-loud) and Zeela window `x∈[96,160]` post-land.
4. Multi-strategy bounded attempts OK in **one planner card** (this is the
   redesign), but each strategy is a named maneuver class, not a cadence grid.

## Source
`super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state`
→ room `0xA4DA`, natural lower-right (~x=465 y=378)

## Do
1. Replace `play_kihunter_to_zeela_return` climb section with the redesign
   thesis above (seek shaft → clear → vertical HJ → upper window → Zeela drop).
2. Pure probe from named source; on green, write
   `scratch/post_kihunter_to_zeela_return.state`.
3. Residual if still RED: pin + which strategies tried + next **planner** action
   (not another Luna cadence card).
4. No continuous / STATUS / graph promote.

## Do not
- Ship another one-knob cadence residual as the “fix”
- Touch `continuous.py` / `STATUS.md`
- Forge progression / door / capacity RAM
- Unblock SM-K4-R-03 without pure green + source capture

## Acceptance
- [ ] Pure green → ordinary `0xA471` **or** residual with multi-strategy pin table
- [ ] Baby still fail-loud
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green
- [ ] No STATUS claim

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
uv run pytest super_metroid/tests/test_controller_common.py -q
```
