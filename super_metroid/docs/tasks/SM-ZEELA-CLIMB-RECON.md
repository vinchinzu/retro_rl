# TASK SM-ZEELA-CLIMB-RECON: Diagnostic grid — lower Zeela climb to mid band

## Recipe step
diagnostic recon (not pure-green claim)

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe/zeela_climb_recon.py` (create; or extend an existing probe module if one fits cleanly)
- `docs/tasks/SM-ZEELA-CLIMB-RECON-report.md` (create)
- Do **not** edit `routes/kpdr/kraid_return.py` in this card

## Context
- R-03 / R-03B / R-03C all **RED** on reverse climb:
  - R-03: floor-left door `x=19 y=395 dt=1` (wrong class)
  - R-03B: reverse-drop class; second-drop stall `x=122 y=409 pose=65`
  - R-03C: Hi-Jump wall-run/crouch; still stall `x=89 y=395 pose=2`
- One-knob climb spam is the wrong process class (same lesson as Kihunter
  108/108 min_y=371). Need geometry recon **before** R-03D redesign.
- Source: `scratch/post_kihunter_to_zeela_return.state` room `0xA471`
  start ~`x=403 y=362`. Target mid band for reverse second drop:
  forward middle ~`y<=325 x≈105`; upper warehouse door needs `y<=200`.
- Loadout has Hi-Jump (equipped_items=4101).

## Do
1. Write a **read-only / ephemeral** recon script that from the source state:
   - optionally morph-rolls left to candidate x bands (e.g. 80–160, 40–80, 160–220)
   - for each band, tries a small matrix of climb classes (≤8 total strategies), e.g.:
     - standing A spam / crouch-load A
     - LEFT/RIGHT wall-run + A (Hi-Jump)
     - morph bomb-jump cycles
     - unmorph + UP+X open + jump (forward-drop reverse shot)
   - records per trial: start_x, strategy, min_y, end_x/y/pose, door_transition, frames
2. Write report table: which (if any) strategies break `y=395` floor pin; best min_y.
3. Residual/report ends with **one** recommended maneuver class for R-03D
   (or PLANNER redesign if all min_y stuck at floor).
4. No continuous / STATUS / controller promote.

## Acceptance
- [ ] Report with ≥6 trials tabulated
- [ ] Explicit best_min_y across grid
- [ ] Next card ID = SM-K4-R-03D (if a working class found) or PLANNER redesign
- [ ] kraid_return.py untouched

## Verify
```bash
uv run python super_metroid/scripts/probe/zeela_climb_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
test -f super_metroid/docs/tasks/SM-ZEELA-CLIMB-RECON-report.md
```

## Do not
- Claim pure-green
- Edit play_zeela_to_warehouse_return
- continuous / STATUS
