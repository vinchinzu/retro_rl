# TASK SM-K4-R-03C: Zeela→Warehouse — second-drop climb only (Hi-Jump)

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_zeela_to_warehouse_return` **second-drop climb block only**)
- optional residual: `docs/tasks/SM-K4-R-03C-residual.md`

## Context
- R-03B **RED**: reverse class correct (not floor-left); bottom reverse-roll reaches
  pin band but **second-drop climb stalls**:
  `room=0xA471 pose=65 x=122 y=409 door_transition=0` (~5660f)
- Floor-door guard works (no silent wrong door).
- In-place `A` + `A+UP+X` cycles do **not** lift from y=409.
- Loadout on pure source includes **Hi-Jump** (`equipped_items=4101` ≈ HJ+Morph+Varia).
- Forward drops open with unmorph jump+shot; reverse must **gain height** under the
  mid ledge (forward middle band ~`y<=325`, `x≈105`).
- Prefer existing primitive `vertical_hop` from `controller_common` if it fits;
  else one local Hi-Jump wall-run / crouch-load + A vertical sequence.
- Do **not** retouch: bottom reverse-roll gate, first-drop climb, Warehouse door
  exit, or `play_kihunter_to_zeela_return`.

## Do
1. **One change:** replace only the second-drop climb loop (from bottom settle
   until `samus_y <= 325`) with a Hi-Jump capable vertical maneuver aimed at
   the mid band. Keep x band near pin (~100–130) unless a 1-line recenter is
   required for the new maneuver.
2. Pure probe same source; optional `--output` only if green.
3. Residual PROCESS schema if still RED (include min_y if tracked).

## Acceptance
- [ ] Pure green → `0xA6A1` **or** residual with pin + next card + one change
- [ ] Only second-drop climb block changed (diff review)
- [ ] Floor-door guard retained

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state
```

## Do not
- Full rewrite of the hop (R-03B already set class)
- continuous / STATUS / graph promote
- Touch kihunter→zeela redesign
