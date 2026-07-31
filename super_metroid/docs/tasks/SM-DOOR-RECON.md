# TASK SM-DOOR-RECON: Kraid left-door recon probe + report (no pure-green claim)

## Recipe step
diagnostics (supports pure geometry for kraid_to_eye_return)

## Model
Luna

## Own files only
- `scripts/probe/kraid_left_door_recon.py` (**create**)
- `docs/tasks/SM-DOOR-RECON-report.md` (**create**)

Do **not** edit `varia_return.py` / continuous / STATUS / progression.

## Context
- Pure `kraid_to_eye_return` fails at left lip: room 0xA59F, x≈37, y≈307–427,
  pose 138/82 after door-shot attempts
- Source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state`
- Need RAM facts: door transition flags, pose, x/y over time, enemy0, boss bits,
  whether blue door opens

## Read first
- `scripts/probe/kpdr.py` (boot_from_state, pure session pattern)
- `routes/runtime.py` / `dev/common.py` for boot helpers
- `routes/controller_common.py` (hold)
- `routes/kpdr/varia_return.py` (`play_kraid_to_eye_return` current geometry)
- `ram.py` SuperMetroidState fields

## Do (thorough)
1. Write CLI recon script that:
   - Boots the pure source state
   - Applies resource assist only (no progression writes)
   - Runs a short scripted approach to left door (or samples idle + walk LEFT)
   - Every N frames logs JSON lines: frame, room, pose, x, y, game_state,
     door_transition, transition_direction, enemy0_hp, boss_bits[area],
     selected_item
   - Optionally fires 4 standing beam shots facing left and continues sampling
   - Writes report JSON under `debug/` or stdout summary
2. Run it once; capture output
3. Write `docs/tasks/SM-DOOR-RECON-report.md` with:
   - Observed trajectory summary (start → end)
   - Whether room ever changes
   - Hypothesis list for pure fail (door closed, wrong height, pin, enemy)
   - Recommended next geometry card steps (still not continuous)

## Do not
- Claim pure green or promote edges
- Forge door/boss RAM
- Long multi-minute random exploration — bounded script only (≤ ~2k frames)

## Acceptance
- [ ] Script runs and exits 0 with a summary
- [ ] Report file with hypotheses + numbers
- [ ] Diff limited to script + report (+ debug artifact ok if gitignored)

## Verify commands
```bash
uv run python super_metroid/scripts/probe/kraid_left_door_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state \
  --frames 600
test -f super_metroid/docs/tasks/SM-DOOR-RECON-report.md
```
