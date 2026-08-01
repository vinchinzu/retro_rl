# TASK SM-K4-R-01B: Pure `eye-to-baby-return` (jump mid-room + left door)

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_eye_to_baby_return`)
- `progression.py` (edge → `controller_dev` after pure green)
- `tests/test_progression.py` (lock verification)

## Context
- Source: `scratch/post_kraid_to_eye_return.state` → `0xA56B`
- Floor walk pins mid-room ~x=373 pose 138; jump-left clears
- Left hatch is blue; open with X-only beams then jump-enter

## Status
**DONE ✓ GREEN** (~651f → `0xA521`). Graph `controller_dev`. Source
`post_eye_to_baby_return.state` captured.

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure eye-to-baby-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kraid_to_eye_return.state
```
