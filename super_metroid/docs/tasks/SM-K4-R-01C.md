# TASK SM-K4-R-01C: Pure `baby-to-kihunter-return` (gray lock + supers clear)

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_baby_to_kihunter_return`)
- `progression.py` (edge → `controller_dev` after pure)
- `tests/test_progression.py`

## Context
- Source: `scratch/post_eye_to_baby_return.state` → `0xA521`
- Left door is **gray** with `clear_room_enemies` (graph)
- Beams alone leave Mini-Kraid; use weapon 2 (Supers) + `_baby_kraid_sweep`
  (same as forward `play_baby_kraid_to_eye`), then beam-open + exit left

## Status
**DONE ✓ GREEN** (~1248f → `0xA4DA`). Graph `controller_dev`. Source
`post_baby_to_kihunter_return.state` captured.

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure baby-to-kihunter-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_eye_to_baby_return.state
```
