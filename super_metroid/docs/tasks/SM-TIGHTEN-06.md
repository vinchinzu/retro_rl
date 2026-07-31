# TASK SM-TIGHTEN-06: Offline dwell analysis — Bomb Torizo fight split

## Recipe step
efficiency analysis (report only)

## Model
Flash

## Own files only
- `docs/tasks/SM-TIGHTEN-06-report.md` (**create**)

Do **not** edit controllers, continuous, STATUS.

## Context
`bombs` / bomb torizo splits are multi-thousand frames on continuous tips.
Mirror SM-TIGHTEN-05 / TIGHTEN-01 report shape. Combat may live in
`combat/bomb_torizo.py` + route controller — map ownership carefully.

## Read first
- split_dwell on `start_to_kraid.json` or `start_to_varia.json`
- `combat/bomb_torizo.py`, torizo route play functions
- `docs/tasks/SM-TIGHTEN-05-report.md` if present else TIGHTEN-01

## Do
1. Dwell + reasons for bomb/torizo related splits.
2. Phase map + waste rank + 2–3 implement recipes.
3. Non-claims.

## Acceptance
- [ ] Report complete
- [ ] Report-only diff

## Verify commands
```bash
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --top 15
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --reasons --top 40
```
