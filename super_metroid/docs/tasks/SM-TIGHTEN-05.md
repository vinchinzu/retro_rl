# TASK SM-TIGHTEN-05: Offline dwell analysis — Spore Spawn fight split

## Recipe step
efficiency analysis (report only — no controller edit)

## Model
Flash

## Own files only
- `docs/tasks/SM-TIGHTEN-05-report.md` (**create**)

Do **not** edit controllers, continuous, STATUS.

## Context
On `start_to_kraid` / `start_to_varia` dwell tables, `spore_spawn_activated`
is top dwell (~12,182f). Boss pipeline owns fight; report must map controller
ownership + waste candidates + 2–3 future implement recipes (like TIGHTEN-01).

## Read first
- `scripts/export/split_dwell.py` usage
- `recordings/start_to_varia.json` or `start_to_kraid.json` (whichever exists)
- spore fight controller / combat path (find via grep)
- `docs/tasks/SM-TIGHTEN-01-report.md` report shape
- `docs/BOSS_PIPELINE.md` (read only)

## Do
1. Run split_dwell + reasons on best available continuous report JSON.
2. Map split → file:function phases with line/reason labels.
3. Rank waste; propose 2–3 implement recipes with risks + verify command
   (`--to spore` or `--to kraid` as appropriate).
4. Explicit non-claims; no frame savings claimed.

## Acceptance
- [ ] Report with phase table + recipes
- [ ] No code edits outside report

## Verify commands
```bash
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --top 15
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --reasons --top 30
```
