# EPIC / SHELL T4-ASSIST-TECHNO: Cut Technodrome damage (assisted)

Not a single executor session. Spawn thin children:

| Order | Card | Goal |
|------:|------|------|
| 1 | T4-ASSIST-TECHNO-PROBE | RaphFullHardStage4 / Duo metrics only; no policy |
| 2 | T4-ASSIST-TECHNO-KNOB | **One** duo/tank/corridor constant; before/after |
| 3 | T4-ASSIST-TECHNO-STAB | Assisted dry-run deltas; no BASELINE edit |
| 4 | T4-ASSIST-DRYRUN | Planner BASELINE promote |

## Context
- Baseline stage damage **1,022** (21.9% of route) — largest bucket.
- Continuous-faithful: `RaphFullHardStage4` ~30k f / 886 dmg / 13 heals.
- Duo left-flank + stall suppress already landed; tank + wall escape matter.
- Checkpoint gains often fail transfer — STAB dry-run before BASELINE.
- Assisted continuous is already green; damage cut is multi-session.

## Read first
- `docs/tasks/CLEAN_LADDER.md` (assisted thin pattern at bottom)
- `docs/BASELINE_METRICS.md`
- `docs/STATUS.md` (Tokka/Rahzar notes)

## Do not
- Hand this shell as “cut Technodrome in one session”
- STATUS / BASELINE self-apply from a probe win
- Park Slash/other stage knobs here

## Verify pattern
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.run_stage4_segment --help
# PROBE/KNOB: segment or RaphFullHardStage4 probe with emergency assist
# STAB: record_full_hard_run --dry-run
```
