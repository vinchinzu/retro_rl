# Residual — rr-iprz.2 Technodrome pink-Foot jump-behind

**Status:** KEEP (Stage4 dual-green + Tank dual-green). Do **not** STATUS
or rewrite `BASELINE_METRICS.md` from these probes. No commit this sitting.

**Miss class:** none on the KEEP probes. Earlier corridor hop sat on
empty-phase `technodrome_align` (~24k frames, Stage4 timeout at event
`0x0A`). Hop is **tank-only** now.

## Shipped

- `TechnodromeTactics` extracted to `snes/tmnt_iv/tactics/technodrome.py`
  and re-exported from `tmnt_iv.policy` (`policy.py` 2026 → 1734 LOC).
- Raphael (char 8) tank event `0x18`: if blocking Foot `0x6C` is on-lane
  and `adx≤48`, jump-behind (`B`+through) → tap Y → existing toward+Y
  screen throw.
- Fallback: production ram (40f retreat + ≥34f pure-run + toward+Y) when
  far, off-lane, hop misses HP, corridor waves, or not Raphael.
- `blocker_hit_frames` stays **10**. Never A. Never grounded Y+B.

## Live probes (heal=emergency)

Current-code **before** (main-repo production ram, this sitting):

| State | Outcome | Frames | Damage | Heals | Notes |
|-------|---------|-------:|-------:|------:|-------|
| `RaphFullHardTank` `--max-frames 20000` | timeout | 20,000 | 478 | 7 | stayed event `0x18` |
| `RaphFullHardStage4` `--stop-stage-gt 3` | CLEAR | 31,197 | 1,117 | 16 | lives 2→3 |

Published STATUS row (not this sitting): Stage4 **30,379f / 886 / 13**.
Leo-era FullHardTank **9,366f / 232 / 3**.

**After** (jump-behind + ram fallback), two exact repeats:

| State | Outcome | Frames | Damage | Heals | vs before |
|-------|---------|-------:|-------:|------:|-----------|
| `RaphFullHardTank` `--stop-stage-gt 3` | CLEAR ×2 | **12,848** | **236** | **3** | timeout→clear; −242 dmg; −4 heals |
| `RaphFullHardStage4` `--stop-stage-gt 3` | CLEAR ×2 | **29,919** | **1,087** | 16 | −1,278f; −30 dmg; no life_loss |

Tank without `--stop-stage-gt` continues into stage 4 (user command
`--max-frames 20000` then reports timeout at 20k / end_stage 4). Use
`--stop-stage-gt 3` for a tank-only number.

`blocker_jump_behind` is live on tank (~183f) but ram charge is still
the majority (~5,102f). Hop unsticks the empty-foot wait that used to
timeout `RaphFullHardTank`.

## Anti-patterns burned this sitting

- Empty-phase UP/DOWN while waiting to hop (Stage4 24,033× `technodrome_align`).
- Hopping corridor `0x6C` waves (knocks Y lane; let ram own those).
- Mixing B into charge, or Y+B (Power Attack).
- `blocker_hit_frames=8` (Leo KEEP raised continuous Techno 1,022→1,131).

## Next

1. Full-route dry-run before STATUS: hop changes tank length / later RNG.
   Whole-run Technodrome bucket is still the published **1,022**.
2. Hop is a minority of tank frames — a shorter ram once the hop stun
   lands, or a tighter behind-Y→throw gap, is leftover ROI.
3. Do not reopen corridor hop without a Stage4 dual-green.

```bash
PYTHONPATH=snes:. SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics RaphFullHardTank \
  --max-frames 20000 --stop-stage-gt 3 --heal emergency

PYTHONPATH=snes:. SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics RaphFullHardStage4 \
  --max-frames 40000 --stop-stage-gt 3 --heal emergency
```
