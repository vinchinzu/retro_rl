# EPIC T4-CLEAN-S9: Starbase + form-2 Clean (shell)

## Recipe step
epic / tracker

## Model
Planner only

## Children (spawn when unlocked)

| Order | Card ID | Goal |
|------:|---------|------|
| 0 | T4-INFRA-PROBE-S9 | `probe_stage9_clean.py` (+ form-2 entries) |
| 1 | T4-CLEAN-S9-PROBE | Baseline JSON |
| 2 | T4-CLEAN-S9-WAVE | Starbase waves pizza-only (no form-2 yet) |
| 3 | T4-CLEAN-S9-F2 | Form-2 kill with **iframe frames == 0** |
| 4 | T4-CLEAN-S9-REACH | Metric progress if WAVE/F2 RED |
| 5 | T4-CLEAN-S9-SUITE | Required entries |
| 6 | T4-CLEAN-S9-STAB | Suite + assisted dry-run |

## Context
- Stage bytes **8–9**. **Hard Clean gate:** form-2 without Protection iframe write.
- Pairs with `T4-ASSIST-IFRAME` (shrink assist first is OK).
- Hover Foot need jump-slash. Ladder: [CLEAN_LADDER.md](CLEAN_LADDER.md).

## Do not
- Claim whole-run Clean here → residual to FULL-ATTEMPT
- Bundle WAVE + F2 + continuous in one executor card
- Executor session on this epic shell
