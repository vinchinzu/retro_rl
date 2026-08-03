# EPIC T4-CLEAN-FULL: Continuous Clean hard credits (shell) ★

## Recipe step
epic / continuous (planner)

## Model
Planner

## Children

| Order | Card | Goal |
|------:|------|------|
| 0 | [T4-CLEAN-FULL-ATTEMPT](T4-CLEAN-FULL-ATTEMPT.md) | Run clean dry-run; residual first death stage only |
| 1 | (loop) failing stage REACH/CKPT/BRIDGE/F2 | Fix thin rung — not whole route |
| 2 | [T4-CLEAN-STAB](T4-CLEAN-STAB.md) | Dual re-verify when attempt GREEN |
| 3 | [T4-CLEAN-STATUS](T4-CLEAN-STATUS.md) | STATUS secondary — planner only |

## Depends
- Clean infra done
- Prefer S1–S9 thin SUITE/F2 green (S9-F2 is hard gate)
- Never overwrite assisted `tmnt_iv_full_hard_*` baselines

## Context
- Assisted continuous already green. Clean = 0 e-HP + 0 iframe + 0 lives.
- Whole-run Clean is **much harder** than stage assisted segments.

## Do not
- Hand this epic as “make Clean continuous green” to Gemini
- STATUS primary re-label of assisted run as Clean
