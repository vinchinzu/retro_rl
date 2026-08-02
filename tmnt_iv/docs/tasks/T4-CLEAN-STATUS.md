# TASK T4-CLEAN-STATUS: STATUS secondary section for Clean full clear

## Recipe step
status

## Model
Flash proposal → **planner apply**

## Wave type
implement

## Own files only
- `docs/STATUS.md` — **secondary** Clean track section only
- `docs/BASELINE_METRICS.md` — optional Clean continuous subsection
- residual: `docs/tasks/T4-CLEAN-STATUS-residual.md`

## Do not
- Change program gate Intervention class away from assisted unless planner
  explicitly decides Clean is the new published default
- Claim assisted dry-run is Clean
- Overwrite assisted video/manifest paths

## Context
- After `T4-CLEAN-STAB` dual green.
- Labels: **Bronze / Clean** for the full hard continuous only.

## Do
1. Add STATUS subsection: Continuous power-on → hard credits (**Clean**).
2. Link clean reports; frames; integrity zeros including e-heals + iframe.
3. Keep assisted program gate table and best-verified assisted result.
4. Update manifest notes if Clean becomes a second published result.

## Acceptance
- [ ] Primary gate still assisted continuous clear
- [ ] Clean full clear documented with evidence paths
- [ ] Intervention class wording correct (Clean)

## Verify commands
```bash
rg -n "Clean|Resource-assisted|Protection|00:57" tmnt_iv/docs/STATUS.md | head -40
```
