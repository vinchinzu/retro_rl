# TASK SM-CLEAN-STATUS: STATUS secondary section for Clean tip

## Recipe step
status

## Model
Flash proposal → **planner apply**

## Wave type
implement

## Own files only
- `docs/STATUS.md` — **secondary** Clean track section only
- `docs/routes/MILESTONES.md` + `MILESTONES.csv` — mark `C-BOMBS` continuous
- residual: `docs/tasks/SM-CLEAN-STATUS-residual.md`

## Do not
- Change program gate Intervention class from Resource-assisted
- Change primary continuous tip (Frog) or best-verified assisted result
- Claim Clean full clear

## Context
- After `SM-CLEAN-STAB` dual green.
- Labels: **Bronze / Clean** for the bombs tip only.

## Do
1. Add STATUS subsection: Continuous power-on → Bomb Torizo (**Clean**).
2. Link clean reports; frames; integrity zeros including resource writes.
3. Update MILESTONES Clean rows to `continuous`.
4. Keep assisted program gate table unchanged.

## Acceptance
- [ ] Primary gate still Resource-assisted Frog
- [ ] Clean bombs documented with evidence paths
- [ ] MILESTONES Clean marks updated

## Verify commands
```bash
rg -n "Clean|Resource-assisted|Frog" super_metroid/docs/STATUS.md | head -40
```
