# TASK SM-ARCH-RED-DIAG: Pure RED video clip + PLM/door snapshot

## Recipe step
efficiency

## Model
Flash

## Wave type
implement

## Own files only
- `scripts/probe/` diagnostics helpers (or small module under `routes/` if that
  matches existing probe patterns)
- optional residual: `docs/tasks/SM-ARCH-RED-DIAG-residual.md`

Do **not** change geometry controllers or continuous tip tables except to
hook a failure path that already exists (prefer probe-side capture).

## Context (minimal)
- On pure RED, next agent often lacks pin / clip / door RAM context
- Goal: auto-capture short clip + PLM/door snapshot; attach paths to residual
- Related debt: `SM-ARCH-RED-DIAG` in BACKLOG
- Wave: `docs/tasks/WAVE-11.md`

## Read first
- `scripts/probe/kpdr.py` pure probe failure / pin-json path
- existing debug pin writers under `debug/`
- `docs/tasks/PROCESS.md` residual probe-pin section
- `docs/ARCHITECTURE.md` efficiency debt notes

## Do
1. When a pure probe goes RED, auto-capture a short clip (or frame dump if
   video is too heavy) + PLM/door RAM snapshot under repo-relative `debug/`.
2. Surface paths in pin JSON and residual so the next agent is not debugging dark.
3. Residual → next pure card that can use the new diagnostics (or
   `SM-K4-FROG-SPEEDWAY-R1` / planner pick).

## Do not
- Change any geometry timings or continuous path
- STATUS-promote
- Require absolute home paths in residual paste

## Acceptance
- [ ] RED path produces artifact paths in pin or residual schema
- [ ] Green pure path unchanged (no extra failure)
- [ ] Residual next card ID + one change

## Verify commands
```bash
# Prefer a known-safe unit or dry-run path if available; otherwise document
# manual pure RED exercise in residual without claiming geometry green.
rg -n "pin.json|PLM|door_transition|clip|snapshot" super_metroid/scripts/probe/kpdr.py super_metroid/scripts/probe/ | head -40
```

## Done when
Residual filed with artifact paths. No continuous claim.
