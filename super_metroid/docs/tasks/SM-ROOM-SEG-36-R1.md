# TASK SM-ROOM-SEG-36-R1: Dual-track residual — Farming Room upper-platform jump/right

## Recipe step
room practice residual (dual-track — **never** continuous evidence)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_b37a_from_b32e_to_b482.json` (edit this problem only)
- optional residual: `docs/tasks/SM-ROOM-SEG-36-R1-residual.md`
- optional note: `docs/tasks/SM-ROOM-SEG-36-R1-note.md`

Do **not** edit: `routes/continuous.py`, `docs/STATUS.md`, `routes/kpdr/*`,
`progression.py`, other rooms' policies, any spine controller, fixtures,
generated reports outside this problem's run, QUEUE, PROCESS, catalog, or
sm_rev.

## Context
- Parent card **SM-ROOM-SEG-36** filed honest RED (not green, not promoted).
- Problem `room_b37a_from_b32e_to_b482`, room `0xB37A` **Lower Norfair Farming Room**,
  target `0xB482`.
- Parent residual pin (report `recordings/room_clears/room_b37a_from_b32e_to_b482.json`):
  final `room=0xB37A pose=137 x=123 y=219 door_transition=0`,
  `totalFrames=443`, energy resource writes=338, missile resource writes=1,
  progression/capacity writes=0, deaths=0.
- Failure: falling/sticking in lava after the scaffold coarse grounded RIGHT
  approach; policy ended in `0xB37A`; expected `0xB482`.
- Source state (do not replace unless broken):  
  `super_metroid/custom_integrations/SuperMetroid-Snes/room_b37a_from_b32e.state`
- Practice promote ≠ continuous integrity. Dual-track only. No continuous or
  STATUS claims.

## Read first
- `docs/tasks/SM-ROOM-SEG-36-residual.md`
- `docs/tasks/PROCESS.md` (residual schema)
- `policies/room_clears/room_b37a_from_b32e_to_b482.json`
- `scripts/room/run_problem.py`

## One named knob only
Replace the coarse grounded **RIGHT** approach
(`coarse_exit_approach`: RIGHT ×220) with an **initial jump/right traversal
cadence** that keeps Samus on the upper platforms toward exit `0xB482`, still
starting from the same doorway-natural fixture.

Do not retune door-open/wait/enter spans, entry settle, fixtures, other
policies, or any continuous/STATUS surface in the same card.

## Do
1. Edit **only** the named approach knob in this problem's policy JSON.
2. Re-verify isolate (teleport + run required):
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py teleport room_b37a_from_b32e_to_b482
   uv run python super_metroid/scripts/room/run_problem.py run room_b37a_from_b32e_to_b482
   ```
3. Promote **only** on green isolated run (practice track):
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py run room_b37a_from_b32e_to_b482 --promote
   ```
4. File residual `docs/tasks/SM-ROOM-SEG-36-R1-residual.md` with full PROCESS
   schema (pin, acceptance checkboxes, non-claims, verify paste + exit codes).
   Next card may be `SM-ROOM-SEG-36-R2` (one new knob) or `none` if
   green+promoted.

## Do not
- Claim continuous / STATUS green
- Touch another problem's policy or state
- Edit spine controllers, progression, kpdr, continuous routes, or STATUS
- Forge progression/capacity/boss RAM for green claims
- Expand scope beyond the single jump/right upper-platform cadence knob
- Spend the session on open-ended exploration outside this problem

## Acceptance
- [ ] Isolated run **GREEN + promote** **or** honest residual with pin
- [ ] Only own-files touched (this policy + residual/note)
- [ ] Dual-track non-claim in residual
- [ ] Next card ID + one change filled
- [ ] teleport + run executed; residual filed with actual exit codes

## Verify commands
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_b37a_from_b32e_to_b482
uv run python super_metroid/scripts/room/run_problem.py run room_b37a_from_b32e_to_b482
# promote only if green:
# uv run python super_metroid/scripts/room/run_problem.py run room_b37a_from_b32e_to_b482 --promote
```

## Done when
Residual filed (`SM-ROOM-SEG-36-R1-residual.md`). Planner owns queue refresh /
continuous tip; this card never does continuous compose and never claims
STATUS.
