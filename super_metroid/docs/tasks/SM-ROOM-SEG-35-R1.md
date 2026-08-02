# TASK SM-ROOM-SEG-35-R1: Dual-track residual — Fast Ripper Room jump transit

## Recipe step
room practice residual (dual-track — **never** continuous evidence)

## Model
Luna

## Wave type
10→implement

## Own files only
- `policies/room_clears/room_b2da_from_b3a5_to_b6c1.json` (edit this problem only)
- optional residual: `docs/tasks/SM-ROOM-SEG-35-R1-residual.md`
- optional note: `docs/tasks/SM-ROOM-SEG-35-R1-note.md`

Do **not** edit: `routes/continuous.py`, `docs/STATUS.md`, `routes/kpdr/*`,
`progression.py`, other rooms' policies, any spine controller, fixtures,
generated reports outside this problem's run, QUEUE, PROCESS, catalog, or
sm_rev.

## Context
- Parent card **SM-ROOM-SEG-35** filed honest RED (not green, not promoted).
- Problem `room_b2da_from_b3a5_to_b6c1`, room `0xB2DA` **Fast Ripper Room**,
  target `0xB6C1`.
- Parent residual pin (report `recordings/room_clears/room_b2da_from_b3a5_to_b6c1.json`):
  final `room=0xB2DA pose=138 x=853 y=139 door_transition=0`,
  `totalFrames=1281`, energy resource writes=321, missile resource writes=1,
  progression/capacity writes=0, deaths=0.
- Failure: policy ended in `0xB2DA`; expected `0xB6C1`.
- Source state (do not replace unless broken):  
  `custom_integrations/SuperMetroid-Snes/room_b2da_from_b3a5.state`
- Practice promote ≠ continuous integrity. Dual-track only.

## Read first
- `docs/tasks/SM-ROOM-SEG-35-residual.md`
- `docs/tasks/PROCESS.md` (residual schema)
- `policies/room_clears/room_b2da_from_b3a5_to_b6c1.json`
- `scripts/room/run_problem.py`

## One named knob only
Replace the repeated **grounded-left dash/recover** transit
(`grounded_left_transit_with_knockback_recover`: LEFT+B dash + LEFT recover
loop) with a **jumping traversal cadence** that still starts from the same
doorway-natural fixture and still aims left toward exit `0xB6C1`.

Do not retune door-open/wait/enter spans, entry settle, fixtures, other
policies, or any continuous/STATUS surface in the same card.

## Do
1. Edit **only** the named transit knob in this problem's policy JSON.
2. Re-verify isolate (teleport + run required):
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py teleport room_b2da_from_b3a5_to_b6c1
   uv run python super_metroid/scripts/room/run_problem.py run room_b2da_from_b3a5_to_b6c1
   ```
3. Promote **only** on green isolated run (practice track):
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py run room_b2da_from_b3a5_to_b6c1 --promote
   ```
4. File residual `docs/tasks/SM-ROOM-SEG-35-R1-residual.md` with full PROCESS
   schema (pin, acceptance checkboxes, non-claims, verify paste + exit codes).
   Next card may be `SM-ROOM-SEG-35-R2` (one new knob) or `none` if
   green+promoted.

## Do not
- Claim continuous / STATUS green
- Touch another problem's policy or state
- Edit spine controllers, progression, kpdr, continuous routes, or STATUS
- Forge progression/capacity/boss RAM for green claims
- Expand scope beyond the single jumping-cadence knob
- Spend the session on open-ended exploration outside this problem

## Acceptance
- [ ] Isolated run **GREEN + promote** **or** honest residual with pin
- [ ] Only own-files touched (this policy + residual/note)
- [ ] Dual-track non-claim in residual
- [ ] Next card ID + one change filled
- [ ] teleport + run executed; residual filed with actual exit codes

## Verify commands
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_b2da_from_b3a5_to_b6c1
uv run python super_metroid/scripts/room/run_problem.py run room_b2da_from_b3a5_to_b6c1
# promote only if green:
# uv run python super_metroid/scripts/room/run_problem.py run room_b2da_from_b3a5_to_b6c1 --promote
```

## Done when
Residual filed (`SM-ROOM-SEG-35-R1-residual.md`). Planner owns queue refresh /
continuous tip; this card never does continuous compose and never claims
STATUS.
