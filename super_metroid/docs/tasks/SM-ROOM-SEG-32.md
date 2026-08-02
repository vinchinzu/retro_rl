# TASK SM-ROOM-SEG-32: Dual-track room segment — Acid Snakes Tunnel

## Recipe step
room practice segment (dual-track — **never** continuous evidence)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_afce_from_b026_to_a923.json` (create or edit)
- entry fixture under `custom_integrations/SuperMetroid-Snes/` for **this
  problem only** (bootstrap/teleport state if missing)
- optional residual: `docs/tasks/SM-ROOM-SEG-32-residual.md`
- optional note: `docs/tasks/SM-ROOM-SEG-32-note.md`

Do **not** edit: `routes/continuous.py`, `docs/STATUS.md`, `routes/kpdr/*`,
`progression.py`, other rooms' policies, or any spine controller.

## Context
- Dual-track room farm (Wave 10+): continuous tip work is **parked**.
- One agent ↔ one problem — no cross-room edits (collision guard).
- Queue rank **104**, room `0xAFCE` **Acid Snakes Tunnel**, problem `room_afce_from_b026_to_a923`.
- Board practiceStatus: `state_ready`; state_on_disk=True;
  policy_on_disk=True; mode=`iterate`.
- Practice promote ≠ continuous integrity.

## Read first
- `docs/routes/ROOM_WORK_QUEUE.md`
- `scripts/room/run_problem.py` (bootstrap / scaffold / teleport / run / promote)
- If residual exists for this problem, read the latest `docs/tasks/*-residual.md`
  or note mentioning `room_afce_from_b026_to_a923`.

## Do
1. If no teleport fixture: bootstrap this problem only:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py bootstrap room_afce_from_b026_to_a923
   uv run python super_metroid/scripts/room/run_problem.py teleport room_afce_from_b026_to_a923
   ```
2. Scaffold policy if missing:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py scaffold room_afce_from_b026_to_a923
   ```
3. Iterate isolated run until **green** or honest residual with pin:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py run room_afce_from_b026_to_a923
   ```
4. Promote **only** on green isolated run (practice track):
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py run room_afce_from_b026_to_a923 --promote
   ```
5. Write residual with PROCESS schema. Next card may be a one-knob residual
   for this same problem (`SM-ROOM-SEG-32-R1`) or `none` if green+promoted.

## Do not
- Claim continuous / STATUS green
- Touch another problem's policy or state
- Edit spine controllers or progression
- Forge progression/capacity/boss RAM for green claims
- Spend the session on open-ended exploration outside this problem

## Acceptance
- [ ] Isolated run **GREEN + promote** **or** honest residual with pin
- [ ] Only own-files touched
- [ ] Dual-track non-claim in residual
- [ ] Next card ID + one change filled

## Verify commands
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_afce_from_b026_to_a923
uv run python super_metroid/scripts/room/run_problem.py run room_afce_from_b026_to_a923
# promote only if green:
# uv run python super_metroid/scripts/room/run_problem.py run room_afce_from_b026_to_a923 --promote
```

## Done when
Residual filed (message and/or `SM-ROOM-SEG-32-residual.md`). Planner owns queue
refresh / continuous tip; this card never does continuous compose.
