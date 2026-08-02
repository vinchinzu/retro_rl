# TASK SM-ROOM-SEG-19: Dual-track room segment — The Precious Room

## Recipe step
room practice segment (dual-track — **never** continuous evidence)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_d78f_from_da60_to_d72a.json` (create or edit)
- entry fixture under `custom_integrations/SuperMetroid-Snes/` for **this
  problem only** (bootstrap/teleport state if missing)
- optional residual: `docs/tasks/SM-ROOM-SEG-19-residual.md`
- optional note: `docs/tasks/SM-ROOM-SEG-19-note.md`

Do **not** edit: `routes/continuous.py`, `docs/STATUS.md`, `routes/kpdr/*`,
`progression.py`, other rooms' policies, or any spine controller.

## Context
- Dual-track room farm (Wave 10+): continuous tip work is **parked**.
- One agent ↔ one problem — no cross-room edits (collision guard).
- Queue rank **88**, room `0xD78F` **The Precious Room**, problem `room_d78f_from_da60_to_d72a`.
- Board practiceStatus: `state_ready`; state_on_disk=True;
  policy_on_disk=True; mode=`iterate`.
- Practice promote ≠ continuous integrity.

## Read first
- `docs/routes/ROOM_WORK_QUEUE.md`
- `scripts/room/run_problem.py` (bootstrap / scaffold / teleport / run / promote)
- If residual exists for this problem, read the latest `docs/tasks/*-residual.md`
  or note mentioning `room_d78f_from_da60_to_d72a`.

## Do
1. If no teleport fixture: bootstrap this problem only:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py bootstrap room_d78f_from_da60_to_d72a
   uv run python super_metroid/scripts/room/run_problem.py teleport room_d78f_from_da60_to_d72a
   ```
2. Scaffold policy if missing:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py scaffold room_d78f_from_da60_to_d72a
   ```
3. Iterate isolated run until **green** or honest residual with pin:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py run room_d78f_from_da60_to_d72a
   ```
4. Promote **only** on green isolated run (practice track):
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py run room_d78f_from_da60_to_d72a --promote
   ```
5. Write residual with PROCESS schema. Next card may be a one-knob residual
   for this same problem (`SM-ROOM-SEG-19-R1`) or `none` if green+promoted.

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
uv run python super_metroid/scripts/room/run_problem.py teleport room_d78f_from_da60_to_d72a
uv run python super_metroid/scripts/room/run_problem.py run room_d78f_from_da60_to_d72a
# promote only if green:
# uv run python super_metroid/scripts/room/run_problem.py run room_d78f_from_da60_to_d72a --promote
```

## Done when
Residual filed (message and/or `SM-ROOM-SEG-19-residual.md`). Planner owns queue
refresh / continuous tip; this card never does continuous compose.
