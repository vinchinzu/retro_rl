# TASK SM-ROOM-METAL-02: Metal Pirates — Super Missile combat tactic

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local policy only: `room_b62b_from_b482_to_b5d5`
- optional note: `docs/tasks/SM-ROOM-METAL-02-note.md`

## Context
- METAL-01 **RED**: stationary beam `X` — `enemy0_hp` stayed **1800** (no damage).
- Pin: `room=0xB62B pose=137 x=699 y=187 door_transition=0`
- Planner gate resolved: Metal Pirates need **Super Missiles** (or charge), not
  plain beam. Teleport source starts with ordinary ammo inventory from the
  room fixture — select Super Missile weapon, then one combat span.
- Practice only — not continuous Lower Norfair / not KPDR spine.

## Do
1. **One knob:** replace the combat-clear span with a **Super Missile**
   selection + fire tactic (SELECT cycle to supers, then fire at pirates).
   Preserve exit sequence after combat if present.
2. Isolate green or residual with pin (include `enemy0_hp` if still red).
3. No continuous / STATUS / progression / other rooms.

## Acceptance
- [ ] Isolate green **or** residual with pin + next card
- [ ] Dual-track non-claim
- [ ] If still red, residual must note whether `enemy0_hp` decreased

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_b62b_from_b482_to_b5d5
uv run python super_metroid/scripts/room/run_problem.py run room_b62b_from_b482_to_b5d5
```
