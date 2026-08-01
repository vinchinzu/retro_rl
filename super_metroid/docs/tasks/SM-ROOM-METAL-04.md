# TASK SM-ROOM-METAL-04: Metal Pirates — Super Missile combat (fixture unlocked)

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_b62b_from_b482_to_b5d5.json` only
- optional note: `docs/tasks/SM-ROOM-METAL-04-note.md`

## Context
- METAL-03 **PARTIAL**: fixture gate closed — teleport shows
  `max_super_missiles=5`, supers 5/5. Combat still RED:
  `room=0xB62B pose=137 x=699 y=187 door_transition=0 enemy0_hp=1800`
  (`selected_item=2` supers selected but no HP drop).
- One knob: combat fire / aim / approach span only so supers actually hit.
- Do **not** re-bootstrap fixture. Practice only.

## Do
1. **One knob** on combat-clear spans (fire hold, aim UP/DOWN, walk into pirate
   range, or multi-fire). Keep SELECT supers setup if it already lands item=2.
2. Residual must report `enemy0_hp` and `max_super_missiles`.
3. No continuous / STATUS.

## Acceptance
- [ ] Isolate green **or** residual with pin + next card
- [ ] Dual-track non-claim
- [ ] `enemy0_hp` reported if still red

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_b62b_from_b482_to_b5d5
uv run python super_metroid/scripts/room/run_problem.py run room_b62b_from_b482_to_b5d5
```
