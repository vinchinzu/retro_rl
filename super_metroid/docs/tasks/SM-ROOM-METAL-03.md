# TASK SM-ROOM-METAL-03: Metal Pirates — fixture supers capacity (not SELECT spam)

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- entry fixture: `custom_integrations/SuperMetroid-Snes/room_b62b_from_b482.state`
  (+ provenance json if the bootstrap writes it)
- optional: re-run only of existing policy
  `policies/room_clears/room_b62b_from_b482_to_b5d5.json` (no tactic rewrite
  unless needed after capacity is non-zero)
- optional note: `docs/tasks/SM-ROOM-METAL-03-note.md`

## Context
- METAL-02 **RED**: Super Missile SELECT tactic cannot work —
  fixture `max_super_missiles=0`. Assist only refills **unlocked** ammo;
  it never grants capacity. `enemy0_hp` stayed 1800.
- Pin: `room=0xB62B pose=137 x=699 y=187 door_transition=0 enemy0_hp=1800`
- This is a **fixture gate**, not button timing.
- Practice only — not continuous Lower Norfair.

## Do
1. **One change class:** re-bootstrap / replace the Metal entry fixture so
   `max_super_missiles > 0` (and preferably current supers > 0) while keeping
   doorway-natural entry to `0xB62B` from `0xB482`. Prefer existing late-route
   boot anchors that already own supers (see `docs/SOURCE_STATES.md` dev
   route anchors / full loadout boots used by room bootstrap). Do **not**
   forge progression mid-run; fixture capture at boot is allowed for practice.
2. Re-run the existing Super Missile policy (or minimal SELECT-cycle fix only
   if selected_item still wrong after capacity > 0).
3. Residual must report `max_super_missiles` and `enemy0_hp` after the run.
4. If bootstrap tooling cannot produce supers capacity without banned RAM
   forge, residual → `PLANNER-GATE` with the tooling gap (park Metal).

## Acceptance
- [ ] Fixture shows `max_super_missiles > 0` on teleport **or** explicit
      blocked residual with tooling gap
- [ ] Isolate green **or** residual with pin + next card
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_b62b_from_b482_to_b5d5
uv run python super_metroid/scripts/room/run_problem.py run room_b62b_from_b482_to_b5d5
```
