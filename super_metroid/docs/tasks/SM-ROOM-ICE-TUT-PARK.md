# TASK SM-ROOM-ICE-TUT-PARK: Park Ice Tutorial one-knob chain (docs only)

## Recipe step
docs

## Model
Flash

## Wave type
implement

## Own files only
- `docs/tasks/SM-ROOM-ICE-TUT-PARK-note.md` (create)
- `docs/tasks/QUEUE.md` (Ice row → parked only; do not rewrite whole board)

## Context
- ICE-TUT R1→R3: all PARTIAL, same pin
  `room=0xA865 pose=138 x=277 y=139 door_transition=0`
- One-knob span swaps (`landx7`, `jumpx7`) did not leave pose-138.
- Planner: **park** until a larger policy rewrite card (not R4 spam).

## Do
1. Write park note with pin table + reason (pose-138 class stuck).
2. QUEUE: mark ICE-TUT line parked; next card = planner rewrite later
   (`SM-ROOM-ICE-TUT-REWRITE` placeholder only in note — do not create it).
3. No policy edits. No continuous / STATUS.

## Acceptance
- [ ] Park note filed
- [ ] QUEUE Ice row says parked
- [ ] Dual-track non-claim

## Verify
```bash
test -f super_metroid/docs/tasks/SM-ROOM-ICE-TUT-PARK-note.md
rg -n "ICE-TUT|park" super_metroid/docs/tasks/QUEUE.md | head -20
```
