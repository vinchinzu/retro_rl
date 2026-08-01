# Residual — SM-ROOM-ICE-TUT-PARK

## Result
PARTIAL

## Files changed
- `docs/tasks/SM-ROOM-ICE-TUT-PARK-note.md` — records the parked Ice Tutorial residual and its probe pins.
- `docs/tasks/QUEUE.md` — marks the Ice Tutorial one-knob chain as parked.

## Pin table

| Card | Result | Room | Pose | X | Y | Door transition |
|------|--------|------|------|---:|---:|----------------:|
| SM-ROOM-ICE-TUT-R1 | PARTIAL | `0xA865` | `138` | `277` | `139` | `0` |
| SM-ROOM-ICE-TUT-R2 | PARTIAL | `0xA865` | `138` | `277` | `139` | `0` |
| SM-ROOM-ICE-TUT-R3 | PARTIAL | `0xA865` | `138` | `277` | `139` | `0` |

## Reason for parking

The ICE-TUT R1→R3 one-knob span swaps (`landx7`, then `jumpx7`) all remain in
the same pose-138 class. They do not produce a left-door transition, so more
R4-style span tuning would be repetition rather than a meaningful next knob.
Park this line until a larger policy rewrite changes the maneuver class.

## Acceptance
- [x] Park note filed.
- [x] QUEUE Ice row says parked.
- [x] Dual-track non-claim: this is ROOM_WORK_QUEUE practice evidence only,
  not KPDR continuous-spine or STATUS evidence.

## Residual risks
- The Ice Tutorial left exit remains unresolved and is not pure-green.
- No policy was changed by this card.
- Continuous and STATUS work remain unaffected and are not claimed here.

## Next action (required)
- **Next card ID:** `SM-ROOM-ICE-TUT-REWRITE` (planner placeholder only; do not create yet)
- **One change:** Replace the pose-138 one-knob span chain with a larger policy rewrite.
- **Source state:** needs capture: `SM-ROOM-ICE-TUT-REWRITE-SRC`

## Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

## Probe pin
`room=0xA865 pose=138 x=277 y=139 door_transition=0`
