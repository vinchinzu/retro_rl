# Residual — SM-ROOM-EASY-PARK

## Result
PARTIAL

## Files changed
- `docs/tasks/SM-ROOM-EASY-PARK-note.md` — records the parked Crab Hole and Grapple Tutorial residuals and their stable probe pins.
- `docs/tasks/QUEUE.md` — marks the final Crab and Grapple rows as parked.

## Crab Hole pin table

| Card | Result | Room | Pose | X | Y | Door transition |
|------|--------|------|------|---:|---:|----------------:|
| SM-ROOM-EASY-02 | RED | `0xCF80` | `82` | `984` | `118` | `1` |
| SM-ROOM-EASY-02B | RED | `0xCF80` | `82` | `984` | `118` | `1` |
| SM-ROOM-EASY-02C | RED | `0xCF80` | `82` | `984` | `118` | `1` |

## Grapple Tutorial pin table

| Card | Result | Room | Pose | X | Y | Door transition |
|------|--------|------|------|---:|---:|----------------:|
| SM-ROOM-EASY-03-R1 | RED | `0xABD2` | `138` | `21` | `395` | `0` |
| SM-ROOM-EASY-03-R2 | RED | `0xABD2` | `138` | `21` | `395` | `0` |
| SM-ROOM-EASY-03-R3 | RED | `0xABD2` | `138` | `21` | `395` | `0` |

## Reason for parking

The Crab Hole `top_left` span extension plus `UP` bias still selects the wrong
exit `0xCF80`. The Grapple Tutorial `land_shoot` and door-open/entry changes
still stop at the same pose-138 pin. Further one-knob tuning would repeat the
same maneuver classes, so both lines are parked for planner rewrites.

No policy files were changed by this parking card.

## Acceptance

- [x] Park note filed with Crab and Grapple pin tables.
- [x] QUEUE final Crab and Grapple rows say parked.
- [x] Dual-track non-claim: these are room-practice residuals only, not
  continuous-spine or STATUS evidence.

## Residual risks

- Crab Hole still exits to `0xCF80` instead of expected `0xD08A`.
- Grapple Tutorial still does not transition from `0xABD2` to expected `0xAC00`.
- Both policies remain unresolved and unverified; no policy edit belongs in
  this parking card.

## Next action (required)

- **Next card ID:** `PLANNER-GATE` for Grapple; `SM-ROOM-EASY-02-REWRITE`
  planner placeholder for Crab (do not create either card here).
- **One change:** Replace each repeated one-knob maneuver chain with a
  planner-designed policy rewrite, one room at a time.
- **Source state:** needs capture: `SM-ROOM-EASY-PARK-REWRITE-SRC`

## Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence; this remains isolated room practice only.

## Probe pins

- Crab: `room=0xCF80 pose=82 x=984 y=118 door_transition=1`
- Grapple: `room=0xABD2 pose=138 x=21 y=395 door_transition=0`
