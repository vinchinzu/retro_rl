# TASK SM-ROOM-ICE-R1: Ice Beam Room — collect residual after scaffold

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_a890_from_a8b9_to_a8b9.json` only
- optional note: `docs/tasks/SM-ROOM-ICE-R1-note.md`

## Context
- SCAFFOLD-ICE **RED**: crossed to `0xA8B9` but collect objective failed
  (`collected_beams` unchanged). Policy not promoted.
- One knob on item-touch / pedestal approach only.
- Practice only; not continuous Ice tip.

## Do
1. One knob so isolated run collects Ice (or residual with beams pin).
2. Promote only if green isolate.
3. No continuous / STATUS.

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_a890_from_a8b9_to_a8b9
```
