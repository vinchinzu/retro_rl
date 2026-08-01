# TASK SM-ROOM-ICE-TUT-R1: Ice Tutorial left-exit residual

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local: `room_a865_from_a815_to_a8b9`
- optional note

## Context
- SM-ROOM-ICE-TUT PARTIAL: teleport ok; need left exit trigger for `0xA865`.
- One traversal/policy knob only. Not continuous Ice path.

## Do
1. One-knob fix for left exit.
2. Promote only if green isolate.
3. Residual → future pure `SM-K4-ICE-PURE` after reverse→Business (planner).

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_a865_from_a815_to_a8b9
```
