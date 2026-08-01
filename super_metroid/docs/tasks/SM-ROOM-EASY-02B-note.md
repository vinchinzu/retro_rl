## Residual - SM-ROOM-EASY-02B

### Result
RED

### Files changed
- `policies/room_clears/room_d21c_from_d3b6_to_d08a.json` - Extended the existing `top_left` final approach from 40 to 80 frames; inputs and all other spans were unchanged.
- `docs/tasks/SM-ROOM-EASY-02B-note.md` - Records the isolated wrong-exit residual.

### Verify paste
- `uv run python super_metroid/scripts/room/run_problem.py run room_d21c_from_d3b6_to_d08a`
  - Exit 0 as a probe/report command; report result is `success: false`.
  - Policy still ended in `0xCF80`; expected `0xD08A`.
  - `actionFrames.top_left=80`; `progression_writes=0`; `capacity_writes=0`.

### Acceptance
- [x] Isolate green or residual - residual isolated; the one-knob change remains red.
- [x] Dual-track non-claim - development-only room practice; no continuous or STATUS claim.

### Residual risks
- The final approach still reaches the top-right doorway at `x=984`; changing span length alone did not select the target top-left doorway.
- The policy remains `generated_unverified` and was not promoted.

### Next action (required)
- **Next card ID:** SM-ROOM-ICE-TUT-R3
- **One change:** Replace only the Ice Tutorial `jumpx7` traversal span as specified by its queued card.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_a865_from_a815.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin
- Start: room=`0xD21C`, pose=`2`, x=`192`, y=`377`, door_transition=`0`.
- Failure: room=`0xCF80`, pose=`82`, x=`984`, y=`118`, door_transition=`1`.
