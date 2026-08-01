## Residual — SM-ROOM-EASY-02C

### Result
RED

### Files changed
- `policies/room_clears/room_d21c_from_d3b6_to_d08a.json` — Added `UP` to the existing `top_left` final approach span; all other spans are unchanged.
- `docs/tasks/SM-ROOM-EASY-02C-note.md` — Records the isolated wrong-exit residual.

### Verify paste
- `uv run python scripts/room/run_problem.py run room_d21c_from_d3b6_to_d08a`
  - Exit 0 as a probe/report command; report result is `success: false`.
  - Policy still ended in `0xCF80`; expected `0xD08A`.
  - `actionFrames.top_left=80`; `progression_writes=0`; `capacity_writes=0`.

### Acceptance
- [x] Isolate green or residual — residual isolated; the one-span path-class change remains red.
- [x] Dual-track non-claim — development-only room practice; no continuous or STATUS claim.
- [x] If still red, residual says whether end room is still `0xCF80` — it is still `0xCF80`.

### Residual risks
- Adding `UP` to the final `top_left` span did not select the top-left doorway; the run still reaches the top-right doorway at `x=984`.
- The policy remains `generated_unverified` and was not promoted.

### Next action (required)
- **Next card ID:** SM-ROOM-EASY-02D
- **One change:** Replace the single `top_left` span with an explicit stop-right-before-top path, keeping all preceding spans unchanged.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_d21c_from_d3b6.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin
- Start: room=`0xD21C`, pose=`2`, x=`192`, y=`377`, door_transition=`0`.
- Failure: room=`0xCF80`, pose=`82`, x=`984`, y=`118`, door_transition=`1`.
