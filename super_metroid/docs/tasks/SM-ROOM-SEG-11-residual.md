## Residual — SM-ROOM-SEG-11

### Result
BLOCKED

### Files changed
- `docs/tasks/SM-ROOM-SEG-11-residual.md` — records the blocked Waterway dual-track practice result and required source capture.

The existing policy and doorway fixture were present before this session. Exploratory policy edits were reverted after confirming that the entry fixture lacks the capability needed to cross Waterway's speed-block corridor.

### Verify paste
Repository-relative paths are used below.

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_a0d2_from_9d19_to_9d19
exit=0
problemId=room_a0d2_from_9d19_to_9d19
statePath=custom_integrations/SuperMetroid-Snes/room_a0d2_from_9d19.state
room=0xA0D2 game_state=8 phase=ordinary_gameplay
x=960 y=121 pose=2 door_transition=0
equipped_items=0x1004 collected_items=0x1004
morph_ball=true bombs=true

$ uv run python super_metroid/scripts/room/run_problem.py run room_a0d2_from_9d19_to_9d19
exit=1
success=false
failure=policy ended in 0xA0D2; expected 0x9D19
startRoom=0xA0D2 targetRoom=0x9D19
crossingFrame=null settledFrame=null totalFrames=664
finalRoom=0xA0D2 finalPose=9 finalX=1030 finalY=171 door_transition=0
startCollectedItems=0x1004 finalCollectedItems=0x1004
progression_writes=0 capacity_writes=0 deaths=0
policyStatus=generated_unverified
objectiveVerification.status=not_reached
```

Bootstrap and scaffold were skipped because the problem-specific doorway state and policy already existed. Exploratory probes used only temporary `/tmp` screenshots and were not retained as repository artifacts.

### Acceptance
- [ ] Isolated run **GREEN + promote** — failed; the final isolated run remained RED and promotion was not attempted.
- [x] Only own-files touched — the existing Waterway policy and fixture were left unchanged; only this problem's residual was added.
- [x] Dual-track non-claim — this is isolated room practice only and is not continuous evidence.
- [x] Next card ID + one change filled.

### Residual risks
- The fixture starts with `collected_items=0x1004` (Morph + Bombs) and lacks the Speed Booster item bit (`0x2000`); the room's speed-block corridor cannot be crossed by the tested policy on this source.
- The current policy remains `generated_unverified`; it was not practice-promoted.
- Queue refresh, continuous composition, STATUS promotion, and natural-entry claims remain out of scope.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-11-R1
- **One change:** Capture a natural Waterway entry fixture with Speed Booster already collected, then rerun the existing policy without forging progression state.
- **Source state:** needs capture: SM-ROOM-SEG-11-SRC

### Non-claims
- Did not STATUS-promote.
- Did not practice-promote the policy.
- Did not forge progression, capacity, door, event, boss-bit, or item RAM.
- Did not claim continuous green; this is dual-track isolated practice only.

### Probe pin (room-practice failure)
room=0xA0D2 pose=9 x=1030 y=171 door_transition=0
frames=664 dwell=N/A last_pin=room=0xA0D2 pose=9 x=1030 y=171 door_transition=0
