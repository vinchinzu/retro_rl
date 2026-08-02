## Residual -- SM-ROOM-SEG-22

### Result
RED

### Files changed
- `policies/room_clears/room_da2b_from_d913_to_d9fe.json` -- replaced the generated single traversal span with a bounded repeated jump/release traversal attempt; policy remains unverified.
- `docs/tasks/SM-ROOM-SEG-22-residual.md` -- records the failed isolated replay, exact pin, and next one-knob action.

The doorway fixture and provenance already existed and were not edited. Bootstrap
and scaffold were skipped for that reason. The runner's room report is an
expected ignored artifact under `recordings/room_clears/`.

### Verify paste
```text
$ uv run python scripts/room/run_problem.py teleport room_da2b_from_d913_to_d9fe
exit=0
problemId=room_da2b_from_d913_to_d9fe
statePath=custom_integrations/SuperMetroid-Snes/room_da2b_from_d913.state
state.frame=1 room_id_hex=0xDA2B phase=ordinary_gameplay
state.samus_x=1216 state.samus_y=377 pose=2 door_transition=0

$ uv run python scripts/room/run_problem.py run room_da2b_from_d913_to_d9fe
exit=1
success=false
failure=policy ended in 0xDA2B; expected 0xD9FE
crossingFrame=null settledFrame=null totalFrames=808
finalState.room_id_hex=0xDA2B finalState.samus_x=1061 finalState.samus_y=443
finalState.pose=138 finalState.door_transition=0
policy.status=generated_unverified
assist.progression_writes=0 capacity_writes=0 deaths=0
```

The required promote run was not issued because the isolated replay was RED.
No pytest command was specified by this room-policy card.

### Acceptance
- [x] Isolated run produced an honest RED residual with the mandatory pin; green promote was not possible.
- [x] Only card-owned files were touched by this session; the fixture was not edited and temporary diagnostics were outside the repository.
- [x] Dual-track non-claim is explicit below; this is not continuous evidence.
- [x] Next card ID and one change are filled below.

### Residual risks
- The policy remains `generated_unverified` and is not practice-promoted.
- The repeated jump/release span reaches the first vertical barrier but does not cross into the next room.
- The 20-frame entry settle starts the traversal before the doorway fixture has stabilized on its ledge; a longer settle is indicated by bounded diagnostics, but it was not stacked with another traversal change in this session.
- This result does not establish pure-green, continuous, STATUS, natural-entry, or full-run evidence.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-22-R1
- **One change:** Increase only `entry_settle` from 20 frames to a stable landed-ledge window (candidate: 60 frames), leaving the current jump/release traversal unchanged.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_da2b_from_d913.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss-bit RAM.
- Not continuous evidence; this is dual-track room practice only.
- Did not promote the practice policy.

### Probe pin (isolated practice)
room=0xDA2B pose=138 x=1061 y=443 door_transition=0
frames=808 dwell=N/A last_pin=0xDA2B/pose138/x1061/y443/door_transition0
