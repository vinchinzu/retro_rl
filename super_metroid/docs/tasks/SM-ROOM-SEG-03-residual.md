## Residual - SM-ROOM-SEG-03

### Result
GREEN

### Files changed
- policies/room_clears/room_a051_from_a011_to_a011.json - Added the item-recovery sequence and promoted the verified isolated policy.
- custom_integrations/SuperMetroid-Snes/room_a051_from_a011.state - Created the doorway-natural Etacoon Super Room entry fixture.
- custom_integrations/SuperMetroid-Snes/room_a051_from_a011.provenance.json - Recorded the fixture's source door, entry contract, and development-only provenance.
- docs/tasks/SM-ROOM-SEG-03-residual.md - Recorded this task result and acceptance evidence.

### Verify paste
Commands were run from the repository root. All exited 0.

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_a051_from_a011_to_a011
state room=0xA051 phase=ordinary_gameplay game_state=8 x=192 y=121 door_transition=0
stateSha256=50d6e691434c32251ad6cc7ec106a72a1d873498cc1e8ba75f27fb08b305dcb9

$ uv run python super_metroid/scripts/room/run_problem.py run room_a051_from_a011_to_a011
success=true target=0xA011 crossingFrame=735 settledFrame=857 totalFrames=857
objective=collect_and_return status=passed
progression_writes=0 capacity_writes=0 deaths=0
final room=0xA011 phase=ordinary_gameplay game_state=8 x=39 y=139

$ uv run python super_metroid/scripts/room/run_problem.py run room_a051_from_a011_to_a011 --promote
success=true promoted=true policyStatus=verified_development_state

$ uv run pytest super_metroid/tests/test_room_graph.py -q
...........                                                              [100%]
11 passed in 0.23s
```

### Acceptance
- [x] Isolated run GREEN and practice promotion completed.
- [x] Only this problem's policy, entry fixture, and task residual were changed; generated recording output is ignored harness output.
- [x] Dual-track non-claim is explicit below.
- [x] Next card ID and one-change decision are filled below.

### Residual risks
- This is an isolated doorway-natural practice result only; it does not establish continuous route readiness.
- The item-recovery timing is verified against this fixture and should be rechecked if the fixture is re-bootstrapped with a different RNG seed or idle schedule.
- Queue board refresh remains planner-owned and was not performed in this card.

### Next action (required)
- **Next card ID:** none
- **One change:** Leave this promoted practice policy unchanged; planner may refresh the room practice queue.
- **Source state:** custom_integrations/SuperMetroid-Snes/room_a051_from_a011.state

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, event, or boss-bit RAM for the green claim; the fixture used the required development door-warp/bootstrap path.
- This is not continuous evidence; it is a dual-track room practice promotion only.

### Probe pin
N/A - no failed probe; isolated run was green.
