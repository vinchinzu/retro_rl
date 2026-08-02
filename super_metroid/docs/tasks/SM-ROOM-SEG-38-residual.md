## Residual — SM-ROOM-SEG-38

### Result
RED

### Files changed
- `custom_integrations/SuperMetroid-Snes/room_cc27_from_cbd5.state` — ignored, allowed fixture; modified during the first bounded ticket invocation (2026-08-02 04:24:32) to correct doorway entry to the verified x=712 y=139 pose=2 pin.
- `docs/tasks/SM-ROOM-SEG-38-residual.md` — PROCESS residual for the failed isolated practice run; documents pin, non-claims, and next one-knob action.

Fixture changed during the attempt; policy was not modified and remains `generated_unverified`. No source state outside ticket scope was edited.

### Verify paste
Commands were run from the repository root. Paths are repo-relative. Results below are taken from the recorded isolated run report at `super_metroid/recordings/room_clears/room_cc27_from_cbd5_to_cbd5.json` (generatedAt `2026-08-02T09:25:24.670623+00:00`).

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_cc27_from_cbd5_to_cbd5
exit=0
Entry room 0xCC27, pose 2, x=712, y=139, door_transition=0
state sha256 3ff3adaa0735e29be35557a1166231b72d3a2f48672de9d68e6665406dc577f1
statePath=custom_integrations/SuperMetroid-Snes/room_cc27_from_cbd5.state

$ uv run python super_metroid/scripts/room/run_problem.py run room_cc27_from_cbd5_to_cbd5
exit=1
Result RED: policy ended in 0xCC27; expected 0xCBD5
totalFrames=844
final pin room=0xCC27, pose=137, x=691, y=411, door_transition=0
objective status=not_reached (collect_and_return)
policy status=generated_unverified
assist: progression_writes=0 capacity_writes=0 deaths=0
assist missiles: restored=1 writes=1 (resource-only; unlimited ammo assist)
```

Promote was not run because the isolated run was RED. Unlimited-ammo assist wrote one missile restoration (resource-only); no progression or capacity writes occurred. That assist is not a green claim.

### Acceptance
- [ ] Isolated run GREEN + promote (false — isolated run RED; no promote)
- [x] Only own-files touched (fixture changed during the attempt; policy unchanged; residual closeout; no source state outside ticket scope)
- [x] Dual-track non-claim recorded below; this is not continuous evidence
- [x] Next card ID and one change filled below

### Residual risks
- The policy remains `generated_unverified` and is not practice-promoted.
- The isolated run ends still in Wrecked Ship Energy Tank Room `0xCC27` (x=691, y=411, pose=137) after 844 frames; same-door return to expected `0xCBD5` was not achieved and `collect_and_return` is `not_reached`.
- Leftward gap-crossing jump cadence from the doorway-natural fixture does not clear the segment geometry on this attempt.
- This result does not establish pure-green, continuous route readiness, STATUS, natural-entry continuity, or full-run integrity.
- Unlimited-ammo resource assist (one missile restore) does not justify any green or capacity claim.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-38-R1
- **One change:** Adjust the leftward gap-crossing jump cadence from the corrected doorway-natural fixture.
- **Source state:** `super_metroid/custom_integrations/SuperMetroid-Snes/room_cc27_from_cbd5.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; dual-track room practice only.
- Did not practice-promote the policy.
- Unlimited-ammo assist (one missile restoration) is resource-only and is not a green claim; no progression/capacity writes occurred; deaths=0.
- Did not edit policy; fixture (ticket-scope source state) was corrected during the attempt; no source state outside ticket scope; did not edit STATUS, QUEUE, PROCESS, continuous routes, kpdr, progression, catalog, or sm_rev.

### Probe pin (if pure/geometry) — mandatory metrics
room=0xCC27 pose=137 x=691 y=411 door_transition=0
frames=844 dwell=N/A last_pin=room=0xCC27 pose=137 x=691 y=411 door_transition=0

Entry pin (teleport): room=0xCC27 pose=2 x=712 y=139 door_transition=0
