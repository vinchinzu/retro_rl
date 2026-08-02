## Residual — SM-ROOM-SEG-39

### Result
RED

### Files changed
- `docs/tasks/SM-ROOM-SEG-39-residual.md` — PROCESS residual for the failed isolated practice run; documents pin, non-claims, and next one-knob action.

No policy, fixture, source state, or other repository path was edited by this closeout. The existing policy remains `generated_unverified`. No policy changes were made.

### Verify paste
Commands were run from the repository root. Paths are repo-relative.

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_cdf1_from_caf6_to_caf6
exit=0
Entry room 0xCDF1, pose 1, x=64, y=121, door_transition=0
state sha256 1cf0e98db774dcdaa50da6d08ae23df4a2879972e711cd952b0c7e770dd0703b

$ uv run python super_metroid/scripts/room/run_problem.py run room_cdf1_from_caf6_to_caf6
exit=1
Result RED: policy ended in 0xCDF1; expected 0xCAF6
totalFrames=664
final pin room=0xCDF1, pose=138, x=37, y=139, door_transition=0
objective status=not_reached
policy status=generated_unverified
```

Promote was not run because the isolated run was RED. Unlimited-ammo assist write observed in the run is resource-only; no progression or capacity writes occurred. That assist is not a green claim.

### Acceptance
- [ ] Isolated run GREEN + promote (false — isolated run RED; no promote)
- [x] Only own-files touched (this residual only; policy and source state unchanged)
- [x] Dual-track non-claim recorded below; this is not continuous evidence
- [x] Next card ID and one change filled below

### Residual risks
- The policy remains `generated_unverified` and is not practice-promoted.
- The current `deeper_into_room` 70-frame RIGHT+A+B starter does not reach the objective corridor; the run ends still in `0xCDF1` short of expected `0xCAF6`.
- This result does not establish pure-green, continuous route readiness, STATUS, natural-entry continuity, or full-run integrity.
- Unlimited-ammo resource assist does not justify any green or capacity claim.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-39-R1
- **One change:** Replace the current `deeper_into_room` 70-frame RIGHT+A+B starter with a separately verified traversal sequence that reaches the objective corridor from the same doorway-natural source state.
- **Source state:** `super_metroid/custom_integrations/SuperMetroid-Snes/room_cdf1_from_caf6.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; dual-track room practice only.
- Did not practice-promote the policy.
- Unlimited-ammo assist write is resource-only and is not a green claim; no progression/capacity writes occurred.
- Did not edit policy, source state, STATUS, QUEUE, PROCESS, continuous routes, kpdr, progression, catalog, or sm_rev.

### Probe pin (if pure/geometry) — mandatory metrics
room=0xCDF1 pose=138 x=37 y=139 door_transition=0
frames=664 dwell=N/A last_pin=room=0xCDF1 pose=138 x=37 y=139 door_transition=0

Entry pin (teleport): room=0xCDF1 pose=1 x=64 y=121 door_transition=0
