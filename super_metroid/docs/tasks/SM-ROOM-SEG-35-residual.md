## Residual — SM-ROOM-SEG-35

### Result
RED

### Files changed
- `docs/tasks/SM-ROOM-SEG-35-residual.md` — PROCESS residual for the failed isolated practice run; documents pin, non-claims, and next one-knob action.
- `docs/tasks/SM-ROOM-SEG-35-R1.md` — bounded successor card owning only this problem's policy + residual, with one named traversal knob.

No policy, fixture, source state, recording, or other repository path was edited by this closeout. A prior bounded attempt already edited `policies/room_clears/room_b2da_from_b3a5_to_b6c1.json` (still `generated_unverified`) and exhausted its turn budget before filing RED evidence; that policy is left as-is.

### Verify paste
Commands were run from the repository root. Paths are repo-relative. Results below are taken from the recorded isolated run report at `super_metroid/recordings/room_clears/room_b2da_from_b3a5_to_b6c1.json` (generatedAt `2026-08-02T10:33:11.438436+00:00`). Closeout did not re-run teleport/run and did not rewrite the report.

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_b2da_from_b3a5_to_b6c1
exit=0
Entry room 0xB2DA, pose 2, x=960, y=121, door_transition=0
statePath=custom_integrations/SuperMetroid-Snes/room_b2da_from_b3a5.state
state sha256 1505665fb901188921fb25ad9fbb1365ee5c1618a33dbc18c731b08e595d790a

$ uv run python super_metroid/scripts/room/run_problem.py run room_b2da_from_b3a5_to_b6c1
exit=1
Result RED: policy ended in 0xB2DA; expected 0xB6C1
totalFrames=1281
final pin room=0xB2DA, pose=138, x=853, y=139, door_transition=0
objective status=not_reached (traverse_to_exit)
policy status=generated_unverified
assist energy: restored=321 writes=321 (resource-only; unlimited energy assist)
assist missiles: restored=1 writes=1 (resource-only; unlimited ammo assist)
assist: progression_writes=0 capacity_writes=0 deaths=0
```

Promote was not run because the isolated run was RED. Unlimited energy/ammo assist writes are resource-only; no progression or capacity writes occurred. Those assists are not a green claim.

### Acceptance
- [ ] Isolated run GREEN + promote (false — isolated run RED; no promote)
- [x] Only own-files touched for this closeout (residual + successor card; policy/fixture/report unchanged here)
- [x] Dual-track non-claim recorded below; this is not continuous evidence
- [x] Next card ID and one change filled below

### Residual risks
- The policy remains `generated_unverified` and is not practice-promoted.
- Target room `0xB6C1` was not reached; the run ends still in Fast Ripper Room `0xB2DA` after 1281 frames (final pin pose=138, x=853, y=139, door_transition=0).
- The repeated grounded-left dash/recover transit (`grounded_left_transit_with_knockback_recover`: LEFT+B dash + LEFT recover, ×16) does not clear the leftward exit corridor under this doorway-natural fixture.
- High energy resource restores (321 writes) show contact damage during the grounded transit; still not a progression or capacity claim.
- This result does not establish pure-green, continuous route readiness, STATUS, natural-entry continuity, or full-run integrity.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-35-R1
- **One change:** Replace the repeated grounded-left dash/recover transit with a jumping traversal cadence from the same doorway-natural fixture.
- **Source state:** `super_metroid/custom_integrations/SuperMetroid-Snes/room_b2da_from_b3a5.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; dual-track room practice only.
- Did not practice-promote the policy.
- Unlimited energy/ammo assist writes (energy=321, missiles=1) are resource-only and are not a green claim; progression_writes=0, capacity_writes=0, deaths=0.
- Did not edit the policy, fixture, recording/report, STATUS, QUEUE, PROCESS, continuous routes, kpdr, progression, catalog, or sm_rev in this closeout.
- Did not claim GREEN.

### Probe pin (if pure/geometry) — mandatory metrics
room=0xB2DA pose=138 x=853 y=139 door_transition=0
frames=1281 dwell=N/A last_pin=room=0xB2DA pose=138 x=853 y=139 door_transition=0

Entry pin (teleport): room=0xB2DA pose=2 x=960 y=121 door_transition=0
assist energy_writes=321 missile_writes=1 progression_writes=0 capacity_writes=0 deaths=0
