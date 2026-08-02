## Residual — SM-ROOM-SEG-36

### Result
RED

### Files changed
- `docs/tasks/SM-ROOM-SEG-36-residual.md` — PROCESS residual for the failed isolated practice run; documents pin, non-claims, and next one-knob action.
- `docs/tasks/SM-ROOM-SEG-36-R1.md` — bounded successor card owning only this problem's policy + residual, with one named traversal knob.

No policy, fixture, source state, recording, or other repository path was edited by this closeout. Prior bounded attempts left `policies/room_clears/room_b37a_from_b32e_to_b482.json` as scaffold `generated_unverified` (untouched here) and exhausted without filing RED evidence; that policy is left as-is.

### Verify paste
Commands were run from the repository root. Paths are repo-relative. Results below match independent post-run verification and the recorded isolated run report at `super_metroid/recordings/room_clears/room_b37a_from_b32e_to_b482.json` (generatedAt `2026-08-02T11:41:37.199630+00:00`). Closeout did not re-run teleport/run and did not rewrite the report.

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_b37a_from_b32e_to_b482
exit=0
Entry room 0xB37A, pose 1, x=64, y=121, door_transition=0
statePath=custom_integrations/SuperMetroid-Snes/room_b37a_from_b32e.state
state sha256 459da2d8655f7e531d615b7d776123e36acf948442e81266560c1e712c5a7841

$ uv run python super_metroid/scripts/room/run_problem.py run room_b37a_from_b32e_to_b482
exit=1
Result RED: policy ended in 0xB37A; expected 0xB482
totalFrames=443
final pin room=0xB37A, pose=137, x=123, y=219, door_transition=0
objective status=not_reached (traverse_to_exit)
policy status=generated_unverified
assist energy: restored=564 writes=338 (resource-only; unlimited energy assist)
assist missiles: restored=1 writes=1 (resource-only; unlimited ammo assist)
assist: progression_writes=0 capacity_writes=0 deaths=0
```

Promote was not run because the isolated run was RED. Unlimited energy/ammo assist writes are resource-only; no progression or capacity writes occurred. Those assists are dual-track development-only and are not a green claim.

Precise failure: after the scaffold `coarse_exit_approach` (grounded RIGHT ×220), Samus falls/sticks in lava still inside `0xB37A` (pose=137, y=219) instead of staying on the upper platforms toward exit `0xB482`.

### Acceptance
- [ ] Isolated run GREEN + promote (false — isolated run RED; no promote)
- [x] Only own-files touched for this closeout (residual + successor card; policy/fixture/report unchanged here)
- [x] Dual-track non-claim recorded below; this is not continuous evidence
- [x] Next card ID and one change filled below

### Residual risks
- The policy remains `generated_unverified` and is not practice-promoted.
- Target room `0xB482` was not reached; the run ends still in Lower Norfair Farming Room `0xB37A` after 443 frames (final pin pose=137, x=123, y=219, door_transition=0).
- Scaffold coarse grounded RIGHT approach drops Samus into lava / sticky hazard floor rather than holding the upper platform line to the right exit.
- High energy resource restores (338 writes) show lava/contact damage during the approach; still not a progression or capacity claim.
- This result does not establish pure-green, continuous route readiness, STATUS, natural-entry continuity, or full-run integrity.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-36-R1
- **One change:** Replace the coarse grounded RIGHT approach (`coarse_exit_approach`) with an initial jump/right traversal cadence to stay on the upper platforms.
- **Source state:** `super_metroid/custom_integrations/SuperMetroid-Snes/room_b37a_from_b32e.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; dual-track room practice only.
- Did not practice-promote the policy.
- Unlimited energy/ammo assist writes (energy=338, missiles=1) are resource-only, dual-track development-only, and are not a green claim; progression_writes=0, capacity_writes=0, deaths=0.
- Did not edit the policy, fixture, recording/report, STATUS, QUEUE, PROCESS, continuous routes, kpdr, progression, catalog, or sm_rev in this closeout.
- Did not claim GREEN.

### Probe pin (if pure/geometry) — mandatory metrics
room=0xB37A pose=137 x=123 y=219 door_transition=0
frames=443 dwell=N/A last_pin=room=0xB37A pose=137 x=123 y=219 door_transition=0

Entry pin (teleport): room=0xB37A pose=1 x=64 y=121 door_transition=0
assist energy_writes=338 missile_writes=1 progression_writes=0 capacity_writes=0 deaths=0
