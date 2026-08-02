## Residual — SM-ROOM-SEG-12

### Result
GREEN

### Files changed
- `docs/tasks/SM-ROOM-SEG-12-residual.md` — PROCESS residual with isolated GREEN + practice-promote evidence.

Existing owned assets were already present and were not edited this session:
- `policies/room_clears/room_a2f7_from_a322_to_a253.json` (already `verified_development_state`)
- `custom_integrations/SuperMetroid-Snes/room_a2f7_from_a322.state` (doorway-natural entry fixture)

Runner report artifact (expected, gitignored):
- `recordings/room_clears/room_a2f7_from_a322_to_a253.json`

### Verify paste
Commands run from repository root.

```text
$ uv run python super_metroid/scripts/room/run_problem.py teleport room_a2f7_from_a322_to_a253
exit=0
problemId=room_a2f7_from_a322_to_a253
statePath=custom_integrations/SuperMetroid-Snes/room_a2f7_from_a322.state
stateSha256=ab28b15c68d7b2a85b19b0bf5a5f8b50c4ac823efb2a8944c1afc6668b1cff58
room=0xA2F7 game_state=8 phase=ordinary_gameplay
x=704 y=121 pose=2 door_transition=0
collected_items=0x1004 morph_ball=true bombs=true

$ uv run python super_metroid/scripts/room/run_problem.py run room_a2f7_from_a322_to_a253
exit=0
success=true failure=null
startRoom=0xA2F7 targetRoom=0xA253
crossingFrame=444 settledFrame=624 totalFrames=624
finalRoom=0xA253 finalPose=10 finalX=216 finalY=139 door_transition=0
objectiveVerification.status=passed
assist.progression_writes=0 capacity_writes=0 deaths=0
policy.status=verified_development_state
policy.sha256=467c653dc90fad913ba054731f0df56729a5ebbedb2956d078687391dc37a823

$ uv run python super_metroid/scripts/room/run_problem.py run room_a2f7_from_a322_to_a253 --promote
exit=0
success=true promoted=true
startRoom=0xA2F7 targetRoom=0xA253
crossingFrame=444 settledFrame=624 totalFrames=624
finalRoom=0xA253 finalPose=10 finalX=216 finalY=139 door_transition=0
objectiveVerification.status=passed
assist.progression_writes=0 capacity_writes=0 deaths=0
policy.status=verified_development_state
```

Bootstrap and scaffold were skipped: problem-specific doorway fixture and policy already existed on disk. No policy step edits were required; the existing leftward jump/run cadence re-verified GREEN and practice-promoted.

### Acceptance
- [x] Isolated run **GREEN + promote** — teleport exit 0; run success=true; promote success=true promoted=true.
- [x] Only own-files touched — residual only; policy/fixture left unchanged (already practice-ready).
- [x] Dual-track non-claim in residual.
- [x] Next card ID + one change filled.

### Residual risks
- Isolated doorway-natural practice green only; not natural predecessor continuous entry.
- Does not establish continuous route readiness, STATUS promotion, or full-run integrity.
- Practice promote remains dual-track only and must not be treated as continuous evidence.
- Queue refresh / continuous compose remain planner-owned.

### Next action (required)
- **Next card ID:** none
- **One change:** none; no further knob is required for this practice problem.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_a2f7_from_a322.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Practice promotion is dual-track only and is not continuous evidence.
- Did not edit STATUS, QUEUE, PROCESS, continuous routes, kpdr, progression, catalog, or sm_rev.
- Did not modify any other room policy or fixture.

### Probe pin (if pure/geometry) — mandatory metrics
room=0xA2F7 pose=2 x=704 y=121 door_transition=0
frames=624 dwell=180 last_pin=room=0xA253 pose=10 x=216 y=139 door_transition=0
cross=444 settle=624 progression_writes=0 capacity_writes=0 deaths=0
