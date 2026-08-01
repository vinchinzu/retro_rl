## Residual — SM-ROOM-SCAFFOLD-SPAZER

### Result
RED

### Files changed
- `policies/room_clears/room_a447_from_a408_to_a408.json` — generated doorway-natural starter policy.
- `docs/tasks/SM-ROOM-SCAFFOLD-SPAZER-note.md` — records the failed isolated run and next action.

### Verify paste
- `uv run python super_metroid/scripts/room/run_problem.py scaffold room_a447_from_a408_to_a408` — exit 0; generated policy with `status: generated_unverified`, left orientation, and same-door return.
- `uv run python super_metroid/scripts/room/run_problem.py teleport room_a447_from_a408_to_a408` — exit 0; fixture loaded in `0xA447`, `game_state: 8`, `door_transition: 0`, `samus_x: 64`, `samus_y: 121`.
- `uv run python super_metroid/scripts/room/run_problem.py run room_a447_from_a408_to_a408` — exit 0; report `success: false`, failure `policy ended in 0xA447; expected 0xA408`, `crossingFrame: null`, `settledFrame: null`, `totalFrames: 798`.

### Acceptance
- [x] Scaffold command completed and created the policy.
- [x] Teleport command completed and loaded the expected `0xA447` fixture.
- [ ] Isolated run reaches expected `0xA408` exit; starter policy remains unverified.
- [x] Residual uses PROCESS schema with one next action.
- [x] No unrelated file churn was intentionally made.

### Residual risks
- The generated movement does not open or cross the left Spazer door from the doorway-natural fixture.
- This practice result is development-only and is not continuous evidence.
- Policy promotion is blocked until a bounded movement adjustment produces a green isolated run.

### Next action (required)
- **Next card ID:** SM-ROOM-SPAZER-01
- **One change:** Replace the scaffold's single left-door approach/entry timing with one geometry-tuned movement sequence that crosses into `0xA408`.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_a447_from_a408.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry)
room=0xA447 pose=138 x=85 y=187 door_transition=0
