## Residual — SM-ROOM-SEG-08

### Result
PARTIAL

### Files changed
- `policies/room_clears/room_ce40_from_c98e_to_93fe.json` — scaffolded the room policy and added leftward traversal with `X` plus an item-fanfare wait.
- `custom_integrations/SuperMetroid-Snes/room_ce40_from_c98e.state` — doorway-natural `0xCE40` practice fixture bootstrapped through entry door `0xA1A4`.
- `custom_integrations/SuperMetroid-Snes/room_ce40_from_c98e.provenance.json` — records the doorway fixture contract and `developmentOnly` status.
- `docs/tasks/SM-ROOM-SEG-08-residual.md` — records the blocked practice result and next action.

### Verify paste
```text
uv run python super_metroid/scripts/room/run_problem.py bootstrap room_ce40_from_c98e_to_93fe
exit 0
status=bootstrapped room=0xCE40 x=192 y=121 pose=2 door=0xA1A4

uv run python super_metroid/scripts/room/run_problem.py scaffold room_ce40_from_c98e_to_93fe
exit 0
status=generated_unverified orientation=left sameDoorReturn=false

uv run python super_metroid/scripts/room/run_problem.py teleport room_ce40_from_c98e_to_93fe
exit 0
room=0xCE40 game_state=8 phase=ordinary_gameplay x=192 y=121 pose=2 door_transition=0

uv run python super_metroid/scripts/room/run_problem.py run room_ce40_from_c98e_to_93fe
exit 1
failure=room objective incomplete: collected_items did not change
crossingFrame=131 settledFrame=348 finalRoom=0x93FE finalPose=12 finalX=1496 finalY=907 door_transition=0
progression_writes=0 capacity_writes=0 deaths=0
```

### Acceptance
- [ ] Isolated run **GREEN + promote** — failed: the run reached and settled in `0x93FE`, but `collected_items` remained `0x1004`.
- [x] Only own-files touched — room policy, this problem's bootstrap fixture/provenance, and this residual; unrelated pre-existing worktree changes were left untouched.
- [x] Dual-track non-claim in residual — this is practice-only and is not continuous evidence.
- [x] Next card ID + one change filled.

### Residual risks
- The bootstrap selected the early `natural_post_spore_spawn` lineage. The Wrecked Ship is still unpowered before Phantoon, so the Gravity Suit PLM is unavailable; movement/shooting changes cannot make this fixture green.
- The policy remains `generated_unverified`; promotion was not attempted after the failed objective check.
- A natural post-Phantoon, pre-Gravity doorway fixture is required before policy geometry can be judged.
- This result does not establish continuous readiness, STATUS readiness, or practice promotion.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-08-R1
- **One change:** Capture a natural post-Phantoon doorway entry fixture for `0xCE40` with power restored, then rerun the existing room policy without forging progression state.
- **Source state:** needs capture: SM-ROOM-SEG-08-SRC

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; this is dual-track isolated practice only.

### Probe pin (if pure/geometry)
room=0x93FE pose=12 x=1496 y=907 door_transition=0
