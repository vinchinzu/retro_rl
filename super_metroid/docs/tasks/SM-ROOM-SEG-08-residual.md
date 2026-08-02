## Residual — SM-ROOM-SEG-08

### Result
BLOCKED

### Files changed
- `docs/tasks/SM-ROOM-SEG-08-residual.md` — records the blocked dual-track room-practice result and required source capture.

The target policy and doorway fixture had no durable changes from this session. Exploratory policy edits were reverted after the source-state blocker was confirmed.

### Verify paste
```text
uv run python super_metroid/scripts/room/run_problem.py teleport room_ce40_from_c98e_to_93fe
exit 0
problemId=room_ce40_from_c98e_to_93fe
room=0xCE40 game_state=8 phase=ordinary_gameplay
x=192 y=121 pose=2 door_transition=0
selected_item=1 collected_items=4100 (0x1004)
boss_bits=[255, 0, 255, 0, 255, 0, 255, 0]

uv run python super_metroid/scripts/room/run_problem.py run room_ce40_from_c98e_to_93fe
exit 1
failure=room objective incomplete: collected_items did not change
crossingFrame=131 settledFrame=348 totalFrames=348
startRoom=0xCE40 targetRoom=0x93FE
startCollectedItems=0x1004 finalCollectedItems=0x1004
finalRoom=0x93FE finalPose=12 finalX=1496 finalY=907 door_transition=0
progression_writes=0 capacity_writes=0 deaths=0
policyStatus=generated_unverified

python -m json.tool policies/room_clears/room_ce40_from_c98e_to_93fe.json >/dev/null
exit 0
```

### Acceptance
- [ ] Isolated run **GREEN + promote** — failed: the policy reaches and settles in `0x93FE`, but the Gravity Suit item delta remains absent.
- [x] Only own-files touched — the final durable task change is this residual; the target policy and fixture were restored/left unchanged, and unrelated worktree changes were not modified.
- [x] Dual-track non-claim — this is isolated practice only and is not continuous evidence.
- [x] Next card ID + one change filled.

### Residual risks
- The fixture provenance uses `natural_post_spore_spawn.state`; Wrecked Ship is pre-Phantoon/unpowered. Its `boss_bits[3]` byte is `0`, and `collected_items` is `0x1004` without the required Gravity bit `0x0020`, so the Gravity PLM is unavailable.
- Movement, shooting, and item-touch changes cannot prove this room on the current fixture; policy geometry must wait for a valid powered source.
- The policy remains `generated_unverified`; promotion was not attempted after the failed objective check.
- Queue refresh, continuous composition, STATUS promotion, and natural-entry claims remain out of scope.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-08-R1
- **One change:** Capture a controllable natural post-Phantoon, pre-Gravity doorway fixture for `0xCE40` with Wrecked Ship power restored, then rerun the existing policy without forging progression state.
- **Source state:** needs capture: SM-ROOM-SEG-08-SRC

### Non-claims
- Did not STATUS-promote.
- Did not promote the practice policy.
- Did not forge progression, capacity, boss-bit, event, door, or item RAM.
- Did not claim continuous green; this is dual-track isolated practice only.

### Probe pin (if pure/geometry)
room=0x93FE pose=12 x=1496 y=907 door_transition=0
