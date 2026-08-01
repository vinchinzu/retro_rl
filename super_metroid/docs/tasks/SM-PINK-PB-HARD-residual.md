## Residual — SM-PINK-PB-HARD

### Result
PARTIAL

### Files changed
- `routes/kpdr/pink_pb_maze.py` — widened the bounded PB-pocket walk fallback after a morph-bomb roll stalls.
- `docs/tasks/SM-PINK-PB-HARD-residual.md` — records verification and the parked pure-geometry residual.

### Verify paste
`uv run pytest super_metroid/tests/test_post_spore_controller.py -q` (exit 0)

```text
.....                                                                    [100%]
5 passed in 0.61s
```

`uv run python super_metroid/scripts/probe/post_spore_pb.py --to pb-mid-maze --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_pb_door_entered.state --save-fail super_metroid/custom_integrations/SuperMetroid-Snes/scratch/sm_pink_pb_hard_fail.state` (exit 1)

```text
success=false
error=pink_pb_mid_maze: no pure path yet (start=(408,398) -> x=411 y=457 pose=29); deep-pit trap y~=457; mid solid at band
final room=0x9E11 x=411 y=457 pose=29 door_transition=0
```

`uv run python super_metroid/scripts/probe/post_spore_pb.py --to pb-collect --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/dev_b1_pb_from_leftzone.state --save-fail super_metroid/custom_integrations/SuperMetroid-Snes/scratch/sm_pink_pb_hard_collect_fail.state` (exit 0)

```text
developmentOnly=true success=true room=0x9E11 x=116 y=376 pose=31 maxPowerBombs=5
```

### Acceptance
- [x] Code change + residual, pass.
- [x] Tests that import module still pass, pass (`5 passed`).
- [x] Explicit parked / not continuous, pass.

### Residual risks
- The full Pink PB mid-maze remains parked and is not continuous KPDR evidence.
- Pure-green status still requires a successful probe from a named continuous-like source state.

### Next action (required)
- **Next card ID:** SM-PINK-PB-SRC
- **One change:** Capture or catalog a continuous-like PB-pocket source that exercises the widened collect fallback.
- **Source state:** needs capture: SM-PINK-PB-SRC

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence; Pink PB remains parked.

### Probe pin (if pure/geometry)
room=0x9E11 pose=29 x=411 y=457 door_transition=0 (mid-maze probe; deep-pit residual)
