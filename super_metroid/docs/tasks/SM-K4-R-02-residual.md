## Residual — SM-K4-R-02

### Result
RED

### Files changed
- `routes/kpdr/kraid_return.py` — replaced the naive DOWN hold with a bounded shot-block / Hi-Jump climb and upper-door attempt.
- `docs/tasks/SM-K4-R-02-residual.md` — recorded the bounded-probe residual and final pin.

### Verify paste
`uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state`

Exit code: 1. Final third-strategy pin:
```text
success: false
error: kihunter_to_zeela_return: upper traverse crossed wrong door
roomIdHex: 0xA521
samusX: 65522
samusY: 116
pose: 105
frame: 155
door_transition: 1
```

`uv run pytest super_metroid/tests/test_controller_common.py -q`

Exit code: 0.
```text
.............                                                            [100%]
13 passed in 0.23s
```

### Acceptance
- [ ] Pure probe green from named source → ordinary `0xA471` — failed; the climb reaches the upper route but enters adjacent `0xA521` Baby Kraid door.
- [x] `uv run pytest super_metroid/tests/test_controller_common.py -q` green — 13 passed.
- [x] Residual with PROCESS schema + pin if still red.
- [ ] Optional source capture on green — skipped because pure probe was red.

### Residual risks
- The shot-block / Hi-Jump climb itself clears the alcove, but the upper-floor approach crosses the neighboring Baby Kraid door before reaching Zeela's blue down door.
- No pure-green or continuous evidence exists for this hop.
- No graph verification, STATUS promotion, or continuous route changes were made.

### Next action (required)
- **Next card ID:** SM-K4-R-03
- **One change:** Replace the upper-floor left traverse with a door-specific stop/positioning window that avoids the adjacent Baby Kraid hatch.
- **Source state:** `scratch/post_baby_to_kihunter_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry)
room=0xA521 pose=105 x=65522 y=116 door_transition=1
