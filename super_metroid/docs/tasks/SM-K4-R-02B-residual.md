## Residual — SM-K4-R-02B

### Result
RED

### Files changed
- `routes/kpdr/kraid_return.py` — kept the shot-block / Hi-Jump climb, bounded the upper-door motion, and added an explicit Baby Kraid failure guard.
- `docs/tasks/SM-K4-R-02B-residual.md` — recorded the three door-window attempts and final pin.

### Verify paste
`uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state`

Exit code: 1. Final pin:
```text
success: false
error: kihunter_to_zeela_return: blue down-door entered Baby Kraid
roomIdHex: 0xA521
samusX: 39
samusY: 116
pose: 105
frame: 287
door_transition: 1
```

`uv run pytest super_metroid/tests/test_controller_common.py -q`

Exit code: 0.
```text
.............                                                            [100%]
13 passed in 0.15s
```

### Acceptance
- [ ] Pure green from named source → ordinary `0xA471` — failed; the final no-traverse strategy enters adjacent `0xA521` Baby Kraid.
- [x] Never claims green if end room is `0xA521` — explicit guard raises on Baby Kraid transition.
- [x] `uv run pytest super_metroid/tests/test_controller_common.py -q` green — 13 passed.
- [x] Residual with PROCESS schema + pin if still red.
- [ ] Optional source capture on green — skipped because pure probe was red.

### Residual risks
- The shot-block / Hi-Jump climb itself clears the alcove, but the current door setup still selects the neighboring Baby Kraid door.
- No pure-green or continuous evidence exists for this hop.
- No graph verification, STATUS promotion, or continuous route changes were made.

### Next action (required)
- **Next card ID:** SM-K4-R-02C
- **One change:** Capture the post-climb Kihunter position and retune only the down-door setup window before beam shots.
- **Source state:** `scratch/post_baby_to_kihunter_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry)
room=0xA521 pose=105 x=39 y=116 door_transition=1
