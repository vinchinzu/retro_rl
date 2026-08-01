## Residual - SM-K4-R-02F

### Result
RED

### Files changed
- `routes/kpdr/kraid_return.py` - changed only the Kihunter vertical launch cadence and added best-min-Y timeout diagnostics.
- `docs/tasks/SM-K4-R-02F-residual.md` - recorded the failed pure probe and final in-room pin.

### Verify paste
`uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state`

Exit code: 1.

```text
error: kihunter_to_zeela_return: upper tunnel climb timed out: ...; best_min_y=371
roomIdHex: 0xA4DA
samusX: 471
samusY: 395
pose: 1
frame: 336
controllerOnly: true
developmentOnly: false
```

No Baby transition occurred, and no output state was produced because the
probe did not reach ordinary Zeela `0xA471`.

`uv run pytest super_metroid/tests/test_controller_common.py -q`

Exit code: 0:

```text
..............                                                           [100%]
14 passed in 0.19s
```

### Acceptance
- [ ] Pure green into ordinary `0xA471` - RED; climb timed out in source room `0xA4DA`.
- [x] Fail loud on `0xA521` - Baby guard remains active; this run stayed in `0xA4DA`.
- [x] `uv run pytest super_metroid/tests/test_controller_common.py -q` green - 14 passed.

### Residual risks
- The cadence-only variants `((12, 8), (16, 8), (8, 8))` still do not clear the upper tunnel; best observed `min_y` was 371, above the required `<280` threshold.
- The Zeela window `96..160` and down-door behavior remain unvalidated from the natural source.
- No pure-green, continuous evidence, graph promotion, or STATUS promotion exists.

### Next action (required)
- **Next card ID:** PLANNER-GATE
- **One change:** Choose one new vertical-launch cadence or geometry investigation after reviewing the `best_min_y=371` pin.
- **Source state:** `scratch/post_baby_to_kihunter_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin
room=0xA4DA pose=1 x=471 y=395 door_transition=0 frame=336; best_min_y=371
