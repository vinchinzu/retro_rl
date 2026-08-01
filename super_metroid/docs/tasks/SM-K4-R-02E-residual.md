## Residual - SM-K4-R-02E

### Result
RED

### Files changed
- `routes/kpdr/kraid_return.py` - changed only the Kihunter lower-alcove launch setup and bounded right-cap climb variants.
- `docs/tasks/SM-K4-R-02E-residual.md` - recorded the failed pure probe and in-Kihunter pin.

### Verify paste
`uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state`

Exit code: 1. The controller stayed in ordinary Kihunter through all three
launch caps and timed out before the upper-band predicate:
```text
error: kihunter_to_zeela_return: upper tunnel climb timed out
roomIdHex: 0xA4DA
samusX: 470
samusY: 395
pose: 1
frame: 336
door_transition: 0
```

Best observed `min_y`: not emitted by the probe; final observed y was 395.
No Baby transition occurred.

`uv run pytest super_metroid/tests/test_controller_common.py -q`

Exit code: 0:
```text
.............                                                            [100%]
13 passed in 0.15s
```

### Acceptance
- [ ] Pure green into ordinary `0xA471` - RED; the climb timed out in source room `0xA4DA` at x=470, y=395.
- [x] Fail loud on `0xA521` - the Baby guard remains in the climb and upper-drop paths; this run did not enter Baby.
- [x] `uv run pytest super_metroid/tests/test_controller_common.py -q` green - 13 passed.

### Residual risks
- The shorter launch setup and right-cap variants still do not clear the shot-block climb from the named natural source state.
- The recon-derived Zeela window `96..160` and drop behavior remain unvalidated by a natural pure run.
- No pure-green, continuous evidence, graph promotion, or STATUS promotion exists.

### Next action (required)
- **Next card ID:** SM-K4-R-02F (card written; dispatch after / beside climb recon)
- **One change:** Change only the lower-alcove vertical launch cadence while retaining the x-band setup and Baby guard.
- **Source state:** `scratch/post_baby_to_kihunter_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin
room=0xA4DA pose=1 x=470 y=395 door_transition=0 frame=336
