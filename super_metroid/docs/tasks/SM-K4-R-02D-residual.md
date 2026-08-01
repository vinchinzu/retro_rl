## Residual - SM-K4-R-02D

### Result
RED

### Files changed
- `routes/kpdr/kraid_return.py` - gated climb success on the Kihunter room and no door transition, bounded three left/vertical launch strategies, and used the recon Zeela x window.
- `docs/tasks/SM-K4-R-02D-residual.md` - recorded the failed pure probe and source-room pin.

### Verify paste
`uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state`

Exit code: 1. The controller stayed in Kihunter through all three bounded
strategies and timed out before reaching the upper band:
```text
error: kihunter_to_zeela_return: upper tunnel climb timed out
roomIdHex: 0xA4DA
samusX: 357
samusY: 395
pose: 2
frame: 340
door_transition: 0
```

`uv run pytest super_metroid/tests/test_controller_common.py -q`

Exit code: 0:
```text
.............                                                            [100%]
13 passed in 0.16s
```

### Acceptance
- [ ] Pure green into ordinary `0xA471` - RED; the climb timed out in source room `0xA4DA` before an upper-land pin.
- [x] Fail loud on `0xA521` - Baby Kraid is checked and raises immediately during the climb and upper drop path.
- [x] `uv run pytest super_metroid/tests/test_controller_common.py -q` green - 13 passed.

### Residual risks
- The three bounded launches do not clear the shot-block climb from the named natural source state.
- The recon-derived `96..160` Zeela window and drop behavior were not emulator-reached in this run.
- No pure-green, continuous evidence, graph promotion, or STATUS promotion exists.

### Next action (required)
- **Next card ID:** SM-K4-R-02E
- **One change:** Lower-alcove launch only — setup band x≈360–420 with RIGHT-capped climb (never x≥480); do not retune Zeela window `96..160`.
- **Source state:** `scratch/post_baby_to_kihunter_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin
room=0xA4DA pose=2 x=357 y=395 door_transition=0 frame=340
