## Residual — SM-K4-SPEEDWAY-PURE

### Result
GREEN

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — replaced the flat right-door hold with two bounded Hi-Jump pulses around the central save-tube obstruction.
- `super_metroid/docs/tasks/SM-K4-FROG-SPEEDWAY-PURE-residual.md` — recorded the pure-probe result and routing.

### Verify paste

```text
uv run python super_metroid/scripts/probe/kpdr.py pure frog-save-to-speedway --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_continuous.state --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state --pin-json super_metroid/debug/frog_save_to_speedway_pure_pin.json
exit 0
success=true roomIdHex=0xB106 samusX=39 samusY=139 pose=11 doorTransition=0 frames=295 controllerOnly=true developmentOnly=false sourceId=post_frog_continuous
statePath=super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
5 passed in 0.16s
```

### Acceptance

- [x] Source fingerprint loads at `0xB167`.
- [x] Pure controller reaches ordinary `0xB106` without placement or warp.
- [x] Successor source captured at `custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state`.
- [x] Focused unit test is green.
- [x] Residual records the required next card and does not make a continuous or STATUS claim.

### Residual risks

- The controller is pure-green only; no continuous re-record, graph promotion, or STATUS promotion was performed.
- The captured successor still needs source fingerprint registration before Speedway→Farm work starts.

### Next action (required)

- **Next card ID:** `SM-K4-SPEEDWAY-SRC`
- **One change:** Fingerprint-register the captured ordinary Speedway successor source.
- **Source state:** `custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state`

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry) — mandatory metrics

```text
room=0xB106 pose=11 x=39 y=139 door_transition=0
frames=295 dwell=not reported last_pin=room=0xB106 pose=11 x=39 y=139 door_transition=0
```
