## Residual — SM-K4-BUBBLE-PURE

### Result

GREEN

### Files changed

- `custom_integrations/SuperMetroid-Snes/scratch/post_business_to_frog_save_pure.state` — controller-only successor captured after the pure Business→Frog clear.
- `docs/tasks/SM-K4-BUBBLE-PURE-residual.md` — records the independently executed pure acceptance.

### Verify paste

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure business-to-frog-save \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_continuous.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_to_frog_save_pure.state \
  --pin-json super_metroid/debug/business_to_frog_save_pure_pin.json
# exit 0
# success=true roomIdHex=0xB167 frames=1190 controllerOnly=true developmentOnly=false
# final pin: room=0xB167 pose=11 x=39 y=139 door_transition=0
```

The captured successor was independently boot-checked after five idle frames:
`room=0xB167 pose=1 x=60 y=139 door_transition=0`, ordinary gameplay.

### Acceptance

- [x] Named continuous-like source `post_business_continuous` loaded in Business `0xA7DE`.
- [x] Controller-only probe reached ordinary Frog Savestation `0xB167`.
- [x] Successor state captured for the immediate K4 hop.
- [x] This pure result was handed to the separate planner stabilization gate.

### Residual risks

- `frog_save_to_speedway` and every later K4 edge remain unverified.
- The separate planner gate subsequently produced two matching integrity-green
  `--to frog` reports; the authoritative successor is now
  `post_frog_continuous`, not this pure-only capture.

### Next action (required)

- **Next card ID:** SM-K4-SPEEDWAY-PURE
- **One change:** Replace only the bounded Frog Savestation→Speedway scaffold
  with a source-backed pure door-exit controller.
- **Source state:** `scratch/post_frog_continuous.state` (expected
  `0xB167`, reload x=60/y=139/pose=1).

### Non-claims

- Did not itself STATUS-promote or constitute a continuous acceptance.
- Did not use dev placement or a door warp inside the segment.
- Not continuous evidence.

### Probe pin

room=0xB167 pose=11 x=39 y=139 door_transition=0
frames=1190
last_pin=room=0xB167 pose=11 x=39 y=139 door_transition=0
