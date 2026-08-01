## Residual — SM-K4-R-CLIMB-REDESIGN

### Result
GREEN

### Files changed
- `routes/kpdr/kraid_return.py` — redesign `play_kihunter_to_zeela_return`:
  wall-plant → mid ledge → morph bomb-jump through x≈376 hole → morph-roll
  to Zeela window → down door
- `docs/tasks/SM-K4-R-CLIMB-REDESIGN.md` — planner redesign card
- `docs/SOURCE_STATES.md` — capture `post_kihunter_to_zeela` + green note
- `docs/tasks/QUEUE.md` — Wave 8c/d close + Wave 9 board

### Verify paste
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
```

Exit 0, multi-run green:

```text
success: true
roomIdHex: 0xA471
samusX: 403
samusY: 362
frame/frames: ~1716
controllerOnly: true
developmentOnly: false
statePath: .../scratch/post_kihunter_to_zeela_return.state
```

`uv run pytest super_metroid/tests/test_controller_common.py -q` → 14 passed.

### Geometry (why one-knob cadence failed)

| Fact | Detail |
|------|--------|
| Floor hard wall | x≈357 — cannot walk under hole |
| Mid ledge | y≈299, x≈367–379 — Hi-Jump + RIGHT apex drift |
| Bomb hole | **x≈376** (forward bomb scan); upper floor y≈171 |
| Cadence class | right-cap spin from baby door → min_y=371 forever |
| Working class | mid ledge + morph bomb-jump through hole |

### Acceptance
- [x] Pure green → ordinary `0xA471`
- [x] Baby fail-loud retained
- [x] controller_common tests green
- [x] Source captured
- [x] No STATUS claim

### Next action (required)
- **Next card ID:** SM-K4-R-03
- **One change:** Pure `zeela-to-warehouse-return` from new source
  `scratch/post_kihunter_to_zeela_return.state` (room `0xA471`)
- **Source state:** `scratch/post_kihunter_to_zeela_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence; graph stays `controller_dev` until planner promotes.
