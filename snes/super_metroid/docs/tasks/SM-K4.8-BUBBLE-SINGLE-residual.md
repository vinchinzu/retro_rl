# Residual — SM-K4.8 / rr-dbu.1 (Bubble → Single natural-entry re-verify)

## Result

GREEN

## Files changed

- `routes/kpdr/k4_wave.py` — fix module-level `_DOOR_X` collision: Bubble sill
  was overwritten by Double Chamber `_DOOR_X = 920` after K4.9/K4.10 landings.
  Renamed to `_BSC_DOOR_X/_BSC_DOOR_Y` (bubble→single) and `_DC_DOOR_X/_DC_DOOR_Y_MAX`.

## Verify paste

```bash
uv run pytest snes/super_metroid/tests/test_k4_norfair_scaffold.py \
  snes/super_metroid/tests/test_k4_speed_branches.py -q
# 41 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-single-chamber \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speed_return_to_bubble_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_single_chamber_pure.state \
  --expect-room 0xACB3 \
  --no-red-diag
# success=true room=0xAD5E frames=421 xy=(39,133) (×2 match)

# successor still green after refresh:
uv run python snes/super_metroid/scripts/probe/kpdr.py pure single-to-double-chamber \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_single_chamber_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --no-red-diag
# success=true room=0xADAD frames=700 xy=(39,139)
```

## Acceptance

- [x] Pure bubble-to-single GREEN dual from `post_speed_return_to_bubble_pure`
- [x] One-knob root cause: import-time constant shadow (not geometry regression)
- [x] Residual PROCESS schema
- [ ] Graph edge / continuous tip (planner)

## Geometry (unchanged recipe)

| Fact | Value |
|------|------:|
| Entry | Bubble `0xACB3` post Speed return ~(472,115) |
| Drop band | x∈[370,400] |
| Floor sill | middle-right blue ~(492,395) — `_BSC_DOOR_X=470` |
| Exit | Single `0xAD5E` left shaft ~(39,133) |

## Residual risks

- Shared module constants across hops are fragile — prefer segment prefixes
  (`_BSC_`, `_SC_`, `_DC_`) for any new knobs in `k4_wave.py`
- Smoke RED before fix was pose=164 xy=(475,395) frames=3558 (sill without shot
  phase: `x < 920` always reapproach)

## Next action (required)

- **Next card ID:** rr-dbu.10 (gate open) | already in progress
- **One change:** none for Bubble→Single
- **Source state:** `scratch/post_bubble_to_single_chamber_pure.state` refreshed

## Non-claims

- Did not STATUS-promote / continuous compose / forge progression RAM
- Did not claim Double Chamber gate / Wave pure

## Probe pin (GREEN)

room=0xAD5E pose=81 x=39 y=133 door_transition=0 frames=421 last_pin=post_speed_return_to_bubble_pure dual match
