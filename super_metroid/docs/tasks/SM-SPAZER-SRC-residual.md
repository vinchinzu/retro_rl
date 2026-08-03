## Residual — SM-SPAZER-SRC

### Result
GREEN

### Files changed
- `docs/SOURCE_STATES.md` — new row: `post_below_spazer_for_spazer_pure`
- `custom_integrations/SuperMetroid-Snes/scratch/post_below_spazer_for_spazer_pure.state` — captured binary state (gitignored)
- `docs/tasks/SM-SPAZER-SRC-residual.md` — this residual

### Verify paste
```
$ rg -n "spazer|0xA408|below_spazer_for_spazer" super_metroid/docs/SOURCE_STATES.md
51:| post_below_spazer_for_spazer_pure | scratch/post_below_spazer_for_spazer_pure.state | 0xA408 Below Spazer ...

$ uv run python super_metroid/scripts/probe/kpdr.py pure bat-to-below \
  --source .../continuous_like_bat.state \
  --output .../post_below_spazer_for_spazer_pure.state
{"success":true,"roomIdHex":"0xA408","frame":668,...}

$ uv run python -c "..."
room_id: 0xA408, collected_beams: 0x0000, collected_items: 0x1004
```

### Acceptance
- [x] State path documented in `SOURCE_STATES.md`
- [x] Room `0xA408`, Spazer not collected (beams 0x0000)
- [x] Residual points to `SM-SPAZER-PURE`

### Residual risks
- Pure continuous `--to below_spazer` tip has not been re-run to produce this state (used pure handoff from `continuous_like_bat` chain). The full continuous run includes prior rooms' persistent state effects. If pure `below-spazer-to-spazer` behaves differently from a continuous-like bat handoff vs a full-continuous predecessor, capture a fresh state from `continuous.py --to below_spazer --state-output ...`.
- No room-timing or benchmark run on the bat-to-below hop this session (668f matches STATUS `Bat dwell (entry→exit split) 668`).

### Next action (required)
- **Next card ID:** SM-SPAZER-PURE
- **One change:** Implement pure `below-spazer-to-spazer` controller (enter 0xA447, collect Spazer pedestal, return to 0xA408 with beam bit set)
- **Source state:** `scratch/post_below_spazer_for_spazer_pure.state`

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence (dual-track source capture only)

### Probe pin
room=0xA408 pose=9 x=39 y=395 door_transition=0 frames=668 dwell=668 last_pin=room 0xA408 ordinary phase
