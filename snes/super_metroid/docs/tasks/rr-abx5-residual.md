## Residual — rr-abx5 Pure West Tunnel → Below Spazer return (K5 hop 8)

### Result
GREEN — pure one-hop dual GREEN West Tunnel `0xCF54` → Below Spazer `0xA408`
(**272f** ×2 exact). Not continuous. No STATUS change.

### Files changed
- `routes/kpdr/k5/west_to_below.py` — reverse of below floor→west; LEFT run_shoot_exit
- `routes/kpdr/k5/geometry.py` — WEST_TO_BELOW_* / WEST_BELOW_DOOR_X constants
- `routes/kpdr/k5/__init__.py` — export + hop map checkmark
- `routes/kpdr/registry.py` / `__init__.py` / `scripts/probe/kpdr.py` — segment wiring
- `source_states.py` + `docs/SOURCE_STATES.md` — handoff fingerprint
- export: `scratch/post_ice_west_to_below_pure.state` (+ dual)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure west-to-below \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_glass_to_west_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_west_to_below_pure.state
# → GREEN room=0xA408 xy=(472,393) pose=82 frames=272 (×2 exact dual)
```

### Acceptance
- [x] Pure dual green West → Below from post-glass-to-west mid-right pin
- [x] Reverse of below_spazer_floor_to_west (LEFT mirror; Spazer already held)
- [x] Export handoff for Below → Bat reverse
- [ ] Full K5 stack through Alpha PB (parent rr-dbu.8)
- [ ] Continuous / STATUS (planner only)

### Residual risks
1. Below pin ~(472,393) p82 right-floor — next reverse hop LEFT toward Bat door.
2. Fixed settle 260f; optimizable but dual-stable.
3. Parent stack still open through Below→Bat→Red→Alpha PB.

### Next action (required)
- **Next card:** Pure Below Spazer → Bat Room return (K5 hop 9) reverse of bat_to_below_spazer
- **One change:** pure Below `0xA408` → Bat `0xA3DD` reverse of bat_to_below_spazer
- **Source state:** `scratch/post_ice_west_to_below_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not close parent rr-dbu.8 (stack incomplete)

### Probe pin
- Dual GREEN hop8: room=0xA408 pose=82 x=472 y=393 frames=272 ×2
