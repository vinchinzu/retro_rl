## Residual — rr-0ue1 Pure Bat Room → Red Tower return (K5 hop 11)

### Result
GREEN — pure one-hop dual GREEN Bat Room `0xA3DD` → Red Tower `0xA253`
(**718f** ×2 exact). Not continuous. No STATUS change.

### Files changed
- `routes/kpdr/k5/bat_to_red.py` — LEFT platform reverse of bat_to_below + left door into Red bottom
- `routes/kpdr/k5/geometry.py` — BAT_TO_RED_* timing constants
- `routes/kpdr/k5/__init__.py` — export + hop map checkmark
- `routes/kpdr/registry.py` / `__init__.py` / `scripts/probe/kpdr.py` — segment wiring
- `source_states.py` + `docs/SOURCE_STATES.md` — handoff fingerprint (also pose 42 on Bat source reload)
- export: `scratch/post_ice_bat_to_red_pure.state` (+ dual)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bat-to-red \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_below_to_bat_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state
# → GREEN room=0xA253 xy=(216,2443) pose=10 frames=718 (×2 exact dual)
```

### Acceptance
- [x] Pure dual green Bat → Red from post-below-to-bat right high sill pin
- [x] Reverse of red_tower_to_bat bottom exit (LEFT platforms into Red bottom)
- [x] Export handoff for Red → Hellway climb
- [ ] Full K5 stack through Alpha PB (parent rr-dbu.8)
- [ ] Continuous / STATUS (planner only)

### Residual risks
1. Red pin ~(216,2443) p10 bottom floor — next hop climbs ~7k human frames to Hellway.
2. Platform chain thrash-tolerant low-ledge hop; optimizable vs tape.
3. Parent stack still open through Red→Hellway→Caterpillar→Alpha PB.

### Next action (required)
- **Next card:** **rr-av5s** Pure Red Tower → Hellway return (K5 hop 12)
- **One change:** pure Red `0xA253` bottom → Hellway `0xA2F7` climb reverse of red_tower_to_bat descent
- **Source state:** `scratch/post_ice_bat_to_red_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not close parent rr-dbu.8 (stack incomplete)

### Probe pin
- Dual GREEN hop11: room=0xA253 pose=10 x=216 y=2443 frames=718 ×2
