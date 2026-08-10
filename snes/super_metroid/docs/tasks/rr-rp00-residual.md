## Residual — rr-rp00 Pure Below Spazer → Bat Room return (K5 hop 9)

### Result
GREEN — pure one-hop dual GREEN Below Spazer `0xA408` → Bat Room `0xA3DD`
(**485f** ×2 exact). Not continuous. No STATUS change.

### Files changed
- `routes/kpdr/k5/below_to_bat.py` — reverse of bat_to_below_spazer door; LEFT floor runner
- `routes/kpdr/k5/geometry.py` — BELOW_TO_BAT_* / BELOW_BAT_DOOR_X constants
- `routes/kpdr/k5/__init__.py` — export + hop map checkmark
- `routes/kpdr/registry.py` / `__init__.py` / `scripts/probe/kpdr.py` — segment wiring
- `source_states.py` + `docs/SOURCE_STATES.md` — handoff fingerprint
- export: `scratch/post_ice_below_to_bat_pure.state` (+ dual)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure below-to-bat \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_west_to_below_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_below_to_bat_pure.state
# → GREEN room=0xA3DD xy=(472,139) pose=12 frames=485 (×2 exact dual)
```

### Acceptance
- [x] Pure dual green Below → Bat from post-west-to-below right-floor pin
- [x] Reverse of bat_to_below_spazer (LEFT floor path; Spazer already held)
- [x] Export handoff for Bat → Red reverse
- [ ] Full K5 stack through Alpha PB (parent rr-dbu.8)
- [ ] Continuous / STATUS (planner only)

### Residual risks
1. Bat pin ~(472,139) p12 right high sill — next reverse hop LEFT platforms to Red.
2. Floor runner multi-attempt thrash-tolerant (Cacatac / water); optimizable.
3. Parent stack still open through Bat→Red→Hellway→Alpha PB.

### Next action (required)
- **Next card:** **rr-0ue1** Pure Bat Room → Red Tower return (K5 hop 11)
- **One change:** pure Bat `0xA3DD` → Red `0xA253` reverse of red_tower_to_bat
- **Source state:** `scratch/post_ice_below_to_bat_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not close parent rr-dbu.8 (stack incomplete)

### Probe pin
- Dual GREEN hop9: room=0xA3DD pose=12 x=472 y=139 frames=485 ×2
