## Residual — rr-85c4 Pure Glass Tunnel → West Tunnel return (K5 hop 7)

### Result
GREEN — pure one-hop dual GREEN Glass Tunnel `0xCEFB` → West Tunnel `0xCF54`
(**211f** ×2 exact). Not continuous. No STATUS change.

### Files changed
- `routes/kpdr/k5/glass_to_west.py` — reverse of west_to_glass; LEFT run_shoot_exit
- `routes/kpdr/k5/geometry.py` — GLASS_TO_WEST_* / GLASS_WEST_DOOR_X constants
- `routes/kpdr/k5/__init__.py` — export + hop map checkmark
- `routes/kpdr/registry.py` / `__init__.py` / `scripts/probe/kpdr.py` — segment wiring
- `source_states.py` + `docs/SOURCE_STATES.md` — handoff fingerprint
- export: `scratch/post_ice_glass_to_west_pure.state` (+ dual)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure glass-to-west \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_east_to_glass_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_glass_to_west_pure.state
# → GREEN room=0xCF54 xy=(216,139) pose=10 frames=211 (×2 exact dual)
```

### Acceptance
- [x] Pure dual green Glass → West from post-east-to-glass mid-floor pin
- [x] Reverse of west_to_glass (LEFT mirror of RIGHT run/shoot/spin)
- [x] Export handoff for West → Below reverse
- [ ] Full K5 stack through Alpha PB (parent rr-dbu.8)
- [ ] Continuous / STATUS (planner only)

### Residual risks
1. West pin ~(216,139) p10 mid-floor — next reverse hop LEFT to Below Spazer.
2. Fixed settle 260f; optimizable but dual-stable.
3. Parent stack still open through West→Below→Bat→Red→Alpha PB.

### Next action (required)
- **Next card:** Pure West Tunnel → Below Spazer return (K5 hop 8) reverse of below_spazer_to_west
- **One change:** pure West `0xCF54` → Below `0xA408` reverse of below_spazer_to_west
- **Source state:** `scratch/post_ice_glass_to_west_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not close parent rr-dbu.8 (stack incomplete)

### Probe pin
- Dual GREEN hop7: room=0xCF54 pose=10 x=216 y=139 frames=211 ×2
