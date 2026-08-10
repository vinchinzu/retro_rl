## Residual — rr-68ib Pure East Tunnel → Glass return (K5 hop 6)

### Result
GREEN — pure one-hop dual GREEN East Tunnel `0xCF80` → Glass Tunnel `0xCEFB`
(**253f** ×2 exact). Not continuous. No STATUS change.

### Files changed
- `routes/kpdr/k5/east_to_glass.py` — reverse of glass_to_east; uncrouch p26 + LEFT door
- `routes/kpdr/k5/geometry.py` — EAST_GLASS_DOOR_X / EAST_TO_GLASS_* constants
- `routes/kpdr/k5/__init__.py` — export + hop map checkmark
- `routes/kpdr/registry.py` / `__init__.py` / `scripts/probe/kpdr.py` — segment wiring
- `source_states.py` + `docs/SOURCE_STATES.md` — handoff fingerprint
- export: `scratch/post_ice_east_to_glass_pure.state` (+ dual)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure east-to-glass \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_warehouse_to_east_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_east_to_glass_pure.state
# → GREEN room=0xCEFB xy=(216,395) pose=12 frames=253 (×2 exact dual)
```

### Acceptance
- [x] Pure dual green East → Glass from post-warehouse-east crouch pin
- [x] Accept East crouch residual ~(216,364) p26 (uncrouch before LEFT)
- [x] Export handoff for Glass → West reverse
- [ ] Full K5 stack through Alpha PB (parent rr-dbu.8)
- [ ] Continuous / STATUS (planner only)

### Residual risks
1. Glass pin ~(216,395) p12 mid-floor — next reverse hop LEFT to West Tunnel.
2. Fixed settle 260f; optimizable but dual-stable.
3. Parent stack still open through Glass→West→Below→Bat→Red→Alpha PB.

### Next action (required)
- **Next card:** Pure Glass Tunnel → West Tunnel return (K5 hop 7) reverse of west_to_glass
- **One change:** pure Glass `0xCEFB` → West `0xCF54` reverse of west_to_glass
- **Source state:** `scratch/post_ice_east_to_glass_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not close parent rr-dbu.8 (stack incomplete)

### Probe pin
- Dual GREEN hop6: room=0xCEFB pose=12 x=216 y=395 frames=253 ×2
