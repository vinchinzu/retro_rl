## Residual — rr-bw2w Pure Warehouse → East Tunnel return (K5 hop 5)

### Result
GREEN — pure one-hop dual GREEN Warehouse elev `0xA6A1` → East Tunnel `0xCF80`
(**285f** ×2 exact; third re-run also 285f). Not continuous. No STATUS change.

### Files changed
- `routes/kpdr/k5/warehouse_to_east.py` — reverse of east_to_warehouse; elev-band unmorph + LEFT door
- `routes/kpdr/k5/geometry.py` — room-prefixed WH/EAST constants
- `routes/kpdr/k5/__init__.py` — export + hop map checkmark
- `routes/kpdr/registry.py` / `__init__.py` / `scripts/probe/kpdr.py` — segment wiring
- `source_states.py` + `docs/SOURCE_STATES.md` — handoff fingerprint
- export: `scratch/post_ice_warehouse_to_east_pure.state` (+ dual)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure warehouse-to-east \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_business_to_warehouse_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_warehouse_to_east_pure.state
# → GREEN room=0xCF80 xy=(216,364) pose=26 frames=285 (×2 exact dual)
```

### Acceptance
- [x] Pure dual green Warehouse → East from post-business-warehouse elev band
- [x] Accept upper-left Warehouse elev band source (~37,139 p138)
- [x] Export handoff for East → Glass reverse
- [ ] Full K5 stack through Alpha PB (parent rr-dbu.8)
- [ ] Continuous / STATUS (planner only)

### Residual risks
1. East pin ~(216,364) p26 crouch — next reverse hop should unmorph before LEFT to Glass.
2. Multi-screen East settle fixed 360f; optimizable but dual-stable.
3. Parent stack still open through Glass→West→Below→Bat→Red→Alpha PB.

### Next action (required)
- **Next card:** `rr-68ib` Pure East Tunnel → Glass return (K5 hop 6)
- **One change:** pure East `0xCF80` → Glass `0xCEFB` reverse of glass_to_east
- **Source state:** `scratch/post_ice_warehouse_to_east_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not close parent rr-dbu.8 (stack incomplete)

### Probe pin
- Dual GREEN hop5: room=0xCF80 pose=26 x=216 y=364 frames=285 ×2
