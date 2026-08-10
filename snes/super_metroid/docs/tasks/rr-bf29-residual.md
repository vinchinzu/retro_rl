## Residual — rr-bf29 Pure Ice Snake → Tutorial return (K5 hop 1)

### Result
GREEN — pure dual exact **2386f** ×2. Snake mid-right → Tutorial. Not continuous.
No STATUS change. Parent **rr-dbu.8** stays open (Alpha PB stack incomplete).

### Files changed
- `routes/kpdr/ice/snake_to_tutorial.py` — drop + multi-attempt 2WJ climb + top door
- `routes/kpdr/ice/geometry.py` — Tutorial door / drop settle constants
- `routes/kpdr/ice/__init__.py` — export `play_ice_snake_to_tutorial`
- `routes/kpdr/registry.py` / `scripts/probe/kpdr.py` — `ice_snake_to_tutorial` segment
- `routes/kpdr/k5/__init__.py` — hop map mark GREEN
- `source_states.py` / `docs/SOURCE_STATES.md` — `post_ice_snake_to_tutorial_pure`
- `tests/test_k4_ice_scaffold.py` — registry + door constant

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_ice_scaffold.py -q
# → passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-snake-to-tutorial \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_to_snake_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_snake_to_tutorial_pure.state
# → GREEN room=0xA865 xy=(39,127) pose=81 frames=2386 (×2 exact dual)
```

### Acceptance
- [x] Pure Snake mid-right → Tutorial dual green from `post_ice_to_snake_pure`
- [x] No freeze thrash RLE (morph-tunnel drop open-loop + platform climb helper)
- [x] Export `scratch/post_ice_snake_to_tutorial_pure.state` for next hop
- [ ] Full K5 stack / continuous / STATUS (parent rr-dbu.8)

### Residual risks
1. Climb is multi-attempt (L3 pin-sensitive post-Ice); first pass often needs retry.
2. Tutorial settle pin ~(39,127) p81 may be slightly airborne (vy residual) — next hop should settle/unmorph.
3. Tutorial → Gate not started.

### Next action (required)
- **Next card ID:** `rr-81ek` Pure Ice Tutorial → Gate return (K5 hop 2)
- **One change:** pure controller Tutorial → Ice Gate from `post_ice_snake_to_tutorial_pure`
- **Source state:** `scratch/post_ice_snake_to_tutorial_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not invent Moat hops

### Probe pin
- Dual GREEN: room=0xA865 pose=81 x=39 y=127 frames=2386 ×2
  source=`post_ice_to_snake_pure` items via handoff `0x3105`
