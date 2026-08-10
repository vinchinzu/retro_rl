## Residual — rr-e5i6 Pure Ice Gate → Business return (K5 hop 3)

### Result
GREEN — pure dual exact **879f** ×2. Gate mid-top → Business Super lip.
Not continuous. No STATUS change. Parent **rr-dbu.8** stays open (Alpha PB stack incomplete).

### Files changed
- `routes/kpdr/ice/gate_to_business.py` — hybrid cleaned RLE drop/tunnel + door pressure
- `routes/kpdr/data/ice_gate_to_business_rle.json` — cleaned human mid→Super (tunnel thrash trimmed)
- `routes/kpdr/ice/geometry.py` — Gate mid-top / tunnel / Super door / RLE load
- `routes/kpdr/ice/__init__.py` — export `play_ice_gate_to_business`
- `routes/kpdr/registry.py` / `scripts/probe/kpdr.py` — `ice_gate_to_business` segment
- `routes/kpdr/k5/__init__.py` — hop map mark GREEN
- `source_states.py` / `docs/SOURCE_STATES.md` — `post_ice_gate_to_business_pure`
- `tests/test_k4_ice_scaffold.py` — registry + RLE load

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_ice_scaffold.py -q
# → 7 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-gate-to-business \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_tutorial_to_gate_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_gate_to_business_pure.state
# → GREEN room=0xA7DE xy=(41,907) pose=25 frames=879 (×2 exact dual)
```

### Acceptance
- [x] Pure Gate mid-top → Business Super dual green from `post_ice_tutorial_to_gate_pure`
- [x] Accept mid-top band ~(807,131) not Tutorial door lip
- [x] Export `scratch/post_ice_gate_to_business_pure.state` for next hop
- [ ] Full K5 stack / continuous / STATUS (parent rr-dbu.8)

### Residual risks
1. Super lip settle pose 25 (turn residual) — Business→Warehouse climb must accept Super band / floor fall recover.
2. Cleaned RLE is pin-sensitive on mid-top morph drop column x~870–900.
3. Super door assumed open from outbound Ice path (true on this pure stack).

### Next action (required)
- **Next card:** Pure Business → Warehouse return (K5 hop 4) — re-verify `play_business_to_warehouse` from Super lip
- **One change:** pure controller Business Super/floor → Warehouse elev from `post_ice_gate_to_business_pure`
- **Source state:** `scratch/post_ice_gate_to_business_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not invent Moat hops

### Probe pin
- Dual GREEN: room=0xA7DE pose=25 x=41 y=907 frames=879 ×2
  source=`post_ice_tutorial_to_gate_pure` items `0x3105`
